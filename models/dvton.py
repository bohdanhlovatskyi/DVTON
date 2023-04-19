import os
import numpy as np

import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import (
    UNet2DConditionModel, 
    AutoencoderKL, 
    PNDMScheduler, 
    ControlNetModel
)

from transformers import (
    CLIPTokenizer, 
    CLIPTextModel, 
    CLIPFeatureExtractor,
    CLIPProcessor, 
    CLIPVisionModel
)

from PIL import Image
from typing import Optional
from diffusers.utils import load_image, PIL_INTERPOLATION

class DiffusionBasedVTON(nn.Module):
    
    def __init__(self, wroot: str, controlnet: nn.Module, train: bool = True):
        super().__init__()
        
        self.wroot: str = wroot
        self.num_images_per_prompt: int = 1
        self.guidance_scale: float = 7.5
        self.controlnet_conditioning_scale: float = 1.0
        self.projection_dim: int = 768 # originally 512
        self.vision_embed_dim: int = 768

        self.features_extractor = CLIPFeatureExtractor.from_pretrained(
            os.path.join(self.wroot, "feature_extractor"))
        
        self.clip_vision = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch32")

        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.layer_norm = nn.LayerNorm(self.vision_embed_dim)

        self.vae = AutoencoderKL.from_pretrained(
            os.path.join(self.wroot, "vae"))
        
        self.unet = UNet2DConditionModel.from_pretrained(
            os.path.join(self.wroot, "unet"))
        
        self.scheduler = PNDMScheduler.from_pretrained(
            os.path.join(self.wroot, "scheduler"))

        self.controlnet = controlnet
        
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        if train:
            self.set_grads()

    def set_grads(self):
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.controlnet.requires_grad_(False)
        self.clip_vision.requires_grad_(False)

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.permute(0, 2, 3, 1).float()
        
        return image
        
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents,
                 height // self.vae_scale_factor,
                 width // self.vae_scale_factor)
        
        latents = torch.randn(shape)
        latents = latents * self.scheduler.init_noise_sigma

        return latents
        
    def prepare_mask_latents(self, mask_image, batch_size, height, width, dtype, device, do_classifier_free_guidance):
        mask_image = F.interpolate(
            mask_image, size=(height // self.vae_scale_factor, width // self.vae_scale_factor))
        mask_image = torch.cat([mask_image] * 2) if do_classifier_free_guidance else mask_image
        return mask_image
        
    def prepare_masked_image_latents(
        self, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        
        masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator=generator)
        masked_image_latents = self.vae.config.scaling_factor * masked_image_latents
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )

        return masked_image_latents
        
    def forward(
        self, 
        image: torch.Tensor,
        mask: torch.Tensor,
        cn: torch.Tensor,
        prompt: torch.Tensor, 
        negative_prompt: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20
    ):
        '''
        img BS 3 H W
        msk BS 1 H W 
        cn  BS 3 H W 
        '''
        
        BS, _, H, W = image.shape

        masked_image = image * (mask < 0.5)
        
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        
        do_classifier_free_guidance = self.guidance_scale > 1.0
    
        # Originally prompt embeddings were: [2 * BS, 77, 768]

        # [BS, 50, 768]
        with torch.no_grad():
            pos_prompt_embeds = self.clip_vision(**{"pixel_values": prompt}).last_hidden_state
            neg_prompt_embeds = self.clip_vision(
                **{"pixel_values": negative_prompt if negative_prompt is not None else torch.zeros_like(prompt)}
            ).last_hidden_state

        # TODO: do we really need this one ? 
        pos_prompt_embeds = self.layer_norm(pos_prompt_embeds)
        neg_prompt_embeds = self.layer_norm(neg_prompt_embeds)

        pos_prompt_embeds = self.visual_projection(pos_prompt_embeds)
        neg_prompt_embeds = self.visual_projection(neg_prompt_embeds)

        pos_prompt_embeds = pos_prompt_embeds / pos_prompt_embeds.norm(dim=-1, keepdim=True)
        neg_prompt_embeds = neg_prompt_embeds / neg_prompt_embeds.norm(dim=-1, keepdim=True)

        # [2 * BS, 50, 768]
        prompt_embeds = torch.cat([pos_prompt_embeds, neg_prompt_embeds], axis=0)
        
        # [BS, 4, 64, 64]
        shape = (BS, self.vae.config.latent_channels,
                 H // self.vae_scale_factor,
                 W // self.vae_scale_factor)
        
        # [2 * BS, 1, 64, 64]
        with torch.no_grad(): 
            latents = torch.randn(shape)
            latents = latents.type_as(prompt_embeds)
            latents = latents * self.scheduler.init_noise_sigma

            mask_image_latents = self.prepare_mask_latents(
                mask,
                BS * self.num_images_per_prompt,
                H, W, 
                None, # dtype
                None, # device
                do_classifier_free_guidance,
            )

            # [2 * BS, 4, 64, 64]
            masked_image_latents = self.prepare_masked_image_latents(
                masked_image,
                BS * self.num_images_per_prompt,
                H, W,
                None, # dtype
                None, # device
                None, # generator
                do_classifier_free_guidance,
            )

            if do_classifier_free_guidance:
                cn = torch.cat([cn] * 2)
        
        for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance

            with torch.no_grad():
                # [2 * BS, 4, 64, 64]
                non_inpainting_latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )

                non_inpainting_latent_model_input = self.scheduler.scale_model_input(
                    non_inpainting_latent_model_input, t
                )

                inpainting_latent_model_input = torch.cat(
                    [non_inpainting_latent_model_input, mask_image_latents, masked_image_latents], dim=1
                )

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    non_inpainting_latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_cond=cn,
                    return_dict=False,
                )

                down_block_res_samples = [
                    down_block_res_sample * self.controlnet_conditioning_scale
                    for down_block_res_sample in down_block_res_samples
                ]
                mid_block_res_sample *= self.controlnet_conditioning_scale

                # predict the noise residual
            noise_pred = self.unet(
                inpainting_latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        image = self.decode_latents(latents)
        
        return image

if __name__  == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    from controlnet_aux import OpenposeDetector

    device = "cuda"

    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    cn = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose")

    image = load_image(
        "https://github.com/CompVis/latent-diffusion/raw/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    ).resize((512, 512))

    mask_image = load_image(
        "https://github.com/CompVis/latent-diffusion/raw/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
    ).resize((512, 512), PIL.Image.NEAREST)

    prompt_image = load_image(
        "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/bench2.png"
    ).resize((512, 512))
        
    controlnet_conditioning_image = openpose(image)
    
    dvton = DiffusionBasedVTON(
        "/mnt/vol_c/projects/3d_on_2d/weights/deliberate-inpaint",
        cn,
    ).to(device)

    timage = process_image(image).to(device)
    tmask = process_mask(mask_image).to(device)
    tcond = process_cond(controlnet_conditioning_image).to(device)
    tprompt = process_prompt(prompt_image).to(device)

    res = dvton(timage, tmask, tcond, tprompt)

    ires = (res * 255).detach().numpy().round().astype("uint8")
    imgs = [Image.fromarray(img) for img in ires]

    imgs[0].save("res.png")
