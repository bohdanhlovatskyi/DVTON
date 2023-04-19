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
    CLIPFeatureExtractor
)

from PIL import Image
from typing import Optional
from diffusers.utils import load_image, PIL_INTERPOLATION, randn_tensor

class DiffusionBasedVTON(nn.Module):
    
    def __init__(self, wroot: str, controlnet: nn.Module, device: str = "cpu"):
        super().__init__()
        
        self.wroot: str = wroot
        self.device: str = device
        self.num_images_per_prompt: int = 1
        self.guidance_scale: float = 7.5
        self.controlnet_conditioning_scale: float = 1.0
        
        self.features_extractor = CLIPFeatureExtractor.from_pretrained(
            os.path.join(self.wroot, "feature_extractor"))
        
        self.text_encoder = CLIPTextModel.from_pretrained(
            os.path.join(self.wroot, "text_encoder"))
        
        self.tokenizer = CLIPTokenizer.from_pretrained(
            os.path.join(self.wroot, "tokenizer"))
        
        self.vae = AutoencoderKL.from_pretrained(
            os.path.join(self.wroot, "vae"))
        
        self.unet = UNet2DConditionModel.from_pretrained(
            os.path.join(self.wroot, "unet"))
        
        self.scheduler = PNDMScheduler.from_pretrained(
            os.path.join(self.wroot, "scheduler"))

        self.controlnet = controlnet
        
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        
    def _encode_prompt(
        self,
        prompt: list[str],
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        prompt_embeds = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        
        prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds
    
    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        
        return image
        
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents,
                 height // self.vae_scale_factor,
                 width // self.vae_scale_factor)
        
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = latents * self.scheduler.init_noise_sigma

        return latents
        
    def prepare_mask_latents(self, mask_image, batch_size, height, width, dtype, device, do_classifier_free_guidance):
        mask_image = F.interpolate(
            mask_image, size=(height // self.vae_scale_factor, width // self.vae_scale_factor))
        mask_image = mask_image.to(device=device, dtype=dtype)
        mask_image = torch.cat([mask_image] * 2) if do_classifier_free_guidance else mask_image
        return mask_image
        
    def prepare_masked_image_latents(
        self, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        
        masked_image = masked_image.to(device=device, dtype=dtype)
        masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator=generator)
        masked_image_latents = self.vae.config.scaling_factor * masked_image_latents
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        
        return masked_image_latents
        
    # TODO: preprocessing into dataset
    @torch.no_grad()
    def forward(
        self, 
        image: torch.Tensor,
        mask: torch.Tensor,
        cn: torch.Tensor,
        prompt: list[str], 
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 20
    ):
        '''
        img BS 3 H W
        msk BS 1 H W 
        cn  BS 3 H W 
        '''
        
        BS, _, H, W = image.shape
        
        masked_image = image * (mask < 0.5)
        
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        
        do_classifier_free_guidance = self.guidance_scale > 1.0
    
        # [2 * BS, 77, 768]
        prompt_embeds = self._encode_prompt(
            prompt,
            self.device,
            self.num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
        )
        
        print(f"{prompt_embeds.shape=}")
        
        # [BS, 4, 64, 64]
        latents = self.prepare_latents(
            BS * self.num_images_per_prompt,
            self.vae.config.latent_channels,
            H, W,
            prompt_embeds.dtype,
            self.device,
            None, # generator
            None, # latens
        )
        
        # [2 * BS, 1, 64, 64]
        mask_image_latents = self.prepare_mask_latents(
            mask,
            BS * self.num_images_per_prompt,
            H, W, 
            prompt_embeds.dtype,
            self.device,
            do_classifier_free_guidance,
        )
        
        # [2 * BS, 4, 64, 64]
        masked_image_latents = self.prepare_masked_image_latents(
            masked_image,
            BS * self.num_images_per_prompt,
            H, W,
            prompt_embeds.dtype,
            self.device,
            None, # generator
            do_classifier_free_guidance,
        )
        
        print(f"{masked_image_latents.shape=}")
        
        if do_classifier_free_guidance:
            cn = torch.cat([cn] * 2)
        
        for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
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

def process_image(image: PIL.Image):
    image = [image]
    image = [np.array(i.convert("RGB"))[None, :] for i in image]
    image = np.concatenate(image, axis=0)
    
    image = image.transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    
    return image

def process_mask(mask_image: PIL.Image):
    mask_image = [mask_image]
    
    mask_image = np.concatenate([np.array(m.convert("L"))[None, None, :] for m in mask_image], axis=0)
    mask_image = mask_image.astype(np.float32) / 255.0
    
    mask_image[mask_image < 0.5] = 0
    mask_image[mask_image >= 0.5] = 1
    mask_image = torch.from_numpy(mask_image)
    
    return mask_image

def process_cond(controlnet_conditioning_image: PIL.Image, width: int = 512, height: int = 512):
    controlnet_conditioning_image = [controlnet_conditioning_image]
    
    controlnet_conditioning_image = [
                np.array(i.resize((width, height), resample=PIL_INTERPOLATION["lanczos"]))[None, :]
                for i in controlnet_conditioning_image
            ]
    controlnet_conditioning_image = np.concatenate(controlnet_conditioning_image, axis=0)
    controlnet_conditioning_image = np.array(controlnet_conditioning_image).astype(np.float32) / 255.0
    controlnet_conditioning_image = controlnet_conditioning_image.transpose(0, 3, 1, 2)
    controlnet_conditioning_image = torch.from_numpy(controlnet_conditioning_image)
    
    return controlnet_conditioning_image

if __name__  == "__main__":
    from controlnet_aux import OpenposeDetector

    device = "cuda"

    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    cn = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose")

    image = load_image(
        "https://github.com/CompVis/latent-diffusion/raw/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    )

    mask_image = load_image(
        "https://github.com/CompVis/latent-diffusion/raw/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
    )
        
    controlnet_conditioning_image = openpose(image)
    
    dvton = DiffusionBasedVTON(
        "/mnt/vol_c/projects/3d_on_2d/weights/deliberate-inpaint",
        cn, 
        device="cuda"
    ).to(device)

    timage = process_image(image).to(device)
    tmask = process_mask(mask_image).to(device)
    tcond = process_cond(controlnet_conditioning_image).to(device)

    res = dvton(timage, tmask, tcond, ["alian dog"])

    ires = (res * 255).round().astype("uint8")
    imgs = [Image.fromarray(img) for img in ires]

    imgs[0].save("res.png")
