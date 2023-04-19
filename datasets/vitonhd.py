import os
import cv2
import PIL
import torch
import numpy as np

from torchvision import transforms
from PIL import Image

from transformers import CLIPProcessor
from diffusers.utils import PIL_INTERPOLATION
from controlnet_aux import OpenposeDetector

class VITONDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            root: str = "/mnt/vol_b/DVTON/data/processed_hrviton",
            width: int = 224,
            height: int = 224
    ) -> None:
        self.width: int = width
        self.height: int = height

        self.root = root
        mask_path = os.path.join(self.root, "masks")
        img_path = os.path.join(self.root, "images")
        cloth_path = os.path.join(self.root, "clothes")

        self.img_paths = sorted([os.path.join(img_path, p) for p in os.listdir(img_path)])
        self.cloth_paths = sorted([os.path.join(cloth_path, p) for p in os.listdir(cloth_path)])
        self.mask_paths = sorted([os.path.join(mask_path, p) for p in os.listdir(mask_path)])

        assert len(self.img_paths) == len(self.cloth_paths)
        assert len(self.cloth_paths) == len(self.mask_paths)

        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, mask_path, cloth_path = self.img_paths[idx], self.mask_paths[idx], self.cloth_paths[idx]

        img = Image.open(img_path).resize((self.height, self.width))
        cloth = Image.open(cloth_path).resize((self.height, self.width))
        mask = Image.fromarray(
            cv2.imread(mask_path, 0)
        ).convert("L").resize((self.height, self.width), PIL.Image.NEAREST)
        
        cond = self.openpose(img).resize((self.height, self.width))

        return {
            "img": self.process_image(img)[0], 
            "mask": self.process_mask(mask)[0], 
            "cloth": self.process_prompt(cloth)[0],
            "cond": self.process_cond(cond)[0]
        }

    def process_image(self, image: PIL.Image):
        image = [image]
        image = [np.array(i.convert("RGB"))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        
        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
        
        return image

    def process_mask(self, mask_image: PIL.Image):
        mask_image = [mask_image]
        
        mask_image = np.concatenate([np.array(m.convert("L"))[None, None, :] for m in mask_image], axis=0)
        mask_image = mask_image.astype(np.float32) / 255.0
        
        mask_image[mask_image < 0.5] = 0
        mask_image[mask_image >= 0.5] = 1
        mask_image = torch.from_numpy(mask_image)
        
        return mask_image

    def process_cond(self, controlnet_conditioning_image: PIL.Image):
        controlnet_conditioning_image = [controlnet_conditioning_image]
        
        controlnet_conditioning_image = [
                    np.array(i.resize((self.width, self.height), resample=PIL_INTERPOLATION["lanczos"]))[None, :]
                    for i in controlnet_conditioning_image
                ]
        controlnet_conditioning_image = np.concatenate(controlnet_conditioning_image, axis=0)
        controlnet_conditioning_image = np.array(controlnet_conditioning_image).astype(np.float32) / 255.0
        controlnet_conditioning_image = controlnet_conditioning_image.transpose(0, 3, 1, 2)
        controlnet_conditioning_image = torch.from_numpy(controlnet_conditioning_image)
        
        return controlnet_conditioning_image

    def process_prompt(self, prompt_image: PIL.Image):
        return self.processor(images=prompt_image, return_tensors="pt", padding=True)["pixel_values"]

if __name__ == "__main__":
    ds = VITONDataset()
    for k, v in ds[0].items():
        print(k, v.shape, type(v), v.device)
