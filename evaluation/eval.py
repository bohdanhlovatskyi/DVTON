import os
import sys
import argparse
import numpy as np

import torch
import torchvision.transforms as Transforms

from PIL import Image
from tqdm import tqdm

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class EvalDataset(Dataset):
    def __init__(self, predict_dir='../external/HR-VITON/Output', gt_dir='../data/hrviton/test/image', resolution=1024):
        self.predict_dir = predict_dir
        self.gt_dir = gt_dir
        self.resolution = resolution

        self.gt_images = list(np.sort(os.listdir(self.gt_dir)))
        self.gt_images = list(map(lambda path: os.path.join(self.gt_dir, path), self.gt_images))
        self.predict_images = list(np.sort(os.listdir(self.predict_dir)))
        self.predict_images = list(map(lambda path: os.path.join(self.predict_dir, path), self.predict_images))
    
    def __len__(self):
        return len(self.gt_images)
    
    def read_image(self, path):
        image = np.array(Image.open(path))
        image = Transforms.ToTensor()(image)
        return image

    def __getitem__(self, idx):
        return (self.read_image(self.predict_images[idx]), self.read_image(self.gt_images[idx]))


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', default='LPIPS')
    parser.add_argument('--predict_dir', default='../external/HR-VITON/Output')
    parser.add_argument('--ground_truth_dir', default='../data/hrviton/test/image')
    parser.add_argument('--resolution', type=int, default=1024)
    opt = parser.parse_args()

    return opt

@torch.no_grad()
def eval(opt):
    dataset = EvalDataset(predict_dir=opt.predict_dir, gt_dir=opt.ground_truth_dir)
    dataloader = DataLoader(dataset,
                            batch_size=16,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)

    ssim = StructuralSimilarityIndexMeasure()
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex')
    kid = KernelInceptionDistance(subset_size=100)
    fid = FrechetInceptionDistance()

    print(f"dataset length: {len(dataset)}")

    for _, sample in tqdm(enumerate(dataloader)):
        pred_img, gt_img = sample
        
        ssim.update(pred_img, gt_img)
        kid.update((gt_img * 255).to(torch.uint8), real=True)
        kid.update((pred_img * 255).to(torch.uint8), real=False)

        fid.update((gt_img * 255).to(torch.uint8), real=True)
        fid.update((pred_img * 255).to(torch.uint8), real=False)

        lpips.update((gt_img * 2) - 1, (pred_img * 2) - 1)

    print("average ssim: ", ssim.compute().item())
    print("average kid: ", kid.compute()[0].item() * 100)
    print("average fid: ", fid.compute().item())
    print("average lpips: ", lpips.compute().item())

if __name__ == "__main__":
    opt = get_opt()
    eval(opt)
