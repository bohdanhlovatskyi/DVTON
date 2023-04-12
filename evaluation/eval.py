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

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', default='LPIPS')
    parser.add_argument('--predict_dir', default='../external/HR-VITON/Output')
    parser.add_argument('--ground_truth_dir', default='../data/hrviton/test/image')
    parser.add_argument('--resolution', type=int, default=1024)
    opt = parser.parse_args()

    return opt

ToTensor = Transforms.ToTensor()

@torch.no_grad()
def eval(opt):
    pred_list = sorted(os.listdir(opt.predict_dir))
    gt_list = sorted(os.listdir(opt.ground_truth_dir))

    ssim = StructuralSimilarityIndexMeasure()
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex')
    kid = KernelInceptionDistance(subset_size=2)
    fid = FrechetInceptionDistance()

    for i, _ in tqdm(enumerate(pred_list)):
        print(pred_list[i], gt_list[i])
        pred_img = np.array(
            Image.open(os.path.join(opt.predict_dir, pred_list[i]))
        )

        gt_img = np.array(
            Image.open(os.path.join(opt.ground_truth_dir, gt_list[i]))
        )

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(20, 16))
        ax[0].imshow(pred_img)
        ax[1].imshow(gt_img)
        plt.savefig(f"test/test_{i}.png")

        pred_img, gt_img = ToTensor(pred_img)[None], ToTensor(gt_img)[None]
        
        ssim.update(pred_img, gt_img)
        kid.update((gt_img * 255).to(torch.uint8), real=True)
        kid.update((pred_img * 255).to(torch.uint8), real=False)

        fid.update((gt_img * 255).to(torch.uint8), real=True)
        fid.update((pred_img * 255).to(torch.uint8), real=False)

        lpips.update((gt_img * 2) - 1, (pred_img * 2) - 1)

        if i == 10:
            break

    print("average ssim: ", ssim.compute().item())
    print("average kid: ", kid.compute()[0].item())
    print("average fid: ", fid.compute().item())
    print("average lpips: ", lpips.compute().item())


if __name__ == "__main__":
    opt = get_opt()
    eval(opt)
