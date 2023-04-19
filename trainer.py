import os
import torch
import torchvision
import pytorch_lightning as pl

from diffusers import ControlNetModel

from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from datasets.vitonhd import VITONDataset
from models.dvton import DiffusionBasedVTON
from losses.perceptual import VGGPerceptualLoss

from deepspeed.ops.adam import FusedAdam

class DiffusionBasedVTONModule(pl.LightningModule):

    def __init__(self, model, perc_w: float = 1.0, mse_w: float = 0.05) -> None:
        super().__init__()
        self.model = model

        self.perc_loss = VGGPerceptualLoss()
        self.perc_w = perc_w
        self.mse_w = mse_w

        self.ssim = StructuralSimilarityIndexMeasure()
        self.lpips = LearnedPerceptualImagePatchSimilarity()
        self.fid = FrechetInceptionDistance()

    def forwrd(self, **x):
        return self.model(**x)

    def training_step(self, batch, idx):
        img, mask, cloth_prompt, cond = batch.values()           
        # denorm img 
        orig_img = ((img + 1) * 127.5)

        gen_res = self.model(img, mask, cond, cloth_prompt).permute(0, 3, 1, 2)
        assert orig_img.shape == gen_res.shape

        gen_res = gen_res.type_as(orig_img)
        perc_loss = self.perc_loss(gen_res, orig_img)
        mse_loss = torch.mean((gen_res - orig_img) ** 2)

        loss = self.perc_w * perc_loss + self.mse_w * mse_loss
        self.log_dict(
            {"train/total_loss": loss, "train/mse": mse_loss * self.mse_w, "trian/perc_loss": perc_loss},
            on_step=True,
            prog_bar=True
        )

        return loss

    def validation_step(self, batch, idx):
        img, mask, cloth_prompt, cond = batch.values()   
  
        # denorm img 
        orig_img = (img + 1) * 127.5

        gen_res = self.model(img, mask, cond, cloth_prompt).permute(0, 3, 1, 2)

        assert orig_img.shape == gen_res.shape

        gen_res = gen_res.type_as(orig_img)
        perc_loss = self.perc_loss(gen_res, orig_img)
        mse_loss = torch.mean((gen_res - orig_img) ** 2)

        loss = self.perc_w * perc_loss + self.mse_w * mse_loss

        self.ssim(gen_res, orig_img)
        self.lpips((gen_res / 127.5) - 1, (orig_img / 127.5) - 1)
        # self.fid.update(orig_img.to(torch.uint8), real=True)
        # self.fid.update(gen_res.to(torch.uint8), real=False)

        # "fid": self.fid.compute()
        self.log_dict(
            {"val_loss": loss, "ssim": self.ssim, "lpips": self.lpips}, 
            on_epoch=True,
            prog_bar=True
        )

        return loss

    def configure_optimizers(self):
        return FusedAdam(self.model.parameters())

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

    dataset = VITONDataset(height=256, width=256)
    train_set, val_set = torch.utils.data.random_split(dataset, [1800, 232])
    
    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=1, shuffle=True, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(
        val_set, batch_size=1, shuffle=False, pin_memory=True)

    cn = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose")
    dvton = DiffusionBasedVTON(
        "/mnt/vol_c/projects/3d_on_2d/weights/deliberate-inpaint", 
        cn
    )
  
    model = DiffusionBasedVTONModule(dvton)

    trainer = pl.Trainer(
        accelerator="gpu", 
        devices=1, 
        strategy="deepspeed_stage_3", 
        accumulate_grad_batches=4,
        callbacks=[
            ModelCheckpoint(monitor="val_loss"), 
        ], 
        logger=TensorBoardLogger(save_dir=os.getcwd(), version=1, name="lightning_logs"), 
        precision="16-mixed"
    )
    trainer.fit(model, train_dataloader, val_dataloader)
