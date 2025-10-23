import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataset import YoloDetectionDataset


def collate_fn(batch):
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)



class YoloDataModule(pl.LightningDataModule):
    """
    Lightning-compatible DataModule for Ultralytics-style datasets
    using YAML paths and YOLO TXT label files.
    """

    def __init__(self, data_yaml, batch_size=4, num_workers=4,
                 train_tfms=None, val_tfms=None, img_size=640):
        super().__init__()
        self.data_yaml = data_yaml
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_tfms = train_tfms
        self.val_tfms = val_tfms
        self.img_size = img_size

    def setup(self, stage=None):
        # Import inside to avoid circular import
        from dataset import YoloDetectionDataset
        self.train_dataset = YoloDetectionDataset(
            self.data_yaml, split="train", img_size=self.img_size, transform=self.train_tfms
        )
        self.val_dataset = YoloDetectionDataset(
            self.data_yaml, split="val", img_size=self.img_size, transform=self.val_tfms
        )
        self.test_data = YoloDetectionDataset(
            self.data_yaml, split="val", img_size=self.img_size, transform=self.val_tfms
        )
        

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )