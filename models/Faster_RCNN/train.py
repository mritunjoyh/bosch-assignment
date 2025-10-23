import os
import torch
import warnings
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataloader import YoloDataModule
from models.faster_rcnn_pytorch_lightning import CustomFasterRCNN

warnings.filterwarnings("ignore", category=UserWarning)

DATA_YAML = "../../cfg/bdd_custom.yaml"
IMG_DIR   = "../../dataset"
BATCH_SIZE = 2
EPOCHS = 50
NUM_WORKERS = 0
LR = 1e-5
WEIGHT_DECAY = 5e-4
NUM_CLASSES = 10 

pl.seed_everything(17)

train_tfms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.1),
    A.RandomBrightnessContrast(p=0.25),
    ToTensorV2(),
], bbox_params=A.BboxParams(
        format='pascal_voc',
        min_visibility=0.2,
        label_fields=['class_labels'],
        clip=True,  
))


val_tfms = A.Compose([
    A.Resize(640, 640),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

dm = YoloDataModule(
    data_yaml=DATA_YAML,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    train_tfms=train_tfms,
    val_tfms=val_tfms,
)

model = CustomFasterRCNN(
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    num_classes=NUM_CLASSES,
    batch_size=BATCH_SIZE,
    score_threshold=0.5,
    nms_iou_threshold=0.1,
)

logger = TensorBoardLogger(
    save_dir="runs",
    name="faster_rcnn_yolo",
    default_hp_metric=False,
)

checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join("runs", "faster_rcnn_yolo", "checkpoints"),
    filename="epoch_{epoch:02d}",     
    save_top_k=-1,                    
    every_n_epochs=1,                 
    save_last=True,                   
)

lr_monitor = LearningRateMonitor(logging_interval='step')

trainer = pl.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    max_epochs=EPOCHS,
    gradient_clip_val=0.1,
    accumulate_grad_batches=1,
    log_every_n_steps=50,
    check_val_every_n_epoch=1,
    logger=logger,
    callbacks=[checkpoint_callback, lr_monitor],
)

trainer.fit(model, dm)

if hasattr(model, "predict"):
    model.predict(images=None, dm=dm)

print(f"\nTraining complete! Logs saved to: {logger.log_dir}")
print(f"\nTo view in TensorBoard, run:\n   tensorboard --logdir runs/faster_rcnn_yolo --port 6006")
