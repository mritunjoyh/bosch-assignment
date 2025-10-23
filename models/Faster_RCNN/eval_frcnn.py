import os
import torch
import warnings
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
import pandas as pd
import json

from dataloader import YoloDataModule
from models.faster_rcnn_pytorch_lightning import CustomFasterRCNN

warnings.filterwarnings("ignore", category=UserWarning)
os.system(" rm -rf models/Faster_RCNN/output/bdd_val_*")

DATA_YAML = "../../cfg/bdd_custom.yaml"  # your dataset config
CKPT_PATH = "checkpoints/custom_trained/last.ckpt"  # trained model checkpoint
BATCH_SIZE = 2
NUM_WORKERS = 0
NUM_CLASSES = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_RESULTS = True  # save mAP metrics

val_tfms = A.Compose([
    A.Resize(640, 640),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

dm = YoloDataModule(
    data_yaml=DATA_YAML,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    train_tfms=val_tfms,
    val_tfms=val_tfms,
)

dm.setup("validate")

print(f"📦 Loading model from checkpoint: {CKPT_PATH}")

LR = 1e-5
WEIGHT_DECAY = 5e-4

model = CustomFasterRCNN(
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    num_classes=NUM_CLASSES,
    batch_size=BATCH_SIZE,
    score_threshold=0.5,
    nms_iou_threshold=0.1,
)

if os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print("Loaded weights from Lightning checkpoint.")
    else:
        print("Checkpoint missing 'state_dict' — using initialized weights.")
else:
    print(f"Checkpoint not found: {CKPT_PATH}")

model = model.to(DEVICE)
model.eval()

metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)

val_loader = dm.val_dataloader()
print(f"Evaluating on {len(val_loader.dataset)} validation images...")

with torch.no_grad():
    for images, targets in tqdm(val_loader, total=len(val_loader)):
        images = [img.to(DEVICE) for img in images]

        targets = [
            {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in t.items()}
            for t in targets
        ]

        preds = model(images)

        preds = [{k: v.detach().cpu() for k, v in p.items()} for p in preds]
        targets = [
            {k: v.detach().cpu() for k, v in t.items() if isinstance(v, torch.Tensor)}
            for t in targets
        ]

        metric.update(preds, targets)

results = metric.compute()

print("\nEvaluation Results (COCO-style mAP metrics):")
for k, v in results.items():
    if torch.is_tensor(v):
        if v.numel() == 1:
            v = v.item()
        else:
            v = v.tolist()
    print(f"{k}: {v}")


if SAVE_RESULTS:
    os.makedirs("eval_results", exist_ok=True)

    def tensor_to_json_compatible(v):
        if torch.is_tensor(v):
            if v.numel() == 1:
                return float(v.item())
            else:
                return v.tolist()
        elif isinstance(v, (int, float)):
            return float(v)
        elif isinstance(v, (list, tuple)):
            return list(v)
        else:
            return str(v)

    json_path = os.path.join("eval_results", "fasterrcnn_eval_results.json")
    with open(json_path, "w") as jf:
        json.dump({k: tensor_to_json_compatible(v) for k, v in results.items()}, jf, indent=4)


    if "map_per_class" in results:
        map_per_class = results["map_per_class"].tolist()
        df = pd.DataFrame({"class_id": list(range(len(map_per_class))), "AP": map_per_class})
        csv_path = os.path.join("eval_results", "fasterrcnn_per_class_ap.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n📝 Per-class AP saved to: {csv_path}")

    print(f"Overall metrics saved to: {json_path}")

print("\nEvaluation complete!")
