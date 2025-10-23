import os
import sys
sys.path.append("ultralytics/")
from ultralytics import YOLO
import torch
import numpy as np
import cv2
import random
from pathlib import Path
import matplotlib.pyplot as plt

MODEL_PATH = "runs/detect/train/weights/best.pt"
DATA_YAML = "../../cfg/bdd_custom.yaml"
CONF_THRES = 0.25
IOU_THRES = 0.5
NUM_VISUALS = 10 

model = YOLO(MODEL_PATH)
print("Model loaded successfully")

metrics = model.val(
    data=DATA_YAML,
    imgsz=640,
    conf=CONF_THRES,
    iou=IOU_THRES,
    save_json=True,   
    save_txt=False,
    save_hybrid=False,
    verbose=True,
)

print("\nEvaluation Summary:")
print(f"mAP@0.5:      {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95:  {metrics.box.map:.4f}")
print(f"Precision:     {metrics.box.p.mean():.4f}")
print(f"Recall:        {metrics.box.r.mean():.4f}")