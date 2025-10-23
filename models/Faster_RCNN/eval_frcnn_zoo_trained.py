import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import yaml
DATA_YAML = "../../cfg/bdd_custom.yaml"
with open(DATA_YAML, "r") as f:
    cfg = yaml.safe_load(f)
base_path = cfg.get("path", "")
val_path = cfg.get("val", "")
os.system(" rm -rf models/Faster_RCNN/output/bdd_val_*")
IMG_DIR = os.path.join(base_path, val_path)
LBL_DIR = IMG_DIR.replace("images", "labels")

CKPT_PATH = "checkpoints/pretrained/finetune_bdd.pth"

CLASS_NAMES = [
    "person", "car", "bus", "truck", "traffic light",
    "traffic sign", "rider", "bike", "motor", "train"
]
IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.5

device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES)
cfg.MODEL.WEIGHTS = CKPT_PATH
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONF_THRESHOLD
cfg.MODEL.DEVICE = device

model = build_model(cfg)
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
model.to(device)
model.eval()

print("Model loaded successfully!")

def yolo_to_bbox(line, width, height):
    """
    Converts YOLO format bounding box coordinates to standard bounding box coordinates.

    YOLO format represents bounding boxes as normalized values relative to the image dimensions,
    with the center of the bounding box specified along with its width and height.

    Args:
        line (str): A string containing YOLO format values (class, x_center, y_center, width, height),
                    separated by spaces.
        width (int or float): The width of the image.
        height (int or float): The height of the image.

    Returns:
        tuple: A tuple containing:
            - int: The class label as an integer.
            - list: A list of four float values representing the bounding box in the format 
                    [x1, y1, x2, y2], where:
                    - (x1, y1) is the top-left corner of the bounding box.
                    - (x2, y2) is the bottom-right corner of the bounding box.
    """
    cls, x, y, w, h = map(float, line.strip().split())
    x1 = (x - w / 2) * width
    y1 = (y - h / 2) * height
    x2 = (x + w / 2) * width
    y2 = (y + h / 2) * height
    return int(cls), [x1, y1, x2, y2]


def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.
    Args:
        box1 (tuple): A tuple representing the first bounding box in the format 
                      (x_min, y_min, x_max, y_max).
        box2 (tuple): A tuple representing the second bounding box in the format 
                      (x_min, y_min, x_max, y_max).
    Returns:
        float: The IoU value, a number between 0.0 and 1.0. Returns 0.0 if there 
               is no overlap between the bounding boxes.
    """
    
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    union = (
        (box1[2] - box1[0]) * (box1[3] - box1[1]) +
        (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    )
    return inter / union


def compute_ap(rec, prec):
    """
    Compute the Average Precision (AP) given the recall and precision values.
    This function calculates the AP by first ensuring that the precision is 
    monotonically decreasing and then using the trapezoidal rule to integrate 
    the precision-recall curve.
    Args:
        rec (list or numpy.ndarray): Recall values, a list or array of floats 
            in the range [0, 1].
        prec (list or numpy.ndarray): Precision values, a list or array of floats 
            in the range [0, 1].
    Returns:
        float: The computed Average Precision (AP) value.
    """
    
    rec, prec = np.array(rec), np.array(prec)
    rec = np.concatenate(([0.0], rec, [1.0]))
    prec = np.concatenate(([0.0], prec, [0.0]))
    for i in range(prec.size - 1, 0, -1):
        prec[i - 1] = np.maximum(prec[i - 1], prec[i])
    i = np.where(rec[1:] != rec[:-1])[0]
    ap = np.sum((rec[i + 1] - rec[i]) * prec[i + 1])
    return ap

all_detections = {c: [] for c in range(len(CLASS_NAMES))}
all_annotations = {c: 0 for c in range(len(CLASS_NAMES))}

print("\nRunning inference and evaluation...")

with torch.no_grad():
    for img_file in tqdm(sorted(os.listdir(IMG_DIR))):
    # for img_file in tqdm(sorted(os.listdir(IMG_DIR))[:1000]):
        if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(IMG_DIR, img_file)
        lbl_path = os.path.join(LBL_DIR, os.path.splitext(img_file)[0] + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            continue
        height, width = img.shape[:2]

        img_tensor = torch.as_tensor(img[:, :, ::-1].copy(), dtype=torch.float32).permute(2, 0, 1)
        inputs = [{"image": img_tensor.to(device), "height": height, "width": width}]
        outputs = model(inputs)[0]

        boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        scores = outputs["instances"].scores.cpu().numpy()
        labels = outputs["instances"].pred_classes.cpu().numpy()

        for b, s, l in zip(boxes, scores, labels):
            if s < CONF_THRESHOLD:
                continue
            all_detections[l].append({
                "image_id": img_file,
                "bbox": b.tolist(),
                "score": s,
                "matched": False
            })

        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                lines = f.readlines()
            for line in lines:
                cls, bbox = yolo_to_bbox(line, width, height)
                all_annotations[cls] += 1
                for det in all_detections[cls]:
                    if det["image_id"] != img_file:
                        continue
                    iou = compute_iou(det["bbox"], bbox)
                    if iou >= IOU_THRESHOLD and not det["matched"]:
                        det["matched"] = True
                        break

print("\nCalculating metrics...")

aps = []
for c in range(len(CLASS_NAMES)):
    detections = sorted(all_detections[c], key=lambda x: -x["score"])
    TP, FP = [], []
    npos = all_annotations[c]
    if npos == 0:
        continue

    matched = set()
    for det in detections:
        if det["matched"]:
            TP.append(1)
            FP.append(0)
        else:
            TP.append(0)
            FP.append(1)

    TP = np.cumsum(TP)
    FP = np.cumsum(FP)
    rec = TP / max(npos, np.finfo(np.float64).eps)
    prec = TP / np.maximum(TP + FP, np.finfo(np.float64).eps)

    ap = compute_ap(rec, prec)
    aps.append(ap)
    print(f"{CLASS_NAMES[c]:20s} | AP@0.5 = {ap*100:.2f}%")

mAP = np.mean(aps) * 100 if aps else 0.0
