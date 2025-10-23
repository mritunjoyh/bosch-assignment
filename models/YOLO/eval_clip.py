import os
import sys
sys.path.append("ultralytics/")
from ultralytics import YOLO
import json
import glob
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import yaml
import numpy as np
import torch

try:
    import clip
except Exception:
    clip = None

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


import os
os.system("rm runs/clip_eval/preds_coco.json")
os.system("rm ../../dataset/test/annotations.json")
MODEL_PATH = "runs/detect/train/weights/best.pt"   # path to your trained YOLO
DATA_YAML = "../../cfg/bdd_custom.yaml"            # dataset yaml (must contain 'val' and 'names')
IMG_SIZE = 640
CONF_THRESH = 0.001   # low threshold to collect many preds for debug
USE_CLIP = True     # toggle: False => YOLO-only baseline, True => CLIP relabeling
CLIP_MODEL_NAME = "ViT-B/32"
SAVE_DIR = "runs/clip_eval"
RELABELED_PRED_DIR = os.path.join(SAVE_DIR, "preds_relabel")
os.makedirs(RELABELED_PRED_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading YOLO model:", MODEL_PATH)
model = YOLO(MODEL_PATH)
print("YOLO model loaded.")

with open(DATA_YAML, "r") as f:
    data = yaml.safe_load(f)

base_path = os.path.expanduser(data.get("path", ""))
val_rel = data.get("val")
if val_rel is None:
    raise RuntimeError("DATA_YAML must contain a 'val' entry.")
VAL_IMG_DIR = os.path.abspath(os.path.join(base_path, val_rel))
if not os.path.isdir(VAL_IMG_DIR):
    raise RuntimeError(f"No images found under resolved path: {VAL_IMG_DIR}")

NAMES = data.get("names")
if isinstance(NAMES, dict):
    NAMES = [NAMES[k] for k in sorted(NAMES.keys())]
NUM_CLASSES = len(NAMES)
print(f"Loaded {NUM_CLASSES} classes: {NAMES}")

val_dir = VAL_IMG_DIR
if os.path.basename(val_dir).lower() in ("images", "imgs"):
    images_dir = val_dir
    labels_dir = os.path.abspath(os.path.join(val_dir, "..", "labels"))
else:
    candidate_images = os.path.join(val_dir, "images")
    candidate_labels = os.path.join(val_dir, "labels")
    if os.path.isdir(candidate_images) and os.path.isdir(candidate_labels):
        images_dir = candidate_images
        labels_dir = candidate_labels
    else:
        images_dir = val_dir
        labels_dir = candidate_labels if os.path.isdir(candidate_labels) else os.path.join(os.path.dirname(val_dir), "labels")

if not os.path.isdir(images_dir):
    raise RuntimeError(f"Could not find images directory at {images_dir}")

if not os.path.isdir(labels_dir):
    raise RuntimeError(f"Could not find labels directory at {labels_dir}")

print("Images dir:", images_dir)
print("Labels dir:", labels_dir)

img_paths = sorted(list(Path(images_dir).rglob("*.jpg")) + list(Path(images_dir).rglob("*.jpeg")) + list(Path(images_dir).rglob("*.png")))
if len(img_paths) == 0:
    raise RuntimeError(f"No image files found under {images_dir}")

image_files = [str(p) for p in img_paths]
image_id_map = {os.path.splitext(os.path.basename(p))[0]: i for i, p in enumerate(image_files)}
print(f"Discovered {len(image_files)} images and built image_id_map.")

if USE_CLIP:
    if clip is None:
        raise RuntimeError("CLIP requested but python package 'clip' not available.")
    print("Loading CLIP model:", CLIP_MODEL_NAME)
    clip_model, clip_preprocess = clip.load(CLIP_MODEL_NAME, device=DEVICE)
    clip_model.eval()
    text_prompts = [f"a photo of a {c}" for c in NAMES]
    text_tokens = clip.tokenize(text_prompts).to(DEVICE)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    print("CLIP loaded.")

print("Running inference and saving per-image .txt predictions to:", RELABELED_PRED_DIR)
total_detections = 0
clip_max_sims = []

for img_path in tqdm(image_files, desc="Images"):
    base = os.path.splitext(os.path.basename(img_path))[0]
    results = model.predict(source=img_path, imgsz=IMG_SIZE, conf=CONF_THRESH, device=DEVICE, verbose=False)
    if len(results) == 0:
        open(os.path.join(RELABELED_PRED_DIR, f"{base}.txt"), "w").close()
        continue
    res = results[0]
    boxes = getattr(res, "boxes", None)
    if boxes is None or len(boxes) == 0:
        open(os.path.join(RELABELED_PRED_DIR, f"{base}.txt"), "w").close()
        continue

    lines = []
    if USE_CLIP:
        pil_img = Image.open(img_path).convert("RGB")
        W, H = pil_img.size

    for b in boxes:
        cx = cy = bw = bh = None
        try:
            xywhn = getattr(b, "xywhn", None)
            if xywhn is not None:
                arr = xywhn[0].cpu().numpy()
                cx, cy, bw, bh = arr.tolist()
        except Exception:
            cx = cy = bw = bh = None

        if cx is None:
            try:
                xywh = getattr(b, "xywh", None)
                if xywh is not None:
                    arr = xywh[0].cpu().numpy()
                    cx, cy, bw, bh = arr.tolist()
            except Exception:
                cx = cy = bw = bh = None

        if cx is None:
            try:
                xyxy = getattr(b, "xyxy", None)
                if xyxy is not None:
                    arr = xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = arr.tolist()
                    im = Image.open(img_path)
                    W, H = im.size
                    cx = (x1 + x2) / 2.0 / W
                    cy = (y1 + y2) / 2.0 / H
                    bw = (x2 - x1) / W
                    bh = (y2 - y1) / H
            except Exception:
                cx = cy = bw = bh = None

        if cx is None:
            continue

        try:
            cls = int(b.cls[0].cpu().item())
        except Exception:
            try:
                cls = int(getattr(b, "cls", -1))
            except Exception:
                cls = -1

        try:
            conf = float(b.conf[0].cpu().item())
        except Exception:
            try:
                conf = float(getattr(b, "conf", 1.0))
            except Exception:
                conf = 1.0

        if USE_CLIP:
            x1 = max(0, (cx - bw / 2.0) * W)
            y1 = max(0, (cy - bh / 2.0) * H)
            x2 = min(W - 1, (cx + bw / 2.0) * W)
            y2 = min(H - 1, (cy + bh / 2.0) * H)
            if x2 <= x1 or y2 <= y1:
                continue
            roi = pil_img.crop((x1, y1, x2, y2)).convert("RGB")
            roi_input = clip_preprocess(roi).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                img_feat = clip_model.encode_image(roi_input)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                sims = (img_feat @ text_features.T).squeeze(0)
                top_idx = int(torch.argmax(sims).cpu().item())
                top_sim = float(sims[top_idx].cpu().item())
            cls_clip = top_idx
            clip_score = (top_sim + 1.0) / 2.0
            clip_score = max(0.0, min(1.0, clip_score))

            alpha = 1.0  
            beta  = 0.2  

            yolo_score = conf
            clip_score = (top_sim + 1.0) / 2.0
            clip_score = max(0.0, min(1.0, clip_score))

            with torch.no_grad():
                all_sims = (img_feat @ text_features.T).squeeze(0)
                all_clip_scores = (all_sims + 1.0) / 2.0
                all_clip_scores = torch.clamp(all_clip_scores, 0.0, 1.0)

            combined_scores = alpha * yolo_score * torch.nn.functional.one_hot(
                torch.tensor(cls), num_classes=len(all_clip_scores)
            ).float().to(DEVICE) + beta * all_clip_scores

            best_class = int(torch.argmax(combined_scores).cpu().item())
            best_combined_score = float(combined_scores[best_class].cpu().item())

            if best_class != cls:
                cls = best_class
                conf = best_combined_score
            else:
                conf = yolo_score 

            clip_max_sims.append(float(top_sim))

        lines.append(f"{cls} {conf:.6f} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        total_detections += 1

    with open(os.path.join(RELABELED_PRED_DIR, f"{base}.txt"), "w") as wf:
        wf.write("\n".join(lines))

print(f"Finished inference. Total detections saved in .txt files: {total_detections}")

GT_JSON = os.path.join(os.path.dirname(images_dir), "annotations.json")
print("GT JSON path:", GT_JSON)

label_files = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))
if len(label_files) == 0:
    raise RuntimeError(f"No label files found under {labels_dir}")

if not os.path.exists(GT_JSON):
    print("Converting GT labels to COCO JSON...")
    categories = [{"id": i, "name": name} for i, name in enumerate(NAMES)]
    images_json = []
    annotations = []
    ann_id = 0

    for idx, img_path in enumerate(image_files):
        base = os.path.splitext(os.path.basename(img_path))[0]
        img_jpg = img_path
        try:
            im = Image.open(img_jpg)
        except Exception:
            continue
        w, h = im.size
        images_json.append({"id": idx, "file_name": os.path.basename(img_jpg), "width": w, "height": h})

        label_path = os.path.join(labels_dir, base + ".txt")
        if not os.path.exists(label_path):
            continue
        with open(label_path, "r") as lf:
            for line in lf:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    nums = [float(x) for x in parts]
                except Exception:
                    continue
                if len(nums) >= 5:
                    if len(nums) == 5:
                        cls, cx, cy, bw, bh = nums
                    else:
                        second = nums[1]
                        third = nums[2]
                        if 0.0 <= second <= 1.0 and 0.0 <= third <= 1.0:
                            cls, cx, cy, bw, bh = nums[:5]
                        else:
                            cls, conf, cx, cy, bw, bh = nums[:6]
                    x1 = (cx - bw / 2.0) * w
                    y1 = (cy - bh / 2.0) * h
                    annotations.append({
                        "id": ann_id,
                        "image_id": idx,
                        "category_id": int(cls),
                        "bbox": [x1, y1, bw * w, bh * h],
                        "area": (bw * w) * (bh * h),
                        "iscrowd": 0
                    })
                    ann_id += 1

        coco_dict = {
        "info": {"description": "BDD Custom Dataset", "version": "1.0"},
        "licenses": [],
        "images": images_json,
        "annotations": annotations,
        "categories": categories
    }

    with open(GT_JSON, "w") as jf:
        json.dump(coco_dict, jf)
    print(f"Saved GT COCO annotations: {GT_JSON} (images: {len(images_json)}, annotations: {len(annotations)})")
else:
    print("Using existing GT JSON:", GT_JSON)

PRED_JSON = os.path.join(SAVE_DIR, "preds_coco.json")
results_json = []
pred_txts = sorted(glob.glob(os.path.join(RELABELED_PRED_DIR, "*.txt")))
print(f"Found {len(pred_txts)} prediction .txt files in {RELABELED_PRED_DIR}")

for txt_file in pred_txts:
    base = os.path.splitext(os.path.basename(txt_file))[0]
    if base not in image_id_map:
        # debug print a few examples if mismatch
        continue
    image_id = image_id_map[base]
    img_candidates = [os.path.join(images_dir, base + ext) for ext in (".jpg", ".jpeg", ".png")]
    img_path = None
    for c in img_candidates:
        if os.path.exists(c):
            img_path = c
            break
    if img_path is None:
        continue
    im = Image.open(img_path)
    W, H = im.size

    with open(txt_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            try:
                cls, conf, cx, cy, bw, bh = map(float, parts[:6])
            except Exception:
                # try class cx cy w h
                try:
                    cls, cx, cy, bw, bh = map(float, parts[:5])
                    conf = 1.0
                except Exception:
                    continue
            x = (cx - bw / 2.0) * W
            y = (cy - bh / 2.0) * H
            results_json.append({
                "image_id": int(image_id),
                "category_id": int(cls),
                "bbox": [x, y, bw * W, bh * H],
                "score": float(conf)
            })

print(f"Converted predictions to COCO format. Entries: {len(results_json)}")

with open(PRED_JSON, "w") as jf:
    json.dump(results_json, jf)
print("Saved prediction JSON:", PRED_JSON)

if len(results_json) == 0:
    print("WARNING: No prediction entries found in PRED_JSON. This will produce zero scores.")
    print("Debug tips:")
    print(" - Confirm that image basenames in images_dir match label and prediction basenames.")
    print(" - List a few files from images_dir:", image_files[:5])
    print(" - List a few label files:", sorted(glob.glob(os.path.join(labels_dir, '*.txt')))[:5])
    print(" - List a few pred files:", pred_txts[:5])
else:
    print("Running COCO evaluation...")
    coco_gt = COCO(GT_JSON)
    coco_dt = coco_gt.loadRes(PRED_JSON)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    try:
        stats = coco_eval.stats
        print("\nMetrics (from coco_eval.stats):")
        print(f"AP@[.5:.95] (mAP): {stats[0]:.4f}")
        print(f"AP@0.5:           {stats[1]:.4f}")
        print(f"AP@0.75:          {stats[2]:.4f}")
        print(f"AR@1:             {stats[6]:.4f}")
        print(f"AR@10:            {stats[7]:.4f}")
        print(f"AR@100:           {stats[8]:.4f}")
    except Exception:
        print("Could not read coco_eval.stats. Raw object:", getattr(coco_eval, "stats", None))

if USE_CLIP and len(clip_max_sims) > 0:
    print("CLIP mean sim:", np.mean(clip_max_sims))
    print("CLIP median sim:", np.median(clip_max_sims))

print("Done. Files:")
print(" - GT JSON:", GT_JSON)
print(" - Pred JSON:", PRED_JSON)
print("Per-image predictions (.txt) directory:", RELABELED_PRED_DIR)
