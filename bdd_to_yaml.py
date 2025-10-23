import os
import json
from tqdm import tqdm
from pathlib import Path
JSON_PATH = "assignment_data_bdd/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json"   # or det_val.json

OUTPUT_DIR = "/media/mritunjoy/mritunjoy/BOSCH/assignment_data_bdd/bdd100k/val/labels/"           # or val/labels

IMG_DIR = "assignment_data_bdd/bdd100k_images_100k/bdd100k/images/100k/val"

IMG_WIDTH = 1280
IMG_HEIGHT = 720

CLASSES = [
    "person",
    "car",
    "truck",
    "bus",
    "bike",
    "motor",
    "rider",
    "traffic light",
    "traffic sign",
    "train"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f" Saving YOLO labels to: {OUTPUT_DIR}")

with open(JSON_PATH, "r") as f:
    annotations = json.load(f)

print(f"Loaded {len(annotations)} image entries from JSON")

for entry in tqdm(annotations, desc="Converting"):
    img_name = entry["name"]
    labels = entry.get("labels", [])
    if len(labels) == 0:
        continue  # skip images with no detections

    yolo_lines = []
    for obj in labels:
        category = obj.get("category")
        if "box2d" not in obj or category not in CLASSES:
            continue

        box = obj["box2d"]
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

        x_center = ((x1 + x2) / 2) / IMG_WIDTH
        y_center = ((y1 + y2) / 2) / IMG_HEIGHT
        width = (x2 - x1) / IMG_WIDTH
        height = (y2 - y1) / IMG_HEIGHT

        class_id = CLASSES.index(category)
        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        yolo_lines.append(yolo_line)

    label_path = Path(OUTPUT_DIR) / Path(img_name).with_suffix(".txt").name
    with open(label_path, "w") as f:
        f.write("\n".join(yolo_lines))

print("Conversion complete!")
print(f"Saved YOLO labels in: {OUTPUT_DIR}")
