import os
import yaml
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class YoloDetectionDataset(torch.utils.data.Dataset):
    """
    YOLO-style dataset reading .txt label files and paths from a data.yaml.
    Compatible with Ultralytics YOLO format.
    """

    def __init__(self, data_yaml: str, split: str = "train", img_size: int = 640, transform=None):
        super().__init__()

        # --- Load dataset config ---
        with open(data_yaml, "r") as f:
            cfg = yaml.safe_load(f)
        base_path = cfg.get("path", "")
        img_rel = cfg[split]
        self.img_dir = os.path.join(base_path, img_rel)
        self.label_dir = self.img_dir.replace("images", "labels")

        self.names = cfg.get("names", {})
        self.nc = len(self.names)
        self.img_size = img_size
        # boxes = np.clip(boxes, 0, [w - 1, h - 1, w - 1, h - 1])
        self.transform = transform or A.Compose(
            [
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",   # pixel coordinates, not normalized
                label_fields=["class_labels"],
            ),
        )
        # --- Collect all image files ---
        self.img_files = [
            os.path.join(root, f)
            for root, _, files in os.walk(self.img_dir)
            for f in files
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]#[:10]
        # import pdb
        # pdb.set_tr?sace()
        if len(self.img_files) == 0:
            raise FileNotFoundError(f"No images found in {self.img_dir}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = np.array(Image.open(img_path).convert("RGB")).astype(np.float32)
        h, w, _ = img.shape

        # Read YOLO label file
        label_path = os.path.join(
            self.label_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        )

        boxes, labels = [], []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    cls, xc, yc, bw, bh = map(float, line.strip().split())
                    x1 = (xc - bw / 2) * w
                    y1 = (yc - bh / 2) * h
                    x2 = (xc + bw / 2) * w
                    y2 = (yc + bh / 2) * h
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(cls))

        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        if len(boxes) > 0:
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)

        valid_boxes, valid_labels = [], []
        for box, label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box
            if xmax > xmin and ymax > ymin:
                valid_boxes.append([float(xmin), float(ymin), float(xmax), float(ymax)])
                valid_labels.append(int(label))

        if len(valid_boxes) == 0:
            next_idx = (idx + 1) % len(self.img_files)
            return self.__getitem__(next_idx)

        try:
            transformed = self.transform(
                image=img,
                bboxes=valid_boxes,
                class_labels=valid_labels,
            )
        except Exception as e:
            print(f"[WARN] Skipping sample {img_path} due to transform error: {e}")
            with open("bad_boxes.txt", "a") as f:
                f.write(f"{img_path}\n")
            next_idx = (idx + 1) % len(self.img_files)
            return self.__getitem__(next_idx)

        img_t = transformed["image"]

        if len(transformed["bboxes"]) > 0:
            boxes_t = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)

        if len(transformed["class_labels"]) > 0:
            labels_t = torch.as_tensor(transformed["class_labels"], dtype=torch.long)
        else:
            labels_t = torch.zeros((0,), dtype=torch.long)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx]),
            "path": img_path,
        }

        return img_t, target



