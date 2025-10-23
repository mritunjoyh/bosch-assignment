import pytorch_lightning as pl
import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.roi_heads import RoIHeads
import torchvision.ops as ops

import torchvision
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small


class CustomFasterRCNN(pl.LightningModule):
    def __init__(self, lr, weight_decay, num_classes, batch_size, score_threshold, nms_iou_threshold):
        super().__init__()
        """
        FasterR-CNN Model using pre-trained resnet50 as backbone.
        
        lr - Learning rate for optimizer. Optimizer is set to Adam.
        weight_decay - Weight Decay for optimizer.
        num_classes - The number of classes model is detecting. Must add one to your amount of classes to represent the
                      background class. For example if you are trying to detect solely dogs in photos, the num_classes
                      will equal 2 (Dogs + Background).
        batch_size - Batch size of dataloader. This is needed to efficiently log the models loss scores.
        score_threshold - Threshold to use for output 'scores' when predicting images. Any predictions below the 
                          set threshold will be discarded. 
        nms_iou_threshold - The threshold score to use when completing NMS on model predictions.
        """

        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.batch_size = batch_size

        # backbone = mobilenet_v3_small(weights='IMAGENET1K_V1').features
        
        backbone = mobilenet_backbone("mobilenet_v3_small", fpn=True, weights="DEFAULT")

        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            out = backbone(x)

        print("Feature map keys:", list(out.keys()))
        print("Feature map count:", len(out))
        
        backbone.out_channels = 256
        

        # backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet18', weights='DEFAULT', trainable_layers=5)
        anchor_generator = AnchorGenerator(sizes=((16,), (32,), (64,)),
                                           aspect_ratios=tuple([(0.25, 0.5, 1.0) for _ in range(3)]))

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2'],
                                                        output_size=7,
                                                        sampling_ratio=2)
        


        self.model = FasterRCNN(backbone,
                                num_classes=self.num_classes,
                                rpn_anchor_generator=anchor_generator,
                                box_roi_pool=roi_pooler)
        
        # check how many FPN feature maps backbone gives

        # for param in self.model.backbone.parameters():
        #     param.requires_grad = False
        in_channels = self.model.roi_heads.box_head.fc6.in_features
        self.model.roi_heads.box_head = TwoMLPHead(in_channels, representation_size=32)  # ↓ 1024 → 256

        # update the predictor accordingly
        self.model.roi_heads.box_predictor = FastRCNNPredictor(32, self.num_classes)
        self.model.rpn.post_nms_top_n_train = 200
        self.model.rpn.post_nms_top_n_test = 100
        self.model.rpn.pre_nms_top_n_train = 400
        self.model.rpn.pre_nms_top_n_test = 200



    def forward(self, images, targets=None):
        if targets is None:
            return self.model(images)
        return self.model(images=images, targets=targets)

    def common_step(self, batch, batch_idx):
        # inside common_step, before loss computation
        # images = [x.to(self.device) for x in batch[0]]

        # # Move only tensor values to device, keep strings on cpu
        # targets = []
        # for t in batch[1]:
        #     new_t = {}
        #     for k, v in t.items():
        #         if torch.is_tensor(v):
        #             new_t[k] = v.to(self.device)
        #         else:
        #             new_t[k] = v
        #     targets.append(new_t)

        # # Validate label ranges and types
        # for i, t in enumerate(targets):
        #     if "labels" in t:
        #         labels = t["labels"]
        #         # ensure integer dtype
        #         if labels.dtype != torch.int64:
        #             labels = labels.long()
        #             t["labels"] = labels

        #         if torch.isnan(labels.float()).any():
        #             raise ValueError(f"Found NaN labels in sample idx {i}")

        #         min_l = int(labels.min().item()) if labels.numel() > 0 else None
        #         max_l = int(labels.max().item()) if labels.numel() > 0 else None

        #         # self.num_classes is the argument you passed to FasterRCNN
        #         if labels.numel() > 0 and (min_l < 0 or max_l >= self.num_classes):
        #             raise ValueError(
        #                 f"Label range error in batch sample {i}: min={min_l}, max={max_l}, "
        #                 f"model.num_classes={self.num_classes}. "
        #                 "Either adjust dataset labels or set model num_classes correctly."
        #             )

        
        images = [x.to(self.device) for x in batch[0]]
        # targets = [{k: v.to(self.device) for k, v in t.items()} for t in batch[1]]
        targets = [
            {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in t.items()}
            for t in batch[1]
        ]
        # targets = []
        # for t in batch[1]:
        #     new_t = {}
        #     for k, v in t.items():
        #         if torch.is_tensor(v):
        #             new_t[k] = v.to(self.device)
        #         else:
        #             # keep strings, ints, etc. on CPU
        #             new_t[k] = v
        #     targets.append(new_t)

        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        self.model.train()
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step, and the average across the epoch
        self.log("training_loss", loss.item())
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item(), on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        self.model.train()
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss.item())
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item(), on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)

        return loss

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

        return [optimizer]

    def save_checkpoint(self):
        if os.path.exists('model_training') is False:
            os.mkdir('model_training')

        torch.save({'epoch': self.current_epoch,
                    'model_state_dict': self.model.state_dict()
                    }, f'model_training/{self.current_epoch}.pth')

    def on_train_epoch_end(self):

        train_loss_classifier = self.trainer.callback_metrics.get('train_loss_classifier', 'N/A')
        train_loss_objectness = self.trainer.callback_metrics.get('train_loss_objectness', 'N/A')
        train_loss_box_reg = self.trainer.callback_metrics.get('train_loss_box_reg', 'N/A')
        train_loss_rpn_box_reg = self.trainer.callback_metrics.get('train_loss_rpn_box_reg', 'N/A')

        validation_loss_classifier = self.trainer.callback_metrics.get('validation_loss_classifier', 'N/A')
        validation_loss_objectness = self.trainer.callback_metrics.get('validation_loss_objectness', 'N/A')
        validation_loss_box_reg = self.trainer.callback_metrics.get('validation_loss_box_reg', 'N/A')
        validation_rpn_box_reg = self.trainer.callback_metrics.get('validation_rpn_box_reg', 'N/A')

        self.save_checkpoint()

        print(f'Epoch {self.current_epoch}: '
              f'Train Loss Classifier: {train_loss_classifier}, '
              f'Train Loss Objectness: {train_loss_objectness}, '
              f'Train Loss Box Reg: {train_loss_box_reg}, '
              f'Train Loss Rpn Box Reg: {train_loss_rpn_box_reg}, '
              f'Validation Loss Classifier: {validation_loss_classifier}, '
              f'Validation Loss Objectness: {validation_loss_objectness} ',
              f'Validation Loss Box Reg: {validation_loss_box_reg}, '
              f'Validation Loss Rpn Box Reg: {validation_rpn_box_reg},')

    def decode_prediction(self, preds):
        boxes = preds['boxes']
        scores = preds['scores']
        labels = preds['labels']

        if self.score_threshold is not None:
            want = scores > self.score_threshold
            preds['boxes'] = boxes[want]
            preds['scores'] = scores[want]
            preds['labels'] = labels[want]

        if self.score_threshold is not None:
            want = scores > self.score_threshold
            boxes = boxes[want]
            scores = scores[want]
            labels = labels[want]

            keep = torchvision.ops.nms(boxes=boxes, scores=scores, iou_threshold=self.nms_iou_threshold)

            preds['boxes'] = preds['boxes'][keep]
            preds['scores'] = preds['scores'][keep]
            preds['labels'] = preds['labels'][keep]

        return preds

    def predict(self, dm, rand=True, images=None):
        
        self.model.eval().to(self.device)

        if rand:
            images = []
            for r in random.sample(range(0, len(dm.test_data)), 3):
                images.append(dm.test_data[r])

        fig, ax = plt.subplots(len(images), figsize=(10, 5 * len(images)))

        for v, (i, l) in enumerate(images):
            with torch.no_grad():
                outputs = self.model([i.to(self.device)])
                results = self.decode_prediction(*outputs)

            boxes = results['boxes'].cpu()
            labels = results['labels'].cpu()

            ax_curr = ax[v] if len(images) > 1 else ax

            ax_curr.imshow(i.permute(1, 2, 0).numpy().astype("uint8"))

            for box, label in zip(boxes, labels):
                x_min, y_min, x_max, y_max = box[0], box[1], box[2], box[3]
                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
                ax_curr.add_patch(rect)
                if label:
                    ax_curr.text(x_min, y_min, label, color='r', fontsize=10, ha='left', va='bottom')
        return plt.show()
    
class EvalFasterRCNN(pl.LightningModule):
    def __init__(self, score_threshold=0.5, nms_iou_threshold=0.5):
        super().__init__()
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold

        # Load pretrained model
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        self.model.eval()

    def forward(self, images):
        return self.model(images)

    def decode_prediction(self, preds):
        boxes = preds["boxes"]
        scores = preds["scores"]
        labels = preds["labels"]
        keep = scores > self.score_threshold
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        keep2 = torchvision.ops.nms(boxes, scores, self.nms_iou_threshold)
        return {"boxes": boxes[keep2], "scores": scores[keep2], "labels": labels[keep2]}

    # Your predict method similar to your existing one ...

# import pytorch_lightning as pl
# import torch
# import torchvision
# from torchvision.models.detection import FasterRCNN
# from torchvision.models.detection.rpn import AnchorGenerator
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
# from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
# from torchvision.ops import MultiScaleRoIAlign
# import os, random, matplotlib.pyplot as plt, matplotlib.patches as patches


# class LiteFasterRCNN(pl.LightningModule):
#     def __init__(
#         self,
#         lr=1e-4,
#         weight_decay=1e-5,
#         num_classes=2,
#         batch_size=2,
#         score_threshold=0.5,
#         nms_iou_threshold=0.5,
#         backbone_name="resnet18",
#         fpn_out_channels=128,
#         roi_hidden_dim=256,
#         trainable_layers=2,
#     ):
#         super().__init__()

#         self.save_hyperparameters()

#         # --- Backbone ---
#         backbone = resnet_fpn_backbone(
#                 'resnet18',
#                 weights='DEFAULT',
#                 trainable_layers=3,
#             )
#             # DO NOT change out_channels manually

#         backbone.out_channels = fpn_out_channels

#         # --- Anchor Generator ---
#         anchor_generator = AnchorGenerator(
#             sizes=((16,), (32,), (64,), (128,), (256,)),
#             aspect_ratios=((0.5, 1.0, 2.0),) * 5,
#         )

#         # --- ROI Pooler ---
#         roi_pooler = MultiScaleRoIAlign(
#             featmap_names=["0", "1", "2", "3"],
#             output_size=7,
#             sampling_ratio=2,
#         )

#         # --- Faster R-CNN model ---
#         self.model = FasterRCNN(
#             backbone,
#             num_classes=num_classes,
#             rpn_anchor_generator=anchor_generator,
#             box_roi_pool=roi_pooler,
#         )

#         # --- Shrink ROI head ---
#         in_channels = self.model.roi_heads.box_head.fc6.in_features
#         self.model.roi_heads.box_head = TwoMLPHead(in_channels, representation_size=roi_hidden_dim)
#         self.model.roi_heads.box_predictor = FastRCNNPredictor(roi_hidden_dim, num_classes)

#         # --- Reduce proposals to save memory ---
#         self.model.rpn.post_nms_top_n_train = 500
#         self.model.rpn.post_nms_top_n_test = 200

#         self.lr = lr
#         self.weight_decay = weight_decay
#         self.batch_size = batch_size
#         self.score_threshold = score_threshold
#         self.nms_iou_threshold = nms_iou_threshold

#     # ---------------- Training and Validation ----------------
#     def forward(self, images, targets=None):
#         return self.model(images, targets)

#     def common_step(self, batch):
#         images = [x.to(self.device) for x in batch[0]]
#         targets = [
#             {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in t.items()}
#             for t in batch[1]
#         ]

#         loss_dict = self.model(images, targets)
#         loss = sum(loss_dict.values())
#         return loss, loss_dict

#     def training_step(self, batch, batch_idx):
#         self.model.train()
#         loss, loss_dict = self.common_step(batch)
#         self.log("train_loss", loss.item(), on_step=False, on_epoch=True)
#         for k, v in loss_dict.items():
#             self.log(f"train_{k}", v.item(), on_step=False, on_epoch=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         self.model.train()
#         loss, loss_dict = self.common_step(batch)
#         self.log("val_loss", loss.item(), on_step=False, on_epoch=True)
#         for k, v in loss_dict.items():
#             self.log(f"val_{k}", v.item(), on_step=False, on_epoch=True)
#         return loss

#     def configure_optimizers(self):
#         params = [p for p in self.model.parameters() if p.requires_grad]
#         optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
#         return optimizer

#     # ---------------- Utility ----------------
#     def decode_prediction(self, preds):
#         boxes = preds["boxes"]
#         scores = preds["scores"]
#         labels = preds["labels"]

#         if self.score_threshold is not None:
#             keep = scores > self.score_threshold
#             boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
#         keep = torchvision.ops.nms(boxes, scores, self.nms_iou_threshold)
#         preds = {k: v[keep] for k, v in {"boxes": boxes, "scores": scores, "labels": labels}.items()}
#         return preds

#     def predict(self, dm, rand=True, images=None):
#         self.model.eval().to(self.device)
#         if rand:
#             images = [dm.test_data[random.randint(0, len(dm.test_data) - 1)] for _ in range(3)]

#         fig, ax = plt.subplots(len(images), figsize=(10, 5 * len(images)))
#         for i, (img, _) in enumerate(images):
#             with torch.no_grad():
#                 preds = self.model([img.to(self.device)])[0]
#                 preds = self.decode_prediction(preds)

#             boxes, labels = preds["boxes"].cpu(), preds["labels"].cpu()
#             ax_i = ax[i] if len(images) > 1 else ax
#             ax_i.imshow(img.permute(1, 2, 0))
#             for box, label in zip(boxes, labels):
#                 x1, y1, x2, y2 = box
#                 rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
#                                          linewidth=1, edgecolor="r", facecolor="none")
#                 ax_i.add_patch(rect)
#                 ax_i.text(x1, y1, f"{label.item()}", color="r")
#         plt.show()

import pytorch_lightning as pl
import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.backbone_utils import mobilenet_backbone
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision
import os, random
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class CustomFasterRCNN_MobileNetV3(pl.LightningModule):
    def __init__(
        self,
        lr=1e-4,
        weight_decay=1e-5,
        num_classes=2,
        batch_size=2,
        score_threshold=0.5,
        nms_iou_threshold=0.5,
        roi_hidden_dim=256,
    ):
        """
        Lightweight Faster-RCNN model using pretrained MobileNetV3-Large backbone (with FPN).
        Perfect for low-memory GPUs.

        Args:
            lr: learning rate
            weight_decay: optimizer L2 penalty
            num_classes: your dataset classes (+1 for background)
            batch_size: DataLoader batch size (for logging)
            score_threshold: confidence threshold for predictions
            nms_iou_threshold: NMS IoU cutoff
            roi_hidden_dim: hidden layer dim in ROI box head (default 256)
        """
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold

        backbone = mobilenet_backbone("mobilenet_v3_large", fpn=True, weights="DEFAULT")
        # Detect how many feature maps backbone outputs
        try:
            num_feature_maps = len(backbone.fpn.inner_blocks)
        except AttributeError:
            num_feature_maps = len(backbone.body.return_layers)

        # Generate anchors dynamically
        # --- Anchor Generator ---
        anchor_generator = AnchorGenerator(
            sizes=tuple((2 ** (4 + i),) for i in range(num_feature_maps)),
            aspect_ratios=((0.5, 1.0, 2.0),) * num_feature_maps,
        )

        # --- ROI Pooler ---
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=[str(i) for i in range(num_feature_maps)],
            output_size=7,
            sampling_ratio=2,
        )


        self.model = FasterRCNN(
            backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
        )

        in_channels = self.model.roi_heads.box_head.fc6.in_features
        self.model.roi_heads.box_head = TwoMLPHead(in_channels, representation_size=roi_hidden_dim)
        self.model.roi_heads.box_predictor = FastRCNNPredictor(roi_hidden_dim, num_classes)

        self.model.rpn.post_nms_top_n_train = 500
        self.model.rpn.post_nms_top_n_test = 200

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def common_step(self, batch, batch_idx=None):
        images = [x.to(self.device) for x in batch[0]]
        targets = [
            {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in t.items()}
            for t in batch[1]
        ]
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        self.model.train()
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("training_loss", loss.item())
        for k, v in loss_dict.items():
            self.log(f"train_{k}", v.item(), on_epoch=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.train()
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss.item())
        for k, v in loss_dict.items():
            self.log(f"validation_{k}", v.item(), on_epoch=True, batch_size=self.batch_size)
        return loss

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def save_checkpoint(self):
        os.makedirs("model_training", exist_ok=True)
        torch.save(
            {"epoch": self.current_epoch, "model_state_dict": self.model.state_dict()},
            f"model_training/epoch_{self.current_epoch}.pth",
        )

    def decode_prediction(self, preds):
        boxes = preds["boxes"]
        scores = preds["scores"]
        labels = preds["labels"]

        keep = scores > self.score_threshold
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        keep = torchvision.ops.nms(boxes, scores, self.nms_iou_threshold)

        return {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]}

    def predict(self, dm, rand=True, images=None):
        self.model.eval().to(self.device)
        if rand:
            images = [dm.test_data[random.randint(0, len(dm.test_data) - 1)] for _ in range(3)]

        fig, ax = plt.subplots(len(images), figsize=(10, 5 * len(images)))
        for i, (img, _) in enumerate(images):
            with torch.no_grad():
                preds = self.model([img.to(self.device)])[0]
                preds = self.decode_prediction(preds)

            boxes, labels = preds["boxes"].cpu(), preds["labels"].cpu()
            ax_i = ax[i] if len(images) > 1 else ax
            ax_i.imshow(img.permute(1, 2, 0))
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="r", facecolor="none"
                )
                ax_i.add_patch(rect)
                ax_i.text(x1, y1, str(label.item()), color="r", fontsize=10)
        plt.show()
