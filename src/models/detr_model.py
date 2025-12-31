"""DETR Lightning Module - Pretrained model 사용"""
import torch
import pytorch_lightning as pl
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import DetrForObjectDetection

class DetrLightningModule(pl.LightningModule):
    """DETR Lightning Module (Pretrained)"""
    
    def __init__(
        self,
        num_labels: int,
        pretrained_path: str = "facebook/detr-resnet-50",
        lr: float = 1e-4,
        lr_backbone: float = 1e-5,
        weight_decay: float = 1e-4,
        score_threshold: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Pretrained DETR 모델
        self.model = DetrForObjectDetection.from_pretrained(
            pretrained_path,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )
        
        self.map_metric = MeanAveragePrecision(
            box_format="cxcywh", iou_type="bbox", class_metrics=True
        )
    
    def forward(self, pixel_values, pixel_mask=None):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
    
    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch.get("pixel_mask")
        labels = [{k: v.to(self.device) for k, v in t.items()} 
                  for t in batch["labels"]]
        
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        return outputs.loss, outputs.loss_dict
    
    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        for k, v in loss_dict.items():
            self.log(f"train_{k}", v.item())
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        for k, v in loss_dict.items():
            self.log(f"val_{k}", v.item())
        
        # mAP 계산
        with torch.no_grad():
            labels = [{k: v.to(self.device) for k, v in t.items()} 
                     for t in batch["labels"]]
            outputs = self.model(pixel_values=batch["pixel_values"], pixel_mask=batch.get("pixel_mask"))
            self._update_map(outputs, labels)
        return loss
    
    def _update_map(self, outputs, labels):
        probs = outputs.logits.softmax(-1)[..., :-1]
        scores, pred_labels = probs.max(-1)
        pred_boxes = outputs.pred_boxes
        
        preds = []
        targets = []
        
        for i in range(pred_boxes.shape[0]):
            keep = scores[i] > self.hparams.score_threshold
            preds.append({
                "boxes": pred_boxes[i][keep].detach().cpu(),
                "scores": scores[i][keep].detach().cpu(),
                "labels": pred_labels[i][keep].detach().cpu(),
            })
            targets.append({
                "boxes": labels[i]["boxes"].detach().cpu(),
                "labels": labels[i]["class_labels"].detach().cpu(),
            })
        
        if preds:
            self.map_metric.update(preds, targets)
    
    def on_validation_epoch_end(self):
        metrics = self.map_metric.compute()
        for k, v in metrics.items():
            if torch.is_tensor(v) and v.ndim == 0:
                self.log(f"val_{k}", v, prog_bar=True)
        self.map_metric.reset()
    
    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() 
                       if "backbone" not in n and p.requires_grad]},
            {"params": [p for n, p in self.named_parameters() 
                       if "backbone" in n and p.requires_grad],
             "lr": self.hparams.lr_backbone},
        ]
        return torch.optim.AdamW(
            param_dicts, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )