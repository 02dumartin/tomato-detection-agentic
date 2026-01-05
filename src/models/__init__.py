"""Models module"""
from .detr_model import DetrLightningModule
from .yolov11_model import YOLOv11LightningModule, YOLOv11Wrapper
# from .yolov12_model import YOLOv12LightningModule
# from .mmdet_model import MMDetLightningModule
# from .florence_model import FlorenceLightningModule
# from .gdino_model import GroundingDINOLightningModule

# Model Registry
MODEL_REGISTRY = {
    'detr': DetrLightningModule,
    'DETR': DetrLightningModule,
    'yolov11': YOLOv11Wrapper,  # Lightning 없이 직접 사용
    'YOLOv11': YOLOv11Wrapper,
    'yolo': YOLOv11Wrapper,
    'YOLO': YOLOv11Wrapper,
    # 'YOLOv12': YOLOv12LightningModule,
    # 'MMDetection': MMDetLightningModule,
    # 'Florence': FlorenceLightningModule,
    # 'GroundingDINO': GroundingDINOLightningModule,
}

__all__ = [
    'MODEL_REGISTRY',
    'DetrLightningModule',
    'YOLOv11LightningModule',
    'YOLOv11Wrapper',
]