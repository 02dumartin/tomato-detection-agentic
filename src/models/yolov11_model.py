"""YOLOv11 Model"""
from typing import Dict, Any
from .yolo_base import BaseYOLOWrapper


class YOLOv11Wrapper(BaseYOLOWrapper):
    """YOLOv11 래퍼"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, model_version="yolo11")
