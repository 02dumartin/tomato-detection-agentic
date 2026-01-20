"""YOLOv12 Model"""
from typing import Dict, Any
from .yolo_base import BaseYOLOWrapper


class YOLOv12Wrapper(BaseYOLOWrapper):
    """YOLOv12 래퍼"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, model_version="yolo12")
