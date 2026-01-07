"""Datasets module"""
from .registry import DatasetRegistry
from .TomatOD_3 import TomatOD3Meta
from .TomatOD_1 import TomatOD1Meta

# 등록
DatasetRegistry.register("TomatOD_3")(TomatOD3Meta)
DatasetRegistry.register("TomatOD_1")(TomatOD1Meta)
DatasetRegistry.register("TomatOD_COCO_3")(TomatOD3Meta)  # Alias
DatasetRegistry.register("TomatOD_COCO_1")(TomatOD1Meta)  # Alias
DatasetRegistry.register("TomatOD_YOLO_3")(TomatOD3Meta)  # Alias for YOLO format
DatasetRegistry.register("TomatOD_YOLO_1")(TomatOD1Meta)  # Alias for YOLO format

DATASET_REGISTRY = DatasetRegistry

__all__ = ['DATASET_REGISTRY', 'DatasetRegistry', 'TomatOD3Meta', 'TomatOD1Meta']

