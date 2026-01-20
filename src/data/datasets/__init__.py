"""Datasets module"""
from .registry import DatasetRegistry
from .TomatOD_3 import TomatOD3Meta
from .TomatomMerge_1 import TomatomMerge1Meta

# 등록
DatasetRegistry.register("TomatOD_3")(TomatOD3Meta)
DatasetRegistry.register("TomatOD_COCO_3")(TomatOD3Meta)  # Alias
DatasetRegistry.register("TomatOD_YOLO_3")(TomatOD3Meta)  # Alias for YOLO format

# Tomato_merge dataset (1-class)
DatasetRegistry.register("Tomato_merge_1")(TomatomMerge1Meta)
DatasetRegistry.register("TomatOD_1_YOLO")(TomatomMerge1Meta)  # Alias for YOLO format 1-class
DatasetRegistry.register("TomatOD_1")(TomatomMerge1Meta)  # Alias for 1-class configs

DATASET_REGISTRY = DatasetRegistry

__all__ = ['DATASET_REGISTRY', 'DatasetRegistry', 'TomatOD3Meta', 'TomatomMerge1Meta']
