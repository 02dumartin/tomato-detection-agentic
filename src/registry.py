"""Model and Dataset Registry"""

# Model Registry는 src/models/__init__.py에서 import
from .models import MODEL_REGISTRY

# Dataset Registry는 src/data/datasets/__init__.py에서 import
from .data.datasets import DATASET_REGISTRY


def list_models():
    """등록된 모델 목록"""
    return list(MODEL_REGISTRY.keys())


def list_datasets():
    """등록된 데이터셋 목록"""
    return list(DATASET_REGISTRY.list_datasets())


__all__ = ['MODEL_REGISTRY', 'DATASET_REGISTRY', 'list_models', 'list_datasets']

