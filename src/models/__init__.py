"""Models module - Lazy loading support"""
from typing import Dict, Optional, Type

# DETR는 항상 import (가벼움)
try:
    from .detr_model import DetrLightningModule
    DETR_AVAILABLE = True
except ImportError:
    DETR_AVAILABLE = False
    DetrLightningModule = None

# YOLO는 lazy loading (필요할 때만 import)
YOLO_AVAILABLE = None
YOLOv11Wrapper = None
YOLOv12Wrapper = None

# Grounding DINO는 lazy loading
GROUNDING_DINO_AVAILABLE = None
GroundingDINOWrapper = None

# Florence-2는 lazy loading
FLORENCE2_AVAILABLE = None
Florence2Base = None
Florence2Finetuned = None


def _load_yolo():
    """YOLO 모델 lazy loading"""
    global YOLO_AVAILABLE, YOLOv11Wrapper, YOLOv12Wrapper
    
    if YOLO_AVAILABLE is None:
        try:
            from .yolov11_model import YOLOv11Wrapper
            from .yolov12_model import YOLOv12Wrapper
            YOLO_AVAILABLE = True
        except ImportError:
            YOLO_AVAILABLE = False
            YOLOv11Wrapper = None
            YOLOv12Wrapper = None
    
    return YOLO_AVAILABLE


def _load_grounding_dino():
    """Grounding DINO lazy loading"""
    global GROUNDING_DINO_AVAILABLE, GroundingDINOWrapper
    
    if GROUNDING_DINO_AVAILABLE is None:
        try:
            from .gdino_model import GroundingDINOWrapper
            GROUNDING_DINO_AVAILABLE = True
        except ImportError:
            GROUNDING_DINO_AVAILABLE = False
            GroundingDINOWrapper = None
    
    return GROUNDING_DINO_AVAILABLE


def _load_florence2():
    """Florence-2 모델 lazy loading"""
    global FLORENCE2_AVAILABLE, Florence2Base, Florence2Finetuned
    
    if FLORENCE2_AVAILABLE is None:
        try:
            from .florence2_base import Florence2Base
            from .florence2_finetuned import Florence2Finetuned
            FLORENCE2_AVAILABLE = True
        except Exception as e:
            # ImportError뿐만 아니라 다른 예외도 처리 (예: 의존성 오류)
            FLORENCE2_AVAILABLE = False
            Florence2Base = None
            Florence2Finetuned = None
    
    return FLORENCE2_AVAILABLE


# Lazy loading Model Registry
class LazyModelRegistry(dict):
    """Lazy loading을 지원하는 모델 레지스트리"""
    
    def __getitem__(self, key: str):
        """Lazy loading으로 모델 클래스 반환"""
        key_lower = key.lower()
        
        # DETR
        if key_lower == 'detr':
            if not DETR_AVAILABLE:
                raise KeyError(f"Model 'DETR' not available")
            return DetrLightningModule
        
        # YOLO
        if 'yolo' in key_lower:
            _load_yolo()
            if not YOLO_AVAILABLE:
                raise KeyError(f"Model 'YOLO' not available")
            
            if 'yolo11' in key_lower or key_lower == 'yolo':
                return YOLOv11Wrapper
            elif 'yolo12' in key_lower:
                return YOLOv12Wrapper
            else:
                return YOLOv11Wrapper
        
        # Grounding DINO
        if 'grounding' in key_lower or 'gdino' in key_lower:
            _load_grounding_dino()
            if not GROUNDING_DINO_AVAILABLE:
                raise KeyError(f"Model 'GroundingDINO' not available")
            return GroundingDINOWrapper
        
        # Florence-2
        if 'florence' in key_lower:
            _load_florence2()
            if not FLORENCE2_AVAILABLE:
                raise KeyError(f"Model 'Florence2' not available")
            
            if 'finetuned' in key_lower or 'finetune' in key_lower:
                return Florence2Finetuned
            else:
                # Zero-shot은 Florence2Base 사용
                return Florence2Base
        
        raise KeyError(f"Model '{key}' not found")
    
    def __contains__(self, key: str) -> bool:
        """키가 존재하는지 확인"""
        try:
            _ = self[key]
            return True
        except (KeyError, ImportError):
            return False
    
    def get(self, key: str, default=None):
        """모델 클래스 가져오기 (없으면 default 반환)"""
        try:
            return self[key]
        except (KeyError, ImportError):
            return default
    
    def keys(self):
        """등록된 모델 키 목록"""
        return [
            'detr', 'DETR',
            'yolov11', 'YOLOv11', 'yolov12', 'YOLOv12', 'yolo', 'YOLO',
            'groundingdino', 'GroundingDINO', 'gdino', 'GDINO',
            'florence2', 'Florence2', 'florence2_zeroshot', 'florence2_finetuned'
        ]


# Model Registry (lazy loading)
MODEL_REGISTRY = LazyModelRegistry()

__all__ = [
    'MODEL_REGISTRY',
]