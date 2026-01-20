import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import numpy as np

try:
    from groundingdino.util.inference import load_model
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict
    GROUNDING_DINO_AVAILABLE = True
except ImportError:
    GROUNDING_DINO_AVAILABLE = False


class GroundingDINODataset(Dataset):
    """Grounding DINO용 COCO 데이터셋"""
    
    def __init__(
        self,
        ann_file: str,
        images_dir: str,
        transform: Optional[T.Compose] = None,
        class_names: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)
        
        self.images_dir = Path(images_dir)
        self.config = config or {}
        self.class_names = class_names or []
        
        self.image_id_to_file = {img['id']: img['file_name'] for img in self.coco_data['images']}
        self.image_id_to_info = {img['id']: img for img in self.coco_data['images']}
        self.image_ids = list(self.image_id_to_file.keys())
        
        self.annotations_by_image = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations_by_image:
                self.annotations_by_image[img_id] = []
            self.annotations_by_image[img_id].append(ann)
        
        self.category_id_to_name = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        
        if transform is None:
            self.transform = self._create_default_transform()
        else:
            self.transform = transform
    
    def _create_default_transform(self):
        """Grounding DINO 기본 이미지 전처리"""
        return T.Compose([
            T.Resize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        file_name = self.image_id_to_file[img_id]
        img_path = self.images_dir / file_name
        
        image = Image.open(img_path).convert('RGB')
        img_width, img_height = image.size
        
        anns = self.annotations_by_image.get(img_id, [])
        
        # 텍스트 프롬프트 생성
        if self.class_names:
            text_prompt = ". ".join([f"{name} tomato" for name in self.class_names]) + "."
        else:
            categories = list(set([self.category_id_to_name[ann['category_id']] for ann in anns]))
            text_prompt = ". ".join([f"{cat} tomato" for cat in categories]) + "."
        
        if self.transform:
            image = self.transform(image)
        
        # 타겟 준비
        boxes = []
        labels = []
        for ann in anns:
            boxes.append(ann['bbox'])  # [x, y, w, h]
            category_name = self.category_id_to_name[ann['category_id']]
            if self.class_names and category_name in self.class_names:
                label = self.class_names.index(category_name)
            else:
                label = 0
            labels.append(label)
        
        return {
            'image': image,
            'text_prompt': text_prompt,
            'boxes': boxes,
            'labels': labels,
            'image_id': img_id,
            'image_size': torch.tensor([img_width, img_height], dtype=torch.float32)
        }
    
    @staticmethod
    def collate_fn(batch):
        """가변 길이 데이터 처리"""
        images = torch.stack([item['image'] for item in batch])
        text_prompts = [item['text_prompt'] for item in batch]
        boxes = [item['boxes'] for item in batch]
        labels = [item['labels'] for item in batch]
        image_ids = [item['image_id'] for item in batch]
        image_sizes = torch.stack([item['image_size'] for item in batch])
        
        return {
            'image': images,
            'text_prompt': text_prompts,
            'boxes': boxes,
            'labels': labels,
            'image_id': image_ids,
            'image_size': image_sizes
        }


def create_gdino_dataset(dataset_meta, split: str, transform=None, config=None):
    """
    Grounding DINO 데이터셋 생성 (DETR의 create_detr_dataset과 유사한 인터페이스)
    
    Args:
        dataset_meta: 데이터셋 메타 정보 (예: TomatOD3Meta)
        split: 'train', 'val', 'test'
        transform: 이미지 변환 (None이면 기본 변환 사용)
        config: 설정 딕셔너리
    """
    # 데이터 경로 가져오기
    paths = dataset_meta.get_data_paths(split, config)
    
    # 클래스 이름 가져오기
    class_names = dataset_meta.get_class_names()
    
    # 데이터셋 생성
    dataset = GroundingDINODataset(
        ann_file=paths['ann_file'],
        images_dir=paths['img_folder'],
        transform=transform,
        class_names=class_names,
        config=config
    )
    
    return dataset