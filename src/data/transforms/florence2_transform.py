"""
Florence-2 Fine-tuning Dataset
COCO format annotations를 Florence-2 형식으로 변환
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance, ImageOps
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random


def load_image_exif_corrected(image_path: str) -> Image.Image:
    """
    Load image and apply EXIF orientation correction.
    Always returns an image in the visual (upright) orientation.
    
    Args:
        image_path: Path to image file
    
    Returns:
        PIL Image with EXIF orientation applied (RGB mode)
    """
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)  # ⭐ 핵심: EXIF orientation 보정
    img = img.convert("RGB")
    return img


class Florence2Dataset(Dataset):
    """
    Florence-2 Object Detection Fine-tuning Dataset
    
    COCO format annotations를 Florence-2의 <OD> task 형식으로 변환
    Format: "<OD>" -> "<loc_x1><loc_y1><loc_x2><loc_y2>class_name"
    
    Note: 모든 이미지는 EXIF orientation이 보정된 상태로 로드됩니다.
    COCO annotation의 width/height는 ImageOps.exif_transpose(img).size와 일치해야 합니다.
    """
    
    def __init__(
        self,
        coco_ann_file: str,
        image_dir: str,
        processor,
        max_length: int = 1024,
        task_prompt: str = "<OD>",
        is_train: bool = True,
        augment: bool = True,
        use_letterbox: bool = True,
        aug_brightness_range: Tuple[float, float] = (0.8, 1.2),
        aug_contrast_range: Tuple[float, float] = (0.8, 1.2),
        aug_saturation_range: Tuple[float, float] = (0.8, 1.2),
        max_objects_per_image: Optional[int] = None  # 최대 객체 수 제한 (None이면 제한 없음)
    ):
        """
        Args:
            coco_ann_file: COCO format annotation file (JSON)
            image_dir: Image directory
            processor: Florence-2 processor (AutoProcessor)
            max_length: Maximum sequence length
            task_prompt: Task prompt (default: "<OD>")
            is_train: Training mode (enables augmentation)
            augment: Enable data augmentation
            use_letterbox: Use letterbox resize (maintain aspect ratio)
            aug_brightness_range: Brightness augmentation range (min, max)
            aug_contrast_range: Contrast augmentation range (min, max)
            aug_saturation_range: Saturation augmentation range (min, max)
        """
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.max_length = max_length
        self.task_prompt = task_prompt
        self.is_train = is_train
        self.augment = augment and is_train  # Only augment during training
        self.use_letterbox = use_letterbox
        self.aug_brightness_range = aug_brightness_range
        self.aug_contrast_range = aug_contrast_range
        self.aug_saturation_range = aug_saturation_range
        self.max_objects_per_image = max_objects_per_image
        
        # COCO annotations 로드
        with open(coco_ann_file, 'r') as f:
            coco_data = json.load(f)
        
        # 카테고리 맵 생성
        self.categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        # 이미지별 annotations 그룹화
        self.image_info = {img['id']: img for img in coco_data['images']}
        self.annotations_by_image = {}
        
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations_by_image:
                self.annotations_by_image[img_id] = []
            self.annotations_by_image[img_id].append(ann)
        
        # 유효한 이미지만 필터링 (annotations 있는 것만)
        self.image_ids = [
            img_id for img_id in self.image_info.keys()
            if img_id in self.annotations_by_image
        ]
        
        # 객체 수 통계 계산
        num_objects_per_image = [
            len(self.annotations_by_image[img_id]) 
            for img_id in self.image_ids
        ]
        
        avg_objects = sum(num_objects_per_image) / len(num_objects_per_image) if num_objects_per_image else 0
        min_objects = min(num_objects_per_image) if num_objects_per_image else 0
        max_objects = max(num_objects_per_image) if num_objects_per_image else 0
        
        print(f"  Loaded {len(self.image_ids)} images with annotations")
        print(f"  Categories: {list(self.categories.values())}")
        print(f"  Objects per image - Min: {min_objects}, Avg: {avg_objects:.1f}, Max: {max_objects}")
        
        # 객체가 너무 적은 이미지 경고
        images_with_few_objects = sum(1 for n in num_objects_per_image if n <= 1)
        if images_with_few_objects > 0:
            print(f"  ⚠️  Warning: {images_with_few_objects} images have 1 or fewer objects")
            print(f"     These images will produce short GT strings (low supervision signal)")
        
        if self.augment:
            print(f"  ✅ Data augmentation enabled")
        
        if self.max_objects_per_image is not None:
            print(f"  ✅ Max objects per image: {self.max_objects_per_image}")
    
    def __len__(self):
        return len(self.image_ids)
    
    def _letterbox_resize(
        self,
        image: Image.Image,
        target_size: Tuple[int, int],  # (target_w, target_h)
        fill_color: Tuple[int, int, int] = (0, 0, 0)
    ) -> Tuple[Image.Image, float, int, int]:
        """
        Letterbox resize: Aspect ratio를 유지하며 이미지 resize + padding
        
        Args:
            image: PIL Image
            target_size: (target_width, target_height)
            fill_color: Padding 색상 (RGB)
        
        Returns:
            (resized_image, scale, pad_top, pad_left)
        """
        img_w, img_h = image.size
        target_w, target_h = target_size
        
        # Scale 계산 (aspect ratio 유지)
        scale = min(target_w / img_w, target_h / img_h)
        
        # Resize
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Padding 계산
        pad_left = (target_w - new_w) // 2
        pad_top = (target_h - new_h) // 2
        
        # Padding 적용
        canvas = Image.new('RGB', (target_w, target_h), fill_color)
        canvas.paste(resized, (pad_left, pad_top))
        
        return canvas, scale, pad_top, pad_left
    
    def _apply_augmentation(self, image: Image.Image) -> Image.Image:
        """
        PIL 기반 데이터 증강 (Color jitter만 적용)
        
        Flip, Rotation 제거: Bbox 좌표 변환 없이는 사용 불가
        Color jitter만 유지: 좌표 무관, 색상 변화로 일반화 능력 향상
        
        Args:
            image: PIL Image
        
        Returns:
            Augmented PIL Image
        """
        # Color jittering만 적용 (Bbox에 영향 없음)
        if random.random() < 0.8:
            # Brightness
            if random.random() < 0.5:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(random.uniform(*self.aug_brightness_range))
            
            # Contrast
            if random.random() < 0.5:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(random.uniform(*self.aug_contrast_range))
            
            # Saturation (색상 변화 - tomato ripeness에 중요!)
            if random.random() < 0.5:
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(random.uniform(*self.aug_saturation_range))
        
        return image
    
    def _get_processor_size(self) -> Tuple[int, int]:
        """
        Processor의 기본 이미지 크기 확인
        
        Returns:
            (width, height): Processor가 사용하는 이미지 크기
        """
        # Florence-2 processor의 기본 크기 확인
        if not hasattr(self, "_cached_processor_size"):
            dummy = Image.new("RGB", (64, 64))
            out = self.processor(images=dummy, return_tensors="pt")

            _, _, h, w = out["pixel_values"].shape
            self._cached_processor_size = (w, h)

        return self._cached_processor_size
    
    def __getitem__(self, idx: int) -> Tuple[Dict, str]:
        """
        Returns:
            inputs: Processor output (input_ids, pixel_values, etc.)
            target: Ground truth string for training
        """
        img_id = self.image_ids[idx]
        img_info = self.image_info[img_id]
        annotations = self.annotations_by_image[img_id]
        
        # ✅ 이미지 로드: EXIF orientation 보정 적용
        image_path = self.image_dir / img_info['file_name']
        try:
            image = load_image_exif_corrected(str(image_path))
        except Exception as e:
            print(f"Warning: Failed to load {image_path}: {e}")
            image = Image.new('RGB', (640, 640), color='black')  # Return dummy data
        
        # ✅ EXIF 보정 후 이미지 크기 (이게 "정답")
        original_width, original_height = image.size
        
        # COCO annotation 크기 (EXIF 보정 후 크기와 일치해야 함)
        annotation_width = img_info['width']
        annotation_height = img_info['height']
        
        # ✅ EXIF 보정 후에는 annotation 크기와 실제 이미지 크기가 일치해야 함
        # (COCO JSON이 ImageOps.exif_transpose(img).size로 수정되었다고 가정)
        if abs(original_width - annotation_width) > 1 or abs(original_height - annotation_height) > 1:
            if not hasattr(Florence2Dataset, '_exif_size_mismatch_warned'):
                print(f"\n⚠️ Warning: EXIF-corrected image size doesn't match annotation!")
                print(f"   EXIF-corrected size: {original_width}x{original_height}")
                print(f"   Annotation size: {annotation_width}x{annotation_height}")
                print(f"   Please fix COCO JSON using ImageOps.exif_transpose(img).size")
                print(f"   Using EXIF-corrected size for bbox conversion")
                Florence2Dataset._exif_size_mismatch_warned = True
        
        # 데이터 증강 적용 (training 시에만)
        if self.augment:
            image = self._apply_augmentation(image)

        # Letterbox target size를 processor가 기대하는 크기로
        target_width, target_height = self._get_processor_size()
        
        if self.use_letterbox:
            image, scale, pad_top, pad_left = self._letterbox_resize(
                image=image,
                target_size=(target_width, target_height),
                fill_color=(0, 0, 0)
            )
            expected_width = target_width
            expected_height = target_height
            processed_width = target_width
            processed_height = target_height
        else:
            scale, pad_top, pad_left = 1.0, 0, 0
            expected_width = original_width
            expected_height = original_height
            processed_width = original_width
            processed_height = original_height

        # Processor 호출
        inputs = self.processor(
            text=self.task_prompt,
            images=image,
            return_tensors="pt"
        )

        # Processor 출력 크기 확인
        if 'pixel_values' in inputs:
            if len(inputs['pixel_values'].shape) == 4:
                # pixel_values shape: [batch, channels, height, width]
                batch, channels, h, w = inputs['pixel_values'].shape
                actual_width = w
                actual_height = h
                processed_width = actual_width
                processed_height = actual_height
                
                # Letterbox 사용 시 크기 불일치 확인
                if self.use_letterbox:
                    if abs(actual_width - expected_width) > 1 or abs(actual_height - expected_height) > 1:
                        if not hasattr(self, '_letterbox_size_warning_printed'):
                            print(f"\n⚠️ Warning: Processor resized letterboxed image!")
                            print(f"   Expected: {expected_width}x{expected_height}")
                            print(f"   Actual: {actual_width}x{actual_height}")
                            print(f"   This may cause bbox misalignment!")
                            self._letterbox_size_warning_printed = True
        
        # max_objects 제한
        if self.max_objects_per_image is not None and len(annotations) > self.max_objects_per_image:
            if self.is_train:
                annotations = random.sample(annotations, self.max_objects_per_image)
            else:
                annotations = annotations[:self.max_objects_per_image]

        # ✅ GT 생성: annotation 크기와 실제 이미지 크기 불일치 시 스케일링 적용
        target_str = self._create_target_string(
            annotations=annotations,
            image_width=original_width,      # EXIF 보정 후 실제 이미지 크기
            image_height=original_height,    # EXIF 보정 후 실제 이미지 크기
            annotation_width=annotation_width,  # COCO annotation 크기
            annotation_height=annotation_height,  # COCO annotation 크기
            processed_width=processed_width,
            processed_height=processed_height,
            scale=scale,
            pad_top=pad_top,
            pad_left=pad_left
        )

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
       
        return inputs, target_str
    
    def _create_target_string(
        self,
        annotations: List[Dict],
        image_width: int,       # EXIF 보정 후 실제 이미지 크기
        image_height: int,      # EXIF 보정 후 실제 이미지 크기
        annotation_width: int,  # COCO annotation 크기 (bbox 좌표 기준)
        annotation_height: int, # COCO annotation 크기 (bbox 좌표 기준)
        processed_width: int,   # 최종 처리된 이미지 크기 (letterbox 후)
        processed_height: int,  # 최종 처리된 이미지 크기 (letterbox 후)
        scale: float,           # Letterbox scale
        pad_top: int,           # Letterbox padding (top)
        pad_left: int           # Letterbox padding (left)
    ) -> str:
        """
        COCO annotations를 Florence-2 format으로 변환
        
        Format: "<loc_x1><loc_y1><loc_x2><loc_y2>class_name<loc_x1><loc_y1>..."
        - 좌표는 0~999 scale로 정규화 (Florence-2 표준)
        - 형식: 좌표 먼저, 그 다음 클래스명
        - 여러 객체가 있으면 모두 포함하여 더 긴 supervision 제공
        
        Note: COCO bbox는 annotation width/height 기준이므로,
        실제 이미지 크기와 다를 수 있음 (EXIF 보정 전후 불일치)
        """
        if not annotations:
            # 객체가 없는 경우 빈 문자열 반환
            return ""
        
        target_parts = []
        
        # Annotation 크기와 실제 이미지 크기 불일치 확인
        # (COCO JSON이 아직 EXIF 보정 전 크기일 수 있음)
        if abs(annotation_width - image_width) > 1 or abs(annotation_height - image_height) > 1:
            scale_w_ann_to_actual = image_width / annotation_width if annotation_width > 0 else 1.0
            scale_h_ann_to_actual = image_height / annotation_height if annotation_height > 0 else 1.0
            needs_scaling = True
        else:
            scale_w_ann_to_actual = scale_h_ann_to_actual = 1.0
            needs_scaling = False
        
        for ann in annotations:
            # COCO bbox format: [x, y, width, height] (annotation 크기 기준)
            bbox = ann['bbox']
            category_id = ann['category_id']
            category_name = self.categories[category_id]
            
            # COCO bbox -> xyxy (annotation 크기 기준)
            x1, y1, w, h = bbox
            x2 = x1 + w
            y2 = y1 + h
            
            # ✅ Step 1: Annotation 크기 → 실제 이미지 크기 변환 (필요시)
            if needs_scaling:
                x1 = x1 * scale_w_ann_to_actual
                y1 = y1 * scale_h_ann_to_actual
                x2 = x2 * scale_w_ann_to_actual
                y2 = y2 * scale_h_ann_to_actual
            
            # ✅ Step 2: Letterbox 변환 적용 (실제 이미지 크기 기준)
            if self.use_letterbox:
                # Letterbox: scale + padding
                x1_p = x1 * scale + pad_left
                y1_p = y1 * scale + pad_top
                x2_p = x2 * scale + pad_left
                y2_p = y2 * scale + pad_top
            else:
                # Letterbox 미사용: 단순 비율 변환
                scale_w = processed_width / image_width if image_width > 0 else 1.0
                scale_h = processed_height / image_height if image_height > 0 else 1.0
                x1_p = x1 * scale_w
                y1_p = y1 * scale_h
                x2_p = x2 * scale_w
                y2_p = y2 * scale_h
            
            # 최종 이미지 크기를 기준으로 정규화 (0~999 scale)
            if processed_width > 0 and processed_height > 0:
                x1_norm = int((x1_p / processed_width) * 999)
                y1_norm = int((y1_p / processed_height) * 999)
                x2_norm = int((x2_p / processed_width) * 999)
                y2_norm = int((y2_p / processed_height) * 999)
            else:
                x1_norm = y1_norm = x2_norm = y2_norm = 0
            
            # Clamp
            x1_norm = max(0, min(999, x1_norm))
            y1_norm = max(0, min(999, y1_norm))
            x2_norm = max(0, min(999, x2_norm))
            y2_norm = max(0, min(999, y2_norm))
            
            # Florence-2 location token format: <loc...>class_name (좌표 → 클래스)
            loc_str = f"<loc_{x1_norm}><loc_{y1_norm}><loc_{x2_norm}><loc_{y2_norm}>"
            target_parts.append(f"{loc_str}{category_name}")
        
        # 모든 객체를 연결
        target_str = "".join(target_parts)
        
        return target_str

    @staticmethod
    def collate_fn(batch, processor=None, debug=False):
        """
        Custom collate function for DataLoader
        
        Args:
            batch: List of (inputs, target_str) tuples
            processor: Processor for tokenization (디버깅용)
            debug: 디버깅 정보 출력 여부
        
        Returns:
            inputs: Dict with batched tensors
            targets: List of target strings
        """
        inputs_list = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        
        # Stack tensors
        batched_inputs = {
            'input_ids': torch.stack([inp['input_ids'] for inp in inputs_list]),
            'pixel_values': torch.stack([inp['pixel_values'] for inp in inputs_list])
        }
        
        # attention_mask가 있으면 추가
        if 'attention_mask' in inputs_list[0]:
            batched_inputs['attention_mask'] = torch.stack([inp['attention_mask'] for inp in inputs_list])
        
        return batched_inputs, targets


def get_florence2_dataloaders(
    config: Dict,
    processor,
    batch_size: int = 8,
    num_workers: int = 4
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders for Florence-2
    
    Args:
        config: Configuration dict
        processor: Florence-2 processor
        batch_size: Batch size
        num_workers: Number of workers for DataLoader
    
    Returns:
        train_loader, val_loader
    """
    from ..datasets import DATASET_REGISTRY
    
    dataset_name = config['data']['dataset_name']
    dataset_meta = DATASET_REGISTRY.get(dataset_name)
    
    if dataset_meta is None:
        raise ValueError(f"Dataset '{dataset_name}' not found in registry")
    
    # Florence-2 설정에서 augmentation 범위 읽기
    florence2_config = config.get('florence2', {})
    aug_config = florence2_config.get('augmentation', {})
    
    use_letterbox = florence2_config.get('use_letterbox', True)
    aug_brightness_range = tuple(aug_config.get('brightness_range', [0.8, 1.2]))
    aug_contrast_range = tuple(aug_config.get('contrast_range', [0.8, 1.2]))
    aug_saturation_range = tuple(aug_config.get('saturation_range', [0.8, 1.2]))
    max_objects_per_image = florence2_config.get('max_objects_per_image', None)
    
    # Train dataset (with augmentation)
    train_paths = dataset_meta.get_data_paths('train', config)
    train_dataset = Florence2Dataset(
        coco_ann_file=train_paths['ann_file'],
        image_dir=train_paths['img_folder'],
        processor=processor,
        task_prompt="<OD>",
        is_train=True,     
        augment=True,
        use_letterbox=use_letterbox,
        aug_brightness_range=aug_brightness_range,
        aug_contrast_range=aug_contrast_range,
        aug_saturation_range=aug_saturation_range,
        max_objects_per_image=max_objects_per_image
    )
    
    # Validation dataset (no augmentation, but letterbox 사용)
    val_paths = dataset_meta.get_data_paths('val', config)
    val_dataset = Florence2Dataset(
        coco_ann_file=val_paths['ann_file'],
        image_dir=val_paths['img_folder'],
        processor=processor,
        task_prompt="<OD>",
        is_train=False,   # Validation에서는 증강 비활성화
        augment=False,
        use_letterbox=use_letterbox,  # Letterbox는 validation에서도 사용
        aug_brightness_range=aug_brightness_range,
        aug_contrast_range=aug_contrast_range,
        aug_saturation_range=aug_saturation_range,
        max_objects_per_image=max_objects_per_image
    )
    
    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=Florence2Dataset.collate_fn,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=Florence2Dataset.collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader
    