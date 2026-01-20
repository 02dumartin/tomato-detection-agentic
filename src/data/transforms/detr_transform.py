"""DETR용 데이터 변환 및 Dataset"""
import torchvision
from torchvision import transforms as T
from transformers import DetrImageProcessor
import random
import math
import numpy as np
from PIL import Image
from pathlib import Path

class DetrCocoDataset(torchvision.datasets.CocoDetection):
    """
    DETR 학습을 위한 커스텀 COCO Detection 데이터셋 클래스
    """
    
    def __init__(
        self, 
        img_folder: str, 
        ann_file: str,
        imageprocessor: DetrImageProcessor, 
        train: bool = True,
        config=None
    ):
        super(DetrCocoDataset, self).__init__(img_folder, ann_file)
        
        self.imageprocessor = imageprocessor
        self.train = train
        self.config = config or {}
        
        # Augmentation 설정 가져오기
        aug_config = self.config.get('data', {}).get('augmentation', {})
        
        # 색상 증강 (bbox 수정 불필요)
        color_aug = []
        if aug_config.get('color_jitter'):
            cj = aug_config['color_jitter']
            color_aug.append(T.ColorJitter(
                brightness=cj.get('brightness', 0.4),
                contrast=cj.get('contrast', 0.4),
                saturation=cj.get('saturation', 0.7),
                hue=cj.get('hue', 0.015)
            ))
        if aug_config.get('sharpness'):
            sh = aug_config['sharpness']
            color_aug.append(T.RandomAdjustSharpness(
                sharpness_factor=sh.get('factor', 1.5),
                p=sh.get('prob', 0.2)
            ))
        
        self.color_augment = T.Compose(color_aug) if color_aug and train else None
        
        # 기하학적 증강 설정 (bbox 변환 필요)
        self.horizontal_flip_prob = aug_config.get('horizontal_flip', 0.0) if train else 0.0
        self.vertical_flip_prob = aug_config.get('vertical_flip', 0.0) if train else 0.0
        self.rotation_degrees = aug_config.get('rotation', 0.0) if train else 0.0
        self.translation = aug_config.get('translation', 0.0) if train else 0.0
        self.scale = aug_config.get('scale', 0.0) if train else 0.0
        self.shear = aug_config.get('shear', 0.0) if train else 0.0
        self.perspective = aug_config.get('perspective', 0.0) if train else 0.0

    def _apply_geometric_augmentation(self, img, target):
        """기하학적 증강 적용 (bbox 좌표 변환 포함)"""
        width, height = img.size
        
        # target이 리스트인지 딕셔너리인지 확인
        if isinstance(target, list):
            annotations = target
        elif isinstance(target, dict):
            annotations = target.get('annotations', [])
        else:
            annotations = []
        
        # Horizontal flip
        if random.random() < self.horizontal_flip_prob:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            for ann in annotations:
                bbox = ann['bbox']  # COCO format: [x, y, w, h]
                # 좌우 반전: x 좌표만 변경
                ann['bbox'] = [width - bbox[0] - bbox[2], bbox[1], bbox[2], bbox[3]]
        
        # Vertical flip
        if random.random() < self.vertical_flip_prob:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            for ann in annotations:
                bbox = ann['bbox']
                # 상하 반전: y 좌표만 변경
                ann['bbox'] = [bbox[0], height - bbox[1] - bbox[3], bbox[2], bbox[3]]
        
        # Rotation (간단한 구현 - 중심 기준 회전)
        if self.rotation_degrees > 0:
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            if abs(angle) > 1e-6:
                # 이미지 회전
                img = img.rotate(angle, expand=False, fillcolor=(128, 128, 128))
                # bbox 회전 변환 (중심 기준)
                center_x, center_y = width / 2, height / 2
                rad = math.radians(angle)
                cos_a, sin_a = math.cos(rad), math.sin(rad)
                
                for ann in annotations:
                    bbox = ann['bbox']
                    x, y, w, h = bbox
                    # bbox의 네 모서리 좌표
                    corners = [
                        [x, y],           # top-left
                        [x + w, y],      # top-right
                        [x + w, y + h],  # bottom-right
                        [x, y + h]       # bottom-left
                    ]
                    
                    # 각 모서리를 중심 기준으로 회전
                    rotated_corners = []
                    for cx, cy in corners:
                        # 중심으로 이동
                        dx, dy = cx - center_x, cy - center_y
                        # 회전
                        rx = dx * cos_a - dy * sin_a
                        ry = dx * sin_a + dy * cos_a
                        # 다시 원래 위치로
                        rotated_corners.append([rx + center_x, ry + center_y])
                    
                    # 회전된 bbox의 최소/최대 좌표 계산
                    xs = [c[0] for c in rotated_corners]
                    ys = [c[1] for c in rotated_corners]
                    new_x = max(0, min(xs))
                    new_y = max(0, min(ys))
                    new_w = min(width, max(xs)) - new_x
                    new_h = min(height, max(ys)) - new_y
                    
                    # 유효한 bbox인지 확인
                    if new_w > 0 and new_h > 0:
                        ann['bbox'] = [new_x, new_y, new_w, new_h]
                    else:
                        # 유효하지 않은 경우 원본 유지 (또는 제거)
                        pass
        
        # Translation (이동)
        if self.translation > 0:
            tx = random.uniform(-self.translation, self.translation) * width
            ty = random.uniform(-self.translation, self.translation) * height
            
            if abs(tx) > 1e-6 or abs(ty) > 1e-6:
                # 이미지 이동 (affine transform 사용)
                from torchvision.transforms.functional import affine
                img_tensor = T.ToTensor()(img)
                img_tensor = affine(
                    img_tensor,
                    angle=0,
                    translate=[int(tx), int(ty)],
                    scale=1.0,
                    shear=[0.0, 0.0]
                )
                img = T.ToPILImage()(img_tensor)
                
                # bbox 이동
                for ann in annotations:
                    bbox = ann['bbox']
                    new_x = max(0, min(width, bbox[0] + tx))
                    new_y = max(0, min(height, bbox[1] + ty))
                    # bbox가 이미지 밖으로 나가지 않도록 조정
                    new_w = min(bbox[2], width - new_x)
                    new_h = min(bbox[3], height - new_y)
                    if new_w > 0 and new_h > 0:
                        ann['bbox'] = [new_x, new_y, new_w, new_h]
        
        # Scale (크기 조정)
        if self.scale > 0:
            scale_factor = random.uniform(1.0 - self.scale, 1.0 + self.scale)
            if abs(scale_factor - 1.0) > 1e-6:
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                img = img.resize((new_width, new_height), Image.BILINEAR)
                
                # bbox 스케일 조정
                for ann in annotations:
                    bbox = ann['bbox']
                    ann['bbox'] = [
                        bbox[0] * scale_factor,
                        bbox[1] * scale_factor,
                        bbox[2] * scale_factor,
                        bbox[3] * scale_factor
                    ]
                
                # 이미지를 원래 크기로 리사이즈 (crop or pad)
                if scale_factor > 1.0:
                    # 크롭
                    left = (new_width - width) // 2
                    top = (new_height - height) // 2
                    img = img.crop((left, top, left + width, top + height))
                    # bbox 조정
                    for ann in annotations:
                        bbox = ann['bbox']
                        ann['bbox'] = [
                            max(0, bbox[0] - left),
                            max(0, bbox[1] - top),
                            min(width, bbox[2]),
                            min(height, bbox[3])
                        ]
                else:
                    # 패딩
                    new_img = Image.new(img.mode, (width, height), (128, 128, 128))
                    left = (width - new_width) // 2
                    top = (height - new_height) // 2
                    new_img.paste(img, (left, top))
                    img = new_img
                    # bbox 조정
                    for ann in annotations:
                        bbox = ann['bbox']
                        ann['bbox'] = [
                            bbox[0] + left,
                            bbox[1] + top,
                            bbox[2],
                            bbox[3]
                        ]
        
        # Shear와 Perspective는 복잡하므로 일단 생략 (필요시 추가)
        # 실제로는 torchvision의 functional.affine이나 functional.perspective 사용
        
        return img, target

    def __getitem__(self, idx: int):
        # PIL 이미지와 COCO 형식의 타겟 읽기
        # super().__getitem__은 (img, target)을 반환하고, target은 COCO annotations 리스트
        img, target = super().__getitem__(idx)
        
        # 이미지 회전 처리 (COCO annotation의 orientation 정보 사용)
        from ...utils.vis_utils import load_and_orient_image
        image_id = self.ids[idx]
        # 이미지 경로 가져오기
        img_info = self.coco.loadImgs(image_id)[0]
        img_path = Path(self.root) / img_info['file_name']
        # 회전 처리된 이미지로 교체
        img = load_and_orient_image(img_path, self.coco.dataset, image_id)

        # 색상 증강 (bbox 수정 불필요)
        if self.color_augment is not None:
            img = self.color_augment(img)

        # 기하학적 증강 (bbox 변환 필요)
        # target은 COCO annotations 리스트 형식
        if self.train and (self.horizontal_flip_prob > 0 or self.vertical_flip_prob > 0 or 
                          self.rotation_degrees > 0 or self.translation > 0 or self.scale > 0):
            img, target = self._apply_geometric_augmentation(img, target)

        # DETR 형식으로 이미지와 타겟 전처리
        # target은 여전히 COCO annotations 리스트
        image_id = self.ids[idx]
        target_dict = {'image_id': image_id, 'annotations': target}
        encoding = self.imageprocessor(images=img, annotations=target_dict, return_tensors="pt")
        
        # 배치 차원 제거
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target

    @staticmethod
    def create_collate_fn(imageprocessor: DetrImageProcessor):
        """
        DETR용 collate function 생성
        Dataset과 함께 사용하는 collate_fn
        """
        def collate_fn(batch):
            pixel_values = [item[0] for item in batch]
            labels = [item[1] for item in batch]
            
            encoding = imageprocessor.pad(pixel_values, return_tensors="pt")
            
            return {
                'pixel_values': encoding['pixel_values'],
                'pixel_mask': encoding['pixel_mask'],
                'labels': labels
            }
        return collate_fn

def create_detr_dataset(dataset_meta, split: str, imageprocessor, config=None):
    """DETR Dataset 생성"""
    paths = dataset_meta.get_data_paths(split, config=config)
    train = (split == "train")
    
    return DetrCocoDataset(
        paths['img_folder'],
        paths['ann_file'],
        imageprocessor,
        train,
        config
    )