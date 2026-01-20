"""
Florence-2 Fine-tuned Model
TomatOD 데이터셋으로 fine-tuning된 Florence-2
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Fork 경고 방지

import torch
from .florence2_base import Florence2Base
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from transformers import get_scheduler
from torch.optim import AdamW
from tqdm import tqdm


class Florence2Finetuned(Florence2Base):
    """Fine-tuned Florence-2"""
    
    def __init__(self, checkpoint_path: str, config: Optional[Dict] = None, **kwargs):
        super().__init__(
            is_finetuned=True,
            checkpoint_path=checkpoint_path,
            config=config,
            **kwargs
        )
        
        # Fine-tuned 모델의 클래스 맵
        # Config에서 class_names 가져오기 (없으면 기본값 사용)
        if config and 'data' in config and 'class_names' in config['data']:
            class_names = config['data']['class_names']
            self.class_map = {i: name for i, name in enumerate(class_names)}
        else:
            # 기본값 (3-class)
            self.class_map = {
                0: 'fully_ripe',
                1: 'semi_ripe',
                2: 'unripe'
            }
        
        self.id_to_name = self.class_map
        self.name_to_id = {v: k for k, v in self.class_map.items()}
        
        print(f" Fine-tuned mode initialized")
        print(f"   Classes: {list(self.class_map.values())}")
    
    def _calculate_surrogate_score(self, bbox: List[float], image_size: Tuple[int, int]) -> float:
        """
        Confidence score 대신 사용할 surrogate score 계산
        
        Florence-2는 confidence를 제공하지 않으므로, 여러 지표를 조합하여 점수 생성:
        - Bbox 크기: 큰 객체일수록 높은 점수 (일반적으로 더 확실)
        - 이미지 중앙 거리: 중앙에 가까울수록 높은 점수
        - Bbox aspect ratio: 정사각형에 가까울수록 높은 점수 (tomato는 원형)
        
        Args:
            bbox: [x1, y1, x2, y2] in pixel coordinates
            image_size: (width, height)
        
        Returns:
            Surrogate score (0.2~0.95 범위)
        """
        x1, y1, x2, y2 = bbox
        img_w, img_h = image_size
        width = x2 - x1
        height = y2 - y1
        bbox_area = width * height
        img_area = img_w * img_h
        
        # 최소 크기 체크 (너무 작은 박스는 매우 낮은 점수)
        min_area_ratio = 0.0005  # 이미지의 0.05% 미만이면 매우 낮은 점수
        area_ratio = bbox_area / img_area if img_area > 0 else 0.0
        
        # 1. 크기 기반 점수 (정규화) - 작은 박스에 강한 패널티
        if area_ratio < min_area_ratio:
            # 너무 작은 박스는 점수를 크게 깎음
            size_score = (area_ratio / min_area_ratio) * 0.3  # 최대 0.3
        else:
            # 정상 크기는 선형 스케일링
            size_score = min(1.0, area_ratio / 0.01)  # 이미지의 1%를 최대값으로
            size_score = max(0.3, size_score)  # 최소 0.3
        
        # 2. 중앙 거리 기반 점수
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        img_center_x = img_w / 2
        img_center_y = img_h / 2
        dist_from_center = ((center_x - img_center_x)**2 + (center_y - img_center_y)**2)**0.5
        max_dist = ((img_w/2)**2 + (img_h/2)**2)**0.5
        center_score = 1.0 - (dist_from_center / max_dist) if max_dist > 0 else 0.0
        center_score = max(0.0, min(1.0, center_score))
        
        # 3. Aspect ratio 기반 (tomato는 원형이므로 1:1에 가까울수록 좋음)
        if max(width, height) > 0:
            aspect_ratio = min(width, height) / max(width, height)
        else:
            aspect_ratio = 0.0
        aspect_score = aspect_ratio
        
        # 4. 최소 크기 필터 (너무 작거나 큰 박스는 점수 감소)
        min_dim = min(img_w, img_h)
        min_size_ratio = 0.02  # 이미지의 2% 미만 너비/높이는 감점
        max_size_ratio = 0.5   # 이미지의 50% 이상 너비/높이는 감점
        
        size_penalty = 1.0
        if width < min_dim * min_size_ratio or height < min_dim * min_size_ratio:
            size_penalty = 0.5  # 작은 박스 감점
        elif width > min_dim * max_size_ratio or height > min_dim * max_size_ratio:
            size_penalty = 0.7  # 큰 박스도 약간 감점
        
        # 가중 평균 (크기 50%, 중앙 25%, aspect 25%)
        surrogate_score = 0.5 * size_score + 0.25 * center_score + 0.25 * aspect_score
        surrogate_score *= size_penalty  # 크기 패널티 적용
        
        # 0.2~0.95 범위로 정규화 (작은 박스는 낮은 점수)
        surrogate_score = 0.2 + 0.75 * max(0.0, min(1.0, surrogate_score))
        
        return float(surrogate_score)
    
    @staticmethod
    def _calculate_iou(box1: List[float], box2: List[float]) -> float:
        """
        두 bbox 간의 IoU 계산
        
        Args:
            box1, box2: [x1, y1, x2, y2]
        
        Returns:
            IoU 값 (0~1)
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 교집합 영역 계산
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        
        # 각 박스의 영역
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    @staticmethod
    def _apply_nms(detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """
        Non-Maximum Suppression (NMS) 적용하여 중복 박스 제거
        
        Args:
            detections: List of detection dicts with 'bbox' and 'score'
            iou_threshold: IoU threshold for NMS
        
        Returns:
            Filtered detections after NMS
        """
        if not detections:
            return []
        
        # Score 기준으로 내림차순 정렬
        sorted_detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        
        keep = []
        while sorted_detections:
            # 가장 높은 score의 박스를 keep
            current = sorted_detections.pop(0)
            keep.append(current)
            
            # 현재 박스와 IoU가 높은 박스들 제거
            remaining = []
            for det in sorted_detections:
                iou = Florence2Finetuned._calculate_iou(current['bbox'], det['bbox'])
                if iou < iou_threshold:
                    remaining.append(det)
            sorted_detections = remaining
        
        return keep
    
    def _normalize_bbox_to_pixel(self, bbox: List[float], image_size: Tuple[int, int]) -> List[float]:
        """
        Bbox를 픽셀 좌표로 변환 (정규화된 경우)
        
        Args:
            bbox: [x1, y1, x2, y2] (정규화 또는 픽셀)
            image_size: (width, height)
        
        Returns:
            [x1, y1, x2, y2] in pixel coordinates (x1 <= x2, y1 <= y2 보장)
        """
        x1, y1, x2, y2 = bbox
        img_w, img_h = image_size
        
        # 0~1 정규화인 경우
        if 0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1:
            x1 = x1 * img_w
            y1 = y1 * img_h
            x2 = x2 * img_w
            y2 = y2 * img_h
        
        # 0~1000 정규화인 경우 (Florence-2 location token)
        elif 0 <= x1 <= 1000 and 0 <= y1 <= 1000 and 0 <= x2 <= 1000 and 0 <= y2 <= 1000:
            x1 = (x1 / 1000) * img_w
            y1 = (y1 / 1000) * img_h
            x2 = (x2 / 1000) * img_w
            y2 = (y2 / 1000) * img_h
        
        # 이미 픽셀인 경우 그대로 반환 (이미지 크기 범위 내)
        # 범위를 벗어나면 경고
        if x1 < 0 or y1 < 0 or x2 > img_w * 1.5 or y2 > img_h * 1.5:
            # 이상한 값이지만 일단 그대로 반환 (나중에 필터링)
            pass
        
        # 좌표 정렬: x1 <= x2, y1 <= y2 보장
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # 이미지 범위로 클램핑
        x1 = max(0, min(x1, img_w))
        y1 = max(0, min(y1, img_h))
        x2 = max(x1, min(x2, img_w))  # x2 >= x1 보장
        y2 = max(y1, min(y2, img_h))  # y2 >= y1 보장
        
        return [x1, y1, x2, y2]
    
    def predict_finetuned(
        self, 
        image_path: str,
        conf_threshold: float = 0.0
    ) -> List[Dict]:
        """
        Fine-tuned 모델로 추론
        
        Fine-tuning 후에는 <OD> task로 직접 3개 클래스 예측
        
        Args:
            image_path: 이미지 경로
            conf_threshold: 신뢰도 threshold (⚠️ 주의: Florence-2는 confidence 제공 안 함, 
                            surrogate score로 대체됨)
        
        Returns:
            List[Dict]: 탐지 결과
                - bbox: [x1, y1, x2, y2] in pixel coordinates
                - class: 클래스 이름
                - class_id: 클래스 ID
                - score: Surrogate score (0.3~0.95 범위)
        """
        from PIL import Image
        
        # 이미지 크기 확인을 위해 로드
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path
        image_size = (image.width, image.height)
        
        result = self.predict(image, task="<OD>")
        
        detections = []
        
        if '<OD>' in result:
            det = result['<OD>']
            bboxes = det.get('bboxes', [])
            labels = det.get('labels', [])
            
            original_count = len(bboxes)
            
            # 디버깅: 원본 모델 출력 확인 (첫 번째 이미지만)
            if not hasattr(self, '_debug_model_output_printed'):
                print(f"\n[Debug] Model Raw Output:")
                print(f"  Total raw detections: {original_count}")
                if original_count > 0:
                    unique_labels = set(labels[:20])  # 처음 20개로 증가
                    print(f"  Sample labels: {list(unique_labels)}")
                    print(f"  Available class names in name_to_id: {list(self.name_to_id.keys())}")
                    if bboxes:
                        sample_bbox = bboxes[0]
                        print(f"  Sample bbox format: {sample_bbox} (type: {type(sample_bbox)})")
                    
                    # 레이블 분포 확인
                    from collections import Counter
                    label_counts = Counter(labels)
                    print(f"  Top 10 labels: {dict(label_counts.most_common(10))}")
                self._debug_model_output_printed = True
            
            # 1단계: 모든 예측 수집 및 필터링
            # Zero-shot과 유사하게 단순화: 모델의 원본 출력을 신뢰하되, 최소한의 필터만 적용
            all_detections = []
            img_w, img_h = image_size
            min_dim = min(img_w, img_h)
            
            # 필터 설정 (Zero-shot처럼 최소한의 필터만)
            min_size = min_dim * 0.01  # 1% (너무 작은 박스만 제거)
            max_size = min_dim * 0.9   # 90% (너무 큰 박스만 제거)
            
            filtered_by_size = 0
            filtered_by_aspect = 0
            filtered_by_class = 0
            filtered_by_score = 0
            
            for bbox, label in zip(bboxes, labels):
                # Bbox 좌표를 픽셀 좌표로 변환 (정규화된 경우)
                bbox_pixel = self._normalize_bbox_to_pixel(bbox, image_size)
                
                # 최소 크기 필터 (너무 작은 박스만 제거)
                x1, y1, x2, y2 = bbox_pixel
                width = x2 - x1
                height = y2 - y1
                
                # 최소 크기 필터: 이미지의 1% 이상 (Zero-shot처럼 관대하게)
                if width < min_size or height < min_size:
                    filtered_by_size += 1
                    continue
                
                # 최대 크기 필터: 이미지의 90% 이하
                if width > max_size or height > max_size:
                    filtered_by_size += 1
                    continue
                
                # Aspect ratio 필터 (너무 길쭉한 박스만 제거, Zero-shot처럼 관대하게)
                if max(width, height) > 0:
                    aspect_ratio = min(width, height) / max(width, height)
                    if aspect_ratio < 0.3:  # 0.6 -> 0.3로 더 관대하게 (Zero-shot처럼)
                        filtered_by_aspect += 1
                        continue
                
                # 레이블을 클래스 이름으로 변환 (Zero-shot처럼 단순하게)
                class_name = self._parse_label(label)
                
                # None이거나 매칭 실패한 경우 필터링
                if class_name is None or class_name not in self.name_to_id:
                    filtered_by_class += 1
                    continue
                
                # Surrogate score 계산 (Zero-shot은 score=1.0 사용하지만, fine-tuned는 계산)
                surrogate_score = self._calculate_surrogate_score(bbox_pixel, image_size)
                
                # Confidence threshold 필터링 (Zero-shot은 필터링 안 하지만, fine-tuned는 적용)
                if surrogate_score < conf_threshold:
                    filtered_by_score += 1
                    continue
                
                all_detections.append({
                    'bbox': bbox_pixel,
                    'class': class_name,
                    'class_id': self.name_to_id[class_name],
                    'score': surrogate_score,
                    'label': label
                })
            
            before_nms_count = len(all_detections)
            
            # 2단계: NMS 적용 (중복 박스 제거) - IoU threshold를 더 높게
            detections = self._apply_nms(all_detections, iou_threshold=0.4)  # 0.5 -> 0.4로 더 엄격하게
            
            after_nms_count = len(detections)
            
            # 디버깅 정보 (첫 번째 이미지만 출력)
            if not hasattr(self, '_debug_printed'):
                print(f"\n[Debug] Filtering Statistics:")
                print(f"  Original detections: {original_count}")
                print(f"  Filtered by size: {filtered_by_size}")
                print(f"  Filtered by aspect ratio: {filtered_by_aspect}")
                print(f"  Filtered by class: {filtered_by_class}")
                print(f"  Filtered by score (threshold={conf_threshold}): {filtered_by_score}")
                print(f"  Before NMS: {before_nms_count}")
                print(f"  After NMS: {after_nms_count}")
                print(f"  Final detections: {after_nms_count}")
                self._debug_printed = True
        
        return detections
    
    def _parse_label(self, label: str) -> str:
        """
        모델 출력 레이블을 클래스 이름으로 변환
        
        Fine-tuning 시 학습된 레이블 형식에 따라 수정 필요
        엄격한 매칭: 정확히 'tomato'만 허용
        """
        label_lower = label.lower().strip()
        
        # 허용된 레이블 목록 (정확히 매칭)
        allowed_labels = {
            'tomato': 'tomato',
            'taato': 'tomato',  # 오타 처리
            'tomatoes': 'tomato',  # 복수형
            'tomatos': 'tomato',  # 오타 복수형
        }
        
        # 정확히 매칭되는 경우만 허용
        if label_lower in allowed_labels:
            class_name = allowed_labels[label_lower]
            if class_name in self.name_to_id:
                return class_name
        
        # 정확히 'tomato'만 포함하고 다른 단어가 없는 경우만 허용
        # 'tom tomato', 'tomtomandy' 같은 이상한 레이블은 제외
        if label_lower == 'tomato':
            if 'tomato' in self.name_to_id:
                return 'tomato'
        
        # 3-class 데이터셋용 키워드 매칭
        if 'fully' in label_lower or 'full' in label_lower:
            if 'fully_ripe' in self.name_to_id:
                return 'fully_ripe'
        elif 'semi' in label_lower or 'partial' in label_lower:
            if 'semi_ripe' in self.name_to_id:
                return 'semi_ripe'
        elif 'unripe' in label_lower or 'green' in label_lower:
            if 'unripe' in self.name_to_id:
                return 'unripe'
        
        # 정확히 매칭되는 경우
        if label in self.name_to_id:
            return label
        
        # 매칭 실패: None 반환하여 필터링되도록
        return None
    
    def predict_to_coco_format(
        self,
        image_path: str,
        image_id: int,
        conf_threshold: float = 0.0
    ) -> List[Dict]:
        """
        COCO 형식으로 예측 결과 반환
        
        Args:
            image_path: 이미지 경로
            image_id: COCO image ID
            conf_threshold: 신뢰도 threshold
        
        Returns:
            List[Dict]: COCO 형식의 detection 리스트
        """
        detections = self.predict_finetuned(
            image_path, 
            conf_threshold=conf_threshold
        )
        
        coco_results = []
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            width = x2 - x1
            height = y2 - y1
            
            coco_results.append({
                'image_id': image_id,
                'category_id': det['class_id'],
                'bbox': [x1, y1, width, height],  # COCO format: [x, y, w, h]
                'score': det['score'],
                'category_name': det['class']
            })
        
        return coco_results
    
    @staticmethod
    def train_model(
        config: Dict,
        output_dir: str,
        num_epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.0005,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 8,
        use_amp: bool = True,
        gradient_clip_val: float = 1.0,          
        accumulate_grad_batches: int = 1,       
        device: str = 'cuda'
    ):
        """
        Florence-2 Fine-tuning
        
        Args:
            config: Configuration dict
            output_dir: Output directory for checkpoints
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            use_lora: Use LoRA for efficient fine-tuning
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            device: Device ('cuda' or 'cpu')
        """
        from transformers import AutoProcessor, AutoModelForCausalLM
        from ..data.transforms.florence2_transform import get_florence2_dataloaders
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 통합 학습 관리자 초기화
        from ..utils.training import TrainingManager
        manager = TrainingManager(
            output_dir=output_dir,
            config=config,
            model_name="florence2",
            save_best=True,
            save_last=True
        )
        
        print("\n" + "="*70)
        print("FLORENCE-2 FINE-TUNING")
        print("="*70)
        print(f"Output directory: {output_dir}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Weight decay: {weight_decay}")
        print(f"Use AMP: {use_amp}")
        print(f"Gradient clip value: {gradient_clip_val}")      
        print(f"Accumulate grad batches: {accumulate_grad_batches}") 
        print(f"Use LoRA: {use_lora}")
        if use_lora:
            print(f"LoRA rank: {lora_r}, alpha: {lora_alpha}")
        print("="*70 + "\n")
        
        # 1. Load processor and model
        print("Loading Florence-2 base model...")
        from transformers import AutoProcessor, AutoModelForCausalLM
        
        processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-base",
            trust_remote_code=True
        )
        
        # Florence-2는 AutoModelForCausalLM을 사용 (Florence2ForConditionalGeneration은 존재하지 않음)
        # processor가 이미지와 텍스트를 함께 처리할 때 자동으로 이미지 placeholder 토큰을 삽입함
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-base",
            trust_remote_code=True,
            attn_implementation="eager"
        ).to(device)
        print("   Using AutoModelForCausalLM (Florence-2 standard)")
        
        # CRITICAL FIX: Resize token embeddings to match tokenizer vocab size
        tokenizer_vocab = len(processor.tokenizer)
        embed_vocab = model.get_input_embeddings().weight.shape[0]
        model_vocab_size = getattr(model.config, "vocab_size", None)
        
        print(f"\n[Vocab Size Check]")
        print(f"  Tokenizer vocab size: {tokenizer_vocab}")
        print(f"  Model embedding vocab size: {embed_vocab}")
        print(f"  Model config vocab_size: {model_vocab_size}")
        
        if tokenizer_vocab != embed_vocab:
            print(f"    Mismatch detected! Resizing embeddings from {embed_vocab} to {tokenizer_vocab}")
            model.resize_token_embeddings(tokenizer_vocab)
            embed_vocab = model.get_input_embeddings().weight.shape[0]
            print(f"   Resized to: {embed_vocab}")
        else:
            print(f"   Vocab sizes match!")
        
        # 2. Apply LoRA if enabled
        if use_lora:
            from peft import LoraConfig, get_peft_model
            
            print("Applying LoRA...")
            # Florence-2는 CausalLM 기반이므로 CAUSAL_LM 사용
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        
        # 3. Create dataloaders
        print("\nPreparing datasets...")
        train_loader, val_loader = get_florence2_dataloaders(
            config=config,
            processor=processor,
            batch_size=batch_size,
            num_workers=4
        )
        
        # 4. Setup optimizer and scheduler
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        num_training_steps = len(train_loader) * num_epochs
        
        # Learning rate schedule 설정
        scheduler_type = config.get('training', {}).get('lr_scheduler', 'linear')
        warmup_ratio = config.get('training', {}).get('warmup_ratio', 0.1)
        num_warmup_steps = int(num_training_steps * warmup_ratio)
        
        if scheduler_type == 'cosine':
            scheduler = get_scheduler(
                "cosine",
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
            print(f"Using Cosine learning rate schedule (warmup: {warmup_ratio*100:.1f}%)")
        else:
            scheduler = get_scheduler(
                "linear",
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
            print(f"Using Linear learning rate schedule (warmup: {warmup_ratio*100:.1f}%)")
        
        # 5. Setup AMP scaler
        if use_amp:
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()
            print("Using Automatic Mixed Precision (AMP)")
        
        # 6. Training loop with Early Stopping
        print(f"\nStarting training for {num_epochs} epochs...")
        best_val_loss = float('inf')
        patience = config.get('training', {}).get('patience', 20)
        patience_counter = 0
        
        print(f"Early stopping patience: {patience} epochs\n")
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            
            # Initialize gradient accumulation
            optimizer.zero_grad()
            
            for batch_idx, (inputs, targets) in enumerate(train_pbar):
                # Move inputs to device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                #  Florence-2 방식: processor 출력을 그대로 사용
                # inputs는 이미 processor(text="<OD>", images=images)의 출력
                # processor가 자동으로 이미지 placeholder 토큰을 input_ids에 삽입함
                # - input_ids: prompt + 이미지 placeholder 토큰 포함
                # - pixel_values: 이미지 임베딩
                # - attention_mask: processor가 생성한 mask
                
                # Target tokenize (decoder labels)
                tokenized_targets = processor.tokenizer(
                    text=targets,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024,
                    add_special_tokens=False
                )
                labels = tokenized_targets.input_ids.to(device)
                
                # ✅ CRITICAL: Pad 토큰을 -100으로 변경 (loss 계산에서 제외)
                pad_token_id = processor.tokenizer.pad_token_id
                if pad_token_id is not None:
                    labels[labels == pad_token_id] = -100
                
                # 디버깅 정보 (첫 배치에서만)
                if batch_idx == 0 and epoch == 0:
                    print(f"\n  [Debug] Florence-2 training mode:")
                    print(f"    inputs keys: {list(inputs.keys())}")
                    print(f"    input_ids shape: {inputs['input_ids'].shape}")
                    print(f"    input_ids sample (first 20): {inputs['input_ids'][0][:20].tolist()}")
                    print(f"    pixel_values shape: {inputs['pixel_values'].shape}")
                    if 'attention_mask' in inputs:
                        print(f"    attention_mask shape: {inputs['attention_mask'].shape}")
                    print(f"    labels shape: {labels.shape}")
                    print(f"    labels sample (first 20): {labels[0][:20].tolist()}")
                    print(f"    -100 count (masked): {(labels[0] == -100).sum().item()}")
                    print(f"    Valid labels count: {(labels[0] != -100).sum().item()}")
                    if pad_token_id is not None:
                        print(f"    Pad token ID: {pad_token_id}")
                
                # Forward pass with AMP
                if use_amp:
                    with autocast():
                        # processor 출력을 그대로 사용
                        # Florence-2는 input_ids에 이미지 placeholder 토큰이 포함되어 있어야 함
                        outputs = model(
                            input_ids=inputs['input_ids'],
                            pixel_values=inputs['pixel_values'],
                            attention_mask=inputs.get('attention_mask'),
                            labels=labels  # decoder labels (pad는 -100으로 마스킹됨)
                        )
                        loss = outputs.loss
                    
                    # Gradient accumulation을 위한 loss 스케일링
                    loss = loss / accumulate_grad_batches
                    
                    # Backward pass with AMP
                    scaler.scale(loss).backward()
                    
                    # Gradient accumulation: 지정된 배치마다만 업데이트
                    if (batch_idx + 1) % accumulate_grad_batches == 0:
                        # Gradient clipping
                        if gradient_clip_val > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
                        
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        scheduler.step()
                else:
                    # Normal forward pass
                    # processor 출력을 그대로 사용
                    # Florence-2는 input_ids에 이미지 placeholder 토큰이 포함되어 있어야 함
                    outputs = model(
                        input_ids=inputs['input_ids'],
                        pixel_values=inputs['pixel_values'],
                        attention_mask=inputs.get('attention_mask'),
                        labels=labels  # decoder labels (pad는 -100으로 마스킹됨)
                    )
                    loss = outputs.loss
                    
                    # Gradient accumulation을 위한 loss 스케일링
                    loss = loss / accumulate_grad_batches
                    
                    # Normal backward pass
                    loss.backward()
                    
                    # Gradient accumulation: 지정된 배치마다만 업데이트
                    if (batch_idx + 1) % accumulate_grad_batches == 0:
                        # Gradient clipping
                        if gradient_clip_val > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
                        
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                
                # 실제 loss (accumulation 전) 기록
                train_loss += loss.item() * accumulate_grad_batches
                train_pbar.set_postfix({'loss': f'{loss.item() * accumulate_grad_batches:.4f}'})
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
                
                for inputs, targets in val_pbar:
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # ConditionalGeneration 방식: Training과 동일
                    # Target tokenize (decoder labels)
                    tokenized_targets = processor.tokenizer(
                        text=targets,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=1024,
                        add_special_tokens=False
                    )
                    labels = tokenized_targets.input_ids.to(device)
                    
                    # CRITICAL: Pad 토큰을 -100으로 변경 (loss 계산에서 제외)
                    pad_token_id = processor.tokenizer.pad_token_id
                    if pad_token_id is not None:
                        labels[labels == pad_token_id] = -100
                    
                    # Validation with AMP
                    if use_amp:
                        with autocast():
                            # processor 출력을 그대로 사용
                            # Florence-2는 input_ids에 이미지 placeholder 토큰이 포함되어 있어야 함
                            outputs = model(
                                input_ids=inputs['input_ids'],
                                pixel_values=inputs['pixel_values'],
                                attention_mask=inputs.get('attention_mask'),
                                labels=labels  # decoder labels (pad는 -100으로 마스킹됨)
                            )
                    else:
                        # processor 출력을 그대로 사용
                        # Florence-2는 input_ids에 이미지 placeholder 토큰이 포함되어 있어야 함
                        outputs = model(
                            input_ids=inputs['input_ids'],
                            pixel_values=inputs['pixel_values'],
                            attention_mask=inputs.get('attention_mask'),
                            labels=labels  # decoder labels (pad는 -100으로 마스킹됨)
                        )
                    
                    val_loss += outputs.loss.item()
                    val_pbar.set_postfix({'val_loss': f'{outputs.loss.item():.4f}'})
            
            avg_val_loss = val_loss / len(val_loader)
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log to console
            log_msg = f"\nEpoch {epoch+1}/{num_epochs}:\n"
            log_msg += f"  Train Loss: {avg_train_loss:.4f}\n"
            log_msg += f"  Val Loss:   {avg_val_loss:.4f}\n"
            log_msg += f"  Learning Rate: {current_lr:.2e}"
            print(log_msg)
            
            # 통합 로깅 (TensorBoard, CSV)
            manager.log_losses(
                epoch=epoch + 1,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                learning_rate=current_lr,
                patience_counter=patience_counter,
                flush=True
            )
            
            # Save best model and check early stopping
            is_best = avg_val_loss < best_val_loss
            if is_best:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # LoRA 모델 저장
                if use_lora:
                    best_model_dir = output_dir / "best"
                    model.save_pretrained(str(best_model_dir))
                    readme_path = best_model_dir / "README.md"
                    if readme_path.exists():
                        readme_path.unlink()
                    print(f"   Saved best LoRA weights to {best_model_dir}")
                else:
                    # 체크포인트 관리자 사용
                    manager.save_checkpoint(
                        epoch=epoch,
                        model_state_dict=model.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(),
                        metrics={'val_loss': avg_val_loss, 'train_loss': avg_train_loss},
                        is_best=True,
                        prefix=""
                    )
            else:
                patience_counter += 1
                print(f"    Val loss did not improve ({patience_counter}/{patience})")
                
                # Early stopping check
                if patience_counter >= patience:
                    print(f"\n Early stopping triggered after {epoch+1} epochs")
                    print(f"   No improvement for {patience} consecutive epochs")
                    break
            
            # Save last model (always)
            if use_lora:
                last_model_dir = output_dir / "last"
                model.save_pretrained(str(last_model_dir))
                readme_path = last_model_dir / "README.md"
                if readme_path.exists():
                    readme_path.unlink()
            else:
                manager.save_checkpoint(
                    epoch=epoch,
                    model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    metrics={'val_loss': avg_val_loss, 'train_loss': avg_train_loss},
                    is_best=False,
                    prefix=""
                )
            
            print()
        
        # 로거 종료
        manager.close()
        
        # Final summary
        final_msg = "\n" + "="*70 + "\n"
        final_msg += "TRAINING COMPLETED!\n"
        final_msg += "="*70 + "\n"
        final_msg += f"Best validation loss: {best_val_loss:.4f}\n"
        final_msg += f"Total epochs trained: {epoch+1}/{num_epochs}\n"
        final_msg += f"Checkpoints saved to: {output_dir}\n"
        if manager.tensorboard_enabled:
            tb_dir = output_dir.parent / "tensorboard"
            final_msg += f"TensorBoard logs: {tb_dir}\n"
            final_msg += f"  View with: tensorboard --logdir {tb_dir}\n"
        if manager.csv_enabled and manager.csv_file:
            final_msg += f"Metrics CSV: {manager.csv_file}\n"
        final_msg += "="*70 + "\n"
        
        print(final_msg)
        
        return model

    def __repr__(self):
        return f"Florence2Finetuned(device={self.device}, classes={list(self.class_map.values())})"

