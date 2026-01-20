"""Grounding DINO 모델 래퍼 - Hugging Face Transformers 사용"""
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

try:
    from transformers import (
        GroundingDinoForObjectDetection,
        AutoProcessor,
        GroundingDinoImageProcessor,
        GroundingDinoProcessor
    )
    GROUNDING_DINO_AVAILABLE = True
except ImportError:
    GROUNDING_DINO_AVAILABLE = False
    print("Warning: transformers library not found. Install with: pip install transformers")


class GroundingDINODataset(Dataset):
    """Grounding DINO용 COCO 데이터셋 - Transformers 형식"""
    
    def __init__(
        self,
        ann_file: str,
        images_dir: str,
        processor: Optional[Any] = None,
        class_names: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)
        
        self.images_dir = Path(images_dir)
        self.config = config or {}
        self.class_names = class_names or []
        self.processor = processor
        
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
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        file_name = self.image_id_to_file[img_id]
        img_path = self.images_dir / file_name
        
        # 이미지 로드 및 회전 처리
        from ..utils.vis_utils import load_and_orient_image
        image = load_and_orient_image(img_path, self.coco_data, img_id)
        img_width, img_height = image.size
        
        anns = self.annotations_by_image.get(img_id, [])
        
        # 텍스트 프롬프트 생성
        if self.class_names:
            text_prompt = ". ".join([f"{name} tomato" for name in self.class_names]) + "."
        else:
            categories = list(set([self.category_id_to_name[ann['category_id']] for ann in anns]))
            text_prompt = ". ".join([f"{cat} tomato" for cat in categories]) + "."
        
        # 타겟 준비 (COCO 형식)
        boxes = []
        labels = []
        for ann in anns:
            # COCO 형식: [x, y, w, h] (절대 좌표)
            boxes.append(ann['bbox'])
            category_name = self.category_id_to_name[ann['category_id']]
            if self.class_names and category_name in self.class_names:
                label = self.class_names.index(category_name)
            else:
                label = 0
            labels.append(label)
        
        # Processor로 전처리 (이미지와 텍스트)
        if self.processor:
            inputs = self.processor(
                images=image,
                text=text_prompt,
                return_tensors="pt"
            )
            # 배치 차원 제거
            pixel_values = inputs['pixel_values'].squeeze(0)
            input_ids = inputs['input_ids'].squeeze(0)
            attention_mask = inputs['attention_mask'].squeeze(0)
        else:
            # Processor 없이 기본 처리
            pixel_values = None
            input_ids = None
            attention_mask = None
        
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'text_prompt': text_prompt,
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'image_id': img_id,
            'image_size': torch.tensor([img_width, img_height], dtype=torch.float32)
        }
    
    @staticmethod
    def collate_fn(batch):
        """가변 길이 데이터 처리"""
        # Processor가 있으면 이미 전처리됨
        if batch[0]['pixel_values'] is not None:
            pixel_values = torch.stack([item['pixel_values'] for item in batch])
            input_ids = torch.stack([item['input_ids'] for item in batch])
            attention_mask = torch.stack([item['attention_mask'] for item in batch])
        else:
            pixel_values = None
            input_ids = None
            attention_mask = None
        
        text_prompts = [item['text_prompt'] for item in batch]
        images = [item['image'] for item in batch]
        boxes = [item['boxes'] for item in batch]
        labels = [item['labels'] for item in batch]
        image_ids = [item['image_id'] for item in batch]
        image_sizes = torch.stack([item['image_size'] for item in batch])
        
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'text_prompt': text_prompts,
            'image': images,
            'boxes': boxes,
            'labels': labels,
            'image_id': image_ids,
            'image_size': image_sizes
        }


class GroundingDINOWrapper:
    """Grounding DINO 학습 래퍼 - Hugging Face Transformers 사용"""
    
    def __init__(self, config: Dict[str, Any]):
        if not GROUNDING_DINO_AVAILABLE:
            raise ImportError(
                "Transformers library not found. "
                "Please install: pip install transformers"
            )
        
        self.config = config
        self.model_config = config.get('model', {})
        self.data_config = config.get('data', {})
        self.training_config = config.get('training', {})
        
        # 모델 이름 (기본값: base 모델)
        self.model_name = self.model_config.get(
            'model_name', 
            'IDEA-Research/grounding-dino-base'
        )
        
        # Hugging Face Transformers로 모델 로드
        self._build_model()
        
        # Loss 함수 설정
        self._setup_criterion()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        self.model.train()
    
    def _build_model(self):
        """Hugging Face Transformers로 모델 빌드"""
        try:
            print(f"Loading Grounding DINO model from: {self.model_name}")
            
            # Processor 로드
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            # 모델 로드
            self.model = GroundingDinoForObjectDetection.from_pretrained(
                self.model_name
            )
            
            # 클래스 수 확인 및 저장
            num_classes = self.data_config.get('num_classes', 3)
            # num_labels = background (1) + num_classes
            new_num_labels = num_classes + 1
            
            # 모델 config 업데이트
            self.model.config.num_labels = new_num_labels
            
            # 모델의 모든 관련 설정 업데이트
            # Grounding DINO는 내부적으로 여러 곳에서 num_labels를 사용
            if hasattr(self.model.config, 'num_labels'):
                self.model.config.num_labels = new_num_labels
            
            # decoder의 num_labels도 업데이트 (있는 경우)
            if hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'num_labels'):
                self.model.decoder.num_labels = new_num_labels
            
            # 저장 (나중에 사용)
            self.num_classes = num_classes
            self.num_labels = new_num_labels
            
            print(f"Model loaded successfully")
            print(f"   Num classes: {num_classes}, Num labels: {new_num_labels}")
            print(f"   Model config num_labels: {self.model.config.num_labels}")
            
        except Exception as e:
            raise ImportError(
                f"Failed to load Grounding DINO model from Hugging Face. "
                f"Error: {e}\n"
                f"Make sure transformers is installed: pip install transformers"
            )
    
    def _setup_criterion(self):
        """Loss 함수 설정 - Transformers 모델은 자체 loss 사용"""
        # Transformers의 GroundingDinoForObjectDetection은 
        # forward에서 loss를 계산할 수 있음
        # 또는 표준 PyTorch loss 사용
        from torch.nn import CrossEntropyLoss, L1Loss, SmoothL1Loss
        
        # DETR 스타일 loss (선택사항)
        self.box_loss = SmoothL1Loss()
        self.class_loss = CrossEntropyLoss()
        
        # 또는 모델의 자체 loss 사용 (forward에서 계산)
        self.use_model_loss = True
    
    def _prepare_targets(self, boxes, labels, image_sizes):
        """타겟을 Transformers 형식으로 변환"""
        # Transformers는 정규화된 좌표 [x_center, y_center, width, height]를 받음
        # 또는 [x_min, y_min, x_max, y_max] 형식
        targets = []
        
        num_classes = self.data_config.get('num_classes', 3)
        
        for i, (box_list, label_list, img_size) in enumerate(zip(boxes, labels, image_sizes)):
            img_w, img_h = img_size[0].item(), img_size[1].item()
            
            # Transformers 형식으로 변환
            target_boxes = []
            target_labels = []
            
            for box, label in zip(box_list, label_list):
                x, y, w, h = box
                
                # COCO 형식 [x, y, w, h]를 Transformers 형식 [x_min, y_min, x_max, y_max]로 변환
                # 그리고 정규화
                x_min = max(0, min(x, img_w))
                y_min = max(0, min(y, img_h))
                x_max = max(0, min(x + w, img_w))
                y_max = max(0, min(y + h, img_h))
                
                # 정규화 (0~1 범위)
                x_min_norm = x_min / img_w
                y_min_norm = y_min / img_h
                x_max_norm = x_max / img_w
                y_max_norm = y_max / img_h
                
                # 유효성 검사
                if x_max_norm > x_min_norm and y_max_norm > y_min_norm:
                    target_boxes.append([x_min_norm, y_min_norm, x_max_norm, y_max_norm])
                    # 라벨: Grounding DINO는 zero-shot 모델이므로 클래스 수가 고정되지 않음
                    # loss 계산 시 label_maps를 사용하는데, label_maps는 텍스트 프롬프트에서 생성됨
                    # class_labels는 label_maps의 인덱스이므로, 0부터 시작해야 함
                    # 원래 label이 0, 1, 2이면 그대로 사용 (0부터 시작)
                    label_idx = int(label)
                    # 범위 체크: 0부터 num_classes-1까지 (num_classes=3이면 0,1,2)
                    if 0 <= label_idx < num_classes:
                        target_labels.append(label_idx)
                    else:
                        print(f"Warning: Invalid label {label_idx}, max allowed: {num_classes-1}. Skipping.")
            
            if len(target_boxes) > 0:
                targets.append({
                    'boxes': torch.tensor(target_boxes, dtype=torch.float32, device=self.device),
                    'class_labels': torch.tensor(target_labels, dtype=torch.long, device=self.device)
                })
            else:
                # 빈 타겟
                targets.append({
                    'boxes': torch.zeros((0, 4), dtype=torch.float32, device=self.device),
                    'class_labels': torch.zeros((0,), dtype=torch.long, device=self.device)
                })
        
        return targets
    
    def _forward_pass(self, pixel_values, input_ids, attention_mask, text_prompts, labels=None):
        """Forward pass - Transformers 방식"""
        # Transformers 모델의 forward
        # labels가 있으면 loss도 계산됨
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    def _compute_loss(self, outputs, targets):
        """Loss 계산 - Transformers 방식"""
        if self.use_model_loss and hasattr(outputs, 'loss') and outputs.loss is not None:
            # 모델이 자체 loss를 계산한 경우
            return outputs.loss
        
        # 수동으로 loss 계산 (필요한 경우)
        # outputs.logits와 outputs.pred_boxes를 사용
        # 실제 구현은 복잡하므로 여기서는 기본 구조만
        
        # 임시: 더미 loss (실제 구현 필요)
        return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def train(self, output_dir: Path, experiment_id: str):
        """학습 실행"""
        print("Starting Grounding DINO training (using Hugging Face Transformers)...")
        print(f"Model: {self.model_name}")
        print(f"Output directory: {output_dir}")
        
        # 데이터셋 import
        from ..data.transforms.gdinio_transform import create_gdino_dataset
        from ..registry import DATASET_REGISTRY
        
        # 데이터셋 메타 정보 가져오기
        dataset_name = self.data_config.get('dataset_name')
        dataset_meta = DATASET_REGISTRY.get(dataset_name)
        
        # 데이터셋 생성 (Processor 사용)
        train_dataset = GroundingDINODataset(
            ann_file=str(Path(self.data_config['data_root']) / "train" / "custom_train.json"),
            images_dir=str(Path(self.data_config['data_root']) / "train" / "images"),
            processor=self.processor,
            class_names=self.data_config.get('class_names', []),
            config=self.config
        )
        val_dataset = GroundingDINODataset(
            ann_file=str(Path(self.data_config['data_root']) / "val" / "custom_val.json"),
            images_dir=str(Path(self.data_config['data_root']) / "val" / "images"),
            processor=self.processor,
            class_names=self.data_config.get('class_names', []),
            config=self.config
        )
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        
        # DataLoader 생성
        batch_size = self.training_config.get('batch_size', 2)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=self.data_config.get('num_workers', 0),
            collate_fn=GroundingDINODataset.collate_fn
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=self.data_config.get('num_workers', 0),
            collate_fn=GroundingDINODataset.collate_fn
        )
        
        # 옵티마이저
        lr = self.training_config.get('learning_rate', 1e-5)
        weight_decay = self.training_config.get('weight_decay', 0.0001)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        num_epochs = self.training_config.get('epochs', 10)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Early stopping 설정
        early_stopping_config = self.config.get('trainer', {}).get('early_stopping', {})
        patience = early_stopping_config.get('patience', self.training_config.get('patience', 20))
        monitor = early_stopping_config.get('monitor', 'val_loss')
        min_delta = early_stopping_config.get('min_delta', 0.0)
        mode = early_stopping_config.get('mode', 'min')
        
        # Gradient accumulation 설정
        accumulate_grad_batches = self.training_config.get('accumulate_grad_batches', 1)
        
        # TensorBoard 설정
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = output_dir / "tensorboard"
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(tb_dir), flush_secs=10)  # 10초마다 자동 flush
        print(f"TensorBoard logs will be saved to: {tb_dir}")
        print(f"   View with: tensorboard --logdir {tb_dir}")
        
        # 초기 메트릭 기록 (학습 시작 전)
        writer.add_text('Config/Model', self.model_name, 0)
        writer.add_text('Config/Dataset', dataset_name, 0)
        writer.add_scalar('Config/TrainSamples', len(train_dataset), 0)
        writer.add_scalar('Config/ValSamples', len(val_dataset), 0)
        writer.add_scalar('Config/BatchSize', batch_size, 0)
        writer.add_scalar('Config/AccumulateGradBatches', accumulate_grad_batches, 0)
        writer.add_scalar('Config/LearningRate', lr, 0)
        writer.flush()
        
        # 학습 루프
        best_val_loss = float('inf')
        best_train_loss = float('inf')
        patience_counter = 0
        global_step = 0
        accumulation_step = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training
            self.model.train()
            train_loss = 0.0
            train_count = 0
            
            # Gradient accumulation을 위한 초기화 (epoch마다)
            optimizer.zero_grad()
            accumulation_step = 0
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
                pixel_values = batch['pixel_values'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                boxes = batch['boxes']
                labels = batch['labels']
                image_sizes = batch['image_size']
                
                # 타겟 준비
                targets = self._prepare_targets(boxes, labels, image_sizes)
                
                # Forward pass (labels 포함하여 loss 자동 계산)
                outputs = self._forward_pass(
                    pixel_values, 
                    input_ids, 
                    attention_mask, 
                    batch['text_prompt'],
                    labels=targets
                )
                
                # Loss 계산 (모델이 자동으로 계산)
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    loss = outputs.loss
                else:
                    # Fallback: 수동 loss 계산
                    loss = self._compute_loss(outputs, targets)
                
                # Gradient accumulation: loss를 accumulate_grad_batches로 나눔
                loss = loss / accumulate_grad_batches
                
                # Backward pass
                loss.backward()
                
                accumulation_step += 1
                
                # Gradient accumulation이 충분히 쌓였을 때만 optimizer.step()
                if accumulation_step % accumulate_grad_batches == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                # 원래 loss 값으로 기록 (나눈 값이 아닌)
                train_loss += loss.item() * accumulate_grad_batches
                train_count += 1
                global_step += 1
                
                # TensorBoard에 배치별 loss 기록 (첫 배치와 주기적으로)
                if global_step == 1 or global_step % 10 == 0:
                    writer.add_scalar('Train/BatchLoss', loss.item() * accumulate_grad_batches, global_step)
                    writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], global_step)
                    writer.flush()  
            
            # 남은 gradient가 있으면 업데이트
            if accumulation_step % accumulate_grad_batches != 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_count = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    pixel_values = batch['pixel_values'].to(self.device)
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    boxes = batch['boxes']
                    labels = batch['labels']
                    image_sizes = batch['image_size']
                    
                    targets = self._prepare_targets(boxes, labels, image_sizes)
                    
                    outputs = self._forward_pass(
                        pixel_values, 
                        input_ids, 
                        attention_mask, 
                        batch['text_prompt'],
                        labels=targets
                    )
                    
                    if hasattr(outputs, 'loss') and outputs.loss is not None:
                        loss = outputs.loss
                        
                        # Loss 분해 정보 출력 (첫 배치와 마지막 epoch)
                        if val_count == 0 and epoch == 0:
                            print(f"\nValidation Loss Breakdown (first batch):")
                            print(f"  Total loss: {loss.item():.4f}")
                            # Loss dict가 있으면 각 loss 출력
                            if hasattr(outputs, 'loss_dict') or isinstance(outputs.loss, dict):
                                loss_dict = getattr(outputs, 'loss_dict', outputs.loss if isinstance(outputs.loss, dict) else {})
                                if isinstance(loss_dict, dict):
                                    for key, value in loss_dict.items():
                                        if isinstance(value, torch.Tensor):
                                            print(f"  {key}: {value.item():.4f}")
                    else:
                        loss = self._compute_loss(outputs, targets)
                    
                    val_loss += loss.item()
                    val_count += 1
            
            scheduler.step()
            
            avg_train_loss = train_loss / max(train_count, 1)
            avg_val_loss = val_loss / max(val_count, 1)
            
            print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # TensorBoard에 epoch별 메트릭 기록
            writer.add_scalar('Train/EpochLoss', avg_train_loss, epoch + 1)
            writer.add_scalar('Val/EpochLoss', avg_val_loss, epoch + 1)
            writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], epoch + 1)
            writer.flush()  # 즉시 디스크에 기록
            
            # Early stopping 체크
            improved = False
            if monitor == 'val_loss':
                if mode == 'min':
                    if avg_val_loss < best_val_loss - min_delta:
                        improved = True
                        best_val_loss = avg_val_loss
                else:  # mode == 'max'
                    if avg_val_loss > best_val_loss + min_delta:
                        improved = True
                        best_val_loss = avg_val_loss
            elif monitor == 'train_loss':
                if mode == 'min':
                    if avg_train_loss < best_train_loss - min_delta:
                        improved = True
                        best_train_loss = avg_train_loss
                else:  # mode == 'max'
                    if avg_train_loss > best_train_loss + min_delta:
                        improved = True
                        best_train_loss = avg_train_loss
            
            # 체크포인트 저장
            if improved:
                patience_counter = 0
                checkpoint_path = checkpoint_dir / f"gdino_best_{experiment_id}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'train_loss': avg_train_loss,
                }, checkpoint_path)
                print(f"Best checkpoint saved: {checkpoint_path}")
                # TensorBoard에 best loss 기록
                writer.add_scalar('Val/BestLoss', best_val_loss, epoch + 1)
                writer.flush()  # 즉시 디스크에 기록
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")
            
            # Early stopping 체크
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best {monitor}: {best_val_loss if monitor == 'val_loss' else best_train_loss:.4f}")
                break
        
        # TensorBoard writer 닫기
        writer.close()
        
        # 최종 모델 저장
        final_path = checkpoint_dir / f"gdino_final_{experiment_id}.pth"
        torch.save(self.model.state_dict(), final_path)
        print(f"\nFinal model saved: {final_path}")
        print(f"\nTensorBoard logs saved to: {tb_dir}")
        print(f"   View with: tensorboard --logdir {tb_dir}")
        
        return {
            'best_checkpoint': str(checkpoint_dir / f"gdino_best_{experiment_id}.pth"),
            'final_checkpoint': str(final_path),
            'best_val_loss': best_val_loss
        }
