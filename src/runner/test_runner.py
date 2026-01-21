"""Test/Inference Runner"""
import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from tqdm import tqdm

import torch
import pandas as pd
from torch.utils.data import DataLoader

from ..registry import MODEL_REGISTRY, DATASET_REGISTRY
from ..utils.vis_utils import (
    save_visualization_images, 
    save_yolo_visualization_images, 
    get_model_color, 
    get_class_colors,
    load_and_orient_image
)


class TestRunner:
    """테스트/인퍼런스 실행 클래스 - 예측 결과 생성 및 시각화"""
    
    def __init__(self, config, checkpoint_path, split='test', output_dir=None):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.split = split
        
        # 시각화 옵션 (output_dir 설정 전에 먼저 읽기)
        self.box_color_mode = config.get('test', {}).get('box_color_mode', 'class')  # 'class' or 'model'
        
        # 출력 디렉토리 설정
        if output_dir is None:
            # checkpoint 경로에서 exp 정보 추출
            exp_info = self._extract_exp_info_from_checkpoint(checkpoint_path, config)
            model_name = exp_info['model_name']
            exp_name = exp_info['exp_name']
            
            # exp/{model}/{exp_name}/visualization/{mode} 구조
            # Florence-2의 경우 mode 폴더 포함
            if model_name.lower() in ['florence2', 'florence-2']:
                florence2_mode = config.get('florence2', {}).get('mode', 'zeroshot')
                mode_folder = 'zeroshot' if florence2_mode == 'zeroshot' else 'finetuned'
                exp_dir = Path("exp") / model_name / mode_folder / exp_name
            else:
                exp_dir = Path("exp") / model_name / exp_name
            base_output_dir = exp_dir / "visualization"
            output_dir = base_output_dir / self.box_color_mode
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 테스트 설정
        self.confidence_threshold = config.get('evaluation', {}).get('score_threshold', 
                                                                     config.get('model', {}).get('confidence_threshold', 0.5))
        self.iou_threshold = config.get('evaluation', {}).get('iou_threshold', 0.5)
        
        # 시각화 옵션
        self.show_gt = config.get('test', {}).get('show_gt', False)  # 기본값: False
        # self.box_color_mode은 이미 위에서 설정됨
        
        print(f"Visualization output directory: {self.output_dir}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print(f"IoU threshold: {self.iou_threshold}")
        print(f"Show GT: {self.show_gt}")
        print(f"Box color mode: {self.box_color_mode}")
    
    def _extract_exp_info_from_checkpoint(self, checkpoint_path, config):
        """Checkpoint 경로에서 실험 정보 추출"""
        from datetime import datetime
        
        checkpoint_path = Path(checkpoint_path)
        
        # checkpoint 경로 예시:
        # exp/yolov11/TomatOD_YOLO_3_20260106_105855/checkpoints/yolo_weights/best.pt
        # 또는
        # exp/yolov11/TomatOD_YOLO_3_yolov11_20260106_1/checkpoints/yolo_weights/best.pt
        
        parts = checkpoint_path.parts
        
        # exp 디렉토리 찾기
        try:
            exp_idx = parts.index('exp') if 'exp' in parts else -1
            if exp_idx >= 0 and len(parts) > exp_idx + 2:
                model_name = parts[exp_idx + 1]  # yolov11 또는 florence2
                # Florence-2의 경우 mode 폴더가 있을 수 있음: exp/florence2/zeroshot/exp_name
                if model_name.lower() in ['florence2', 'florence-2'] and len(parts) > exp_idx + 3:
                    # mode 폴더가 있는 경우 (zeroshot/finetuned)
                    if parts[exp_idx + 2] in ['zeroshot', 'finetuned']:
                        exp_name = parts[exp_idx + 3]
                    else:
                        exp_name = parts[exp_idx + 2]
                else:
                    exp_name = parts[exp_idx + 2]    # TomatOD_YOLO_3_20260106_105855 또는 TomatOD_YOLO_3_yolov11_20260106_1
                
                # exp_name이 새 형식인지 확인
                # 새 형식: dataset_model_date_number (예: TomatOD_YOLO_3_yolov11_20260106_1)
                # 기존 형식: dataset_timestamp (예: TomatOD_YOLO_3_20260106_105855)
                exp_parts = exp_name.split('_')
                
                # 새 형식 체크: 마지막에서 3번째가 6자리 숫자(날짜)이고, 마지막이 숫자(번호)인 경우
                is_new_format = (len(exp_parts) >= 5 and 
                               exp_parts[-3].isdigit() and len(exp_parts[-3]) == 6 and
                               exp_parts[-1].isdigit() and
                               exp_parts[-2] == model_name.lower())
                
                if not is_new_format:
                    # 기존 형식이면 그대로 사용 (하위 호환성)
                    # 또는 새 형식으로 변환하려면 아래 주석 해제
                    pass
                    # dataset_name = config['data']['dataset_name']
                    # date_str = datetime.now().strftime("%Y%m%d")
                    # exp_base_name = f"{dataset_name}_{model_name}_{date_str}"
                    # exp_dir = Path("exp") / model_name
                    # exp_name = self._get_experiment_name(exp_dir, exp_base_name)
                
                return {
                    'model_name': model_name,
                    'exp_name': exp_name
                }
        except:
            pass
        
        # 추출 실패 시 config에서 정보 가져오기
        model_name = config['model']['arch_name']
        dataset_name = config['data']['dataset_name']
        date_str = datetime.now().strftime("%y%m%d")
        exp_base_name = f"{dataset_name}_{model_name}_{date_str}"
        
        # Florence-2 등 checkpoint 없는 모델은 exp 디렉토리 확인
        # checkpoint가 'none'이면 exp 디렉토리에서 기존 실험 확인
        if checkpoint_path == Path('none') or str(checkpoint_path).lower() == 'none':
            # Florence-2의 경우 mode 폴더 포함
            if model_name.lower() in ['florence2', 'florence-2']:
                florence2_mode = config.get('florence2', {}).get('mode', 'zeroshot')
                mode_folder = 'zeroshot' if florence2_mode == 'zeroshot' else 'finetuned'
                exp_dir = Path("exp") / model_name / mode_folder
            else:
                exp_dir = Path("exp") / model_name
            exp_name = self._get_experiment_name(exp_dir, exp_base_name)
        else:
            # exp 디렉토리에서 확인 (학습 기반 모델)
            if model_name.lower() in ['florence2', 'florence-2']:
                florence2_mode = config.get('florence2', {}).get('mode', 'zeroshot')
                mode_folder = 'zeroshot' if florence2_mode == 'zeroshot' else 'finetuned'
                exp_dir = Path("exp") / model_name / mode_folder
            else:
                exp_dir = Path("exp") / model_name
            exp_name = self._get_experiment_name(exp_dir, exp_base_name)
        
        return {
            'model_name': model_name,
            'exp_name': exp_name
        }
    
    def _get_experiment_name(self, base_dir: Path, exp_base_name: str) -> str:
        """실험 이름 생성: 같은 날짜에 같은 조합이 있으면 번호 증가"""
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # 기존 실험 디렉토리 확인
        existing_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith(exp_base_name)]
        
        if not existing_dirs:
            # 첫 번째 실험
            return f"{exp_base_name}_1"
        else:
            # 번호 추출 및 최대값 찾기
            numbers = []
            for dir_name in existing_dirs:
                parts = dir_name.name.split('_')
                if len(parts) > 0:
                    try:
                        num = int(parts[-1])
                        numbers.append(num)
                    except ValueError:
                        pass
            
            if numbers:
                next_number = max(numbers) + 1
            else:
                next_number = 1
            
            return f"{exp_base_name}_{next_number}"
    
    def _load_model_and_data(self):
        """모델과 데이터 로드"""
        model_name = self.config['model']['arch_name']
        dataset_name = self.config['data']['dataset_name']
        
        # Dataset 메타정보
        dataset_meta = DATASET_REGISTRY.get(dataset_name)
        if dataset_meta is None:
            raise ValueError(f"Dataset '{dataset_name}' not found. Available: {list(DATASET_REGISTRY.keys())}")
        
        # 모델별 처리
        if model_name.lower() in ["yolov11", "yolov12", "yolo"]:
            # YOLO 모델
            model_name_normalized = model_name.lower()
            if model_name_normalized not in MODEL_REGISTRY:
                model_name_normalized = model_name_normalized.capitalize()
            
            ModelClass = MODEL_REGISTRY.get(model_name_normalized) or MODEL_REGISTRY.get('yolov11')
            if ModelClass is None:
                raise ValueError(f"Model '{model_name}' not found in MODEL_REGISTRY")
            
            # 체크포인트에서 모델 로드
            model = ModelClass.load_from_checkpoint(
                self.checkpoint_path,
                model_size=self.config['model'].get('model_size', 'm'),
                num_classes=self.config['model']['num_labels'],
                lr=self.config['model'].get('learning_rate', self.config['model'].get('lr0', 0.001)),
            )
            dataset = None
            collate_fn = None
            
        elif model_name == "DETR" or model_name == "detr":
            from transformers import DetrImageProcessor
            from ..data.transforms.detr_transform import create_detr_dataset, DetrCocoDataset
            
            data_cfg = self.config.get('data', {})
            processor_kwargs = {}
            img_size = data_cfg.get('image_size')
            max_size = data_cfg.get('max_size')
            # 노트북에서 사용한 DETR resize와 동일하게 shortest_edge/longest_edge를 함께 설정
            if img_size and max_size:
                processor_kwargs["size"] = {"shortest_edge": img_size, "longest_edge": max_size}
            elif img_size:
                processor_kwargs["size"] = {"shortest_edge": img_size}
            elif max_size:
                processor_kwargs["size"] = {"longest_edge": max_size}

            imageprocessor = DetrImageProcessor.from_pretrained(
                self.config['model']['pretrained_path'],
                **processor_kwargs
            )
            
            # 테스트할 데이터셋 로드
            dataset = create_detr_dataset(dataset_meta, self.split, imageprocessor, self.config)
            collate_fn = DetrCocoDataset.create_collate_fn(imageprocessor)
            
            # 체크포인트에서 모델 로드
            ModelClass = MODEL_REGISTRY[model_name]
            model = ModelClass.load_from_checkpoint(
                self.checkpoint_path,
                num_labels=self.config['model']['num_labels'],
                pretrained_path=self.config['model']['pretrained_path'],
                lr=self.config['model']['learning_rate'],
                lr_backbone=self.config['model']['lr_backbone'],
                weight_decay=self.config['model']['weight_decay'],
            )
            model.eval()
        else:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
        
        return model, dataset, collate_fn
    
    def run(self):
        """테스트/인퍼런스 실행"""
        print("\n" + "="*70)
        print("TEST/INFERENCE STARTED")
        print("="*70)
        print(f"Model: {self.config['model']['arch_name']}")
        print(f"Dataset: {self.config['data']['dataset_name']}")
        print(f"Split: {self.split}")
        print(f"Checkpoint: {Path(self.checkpoint_path).name}")
        print("="*70)
        
        model_name = self.config['model']['arch_name'].lower()
        
        if model_name in ["yolov11", "yolov12", "yolo"]:
            return self._run_yolo_test()
        elif model_name in ["groundingdino", "gdino", "grounding_dino"]:
            return self._run_grounding_dino_test()
        elif model_name in ["florence2", "florence-2", "florence"]:
            return self._run_florence2_test()
        
        print("\nLoading model and dataset...")
        model, dataset, collate_fn = self._load_model_and_data()
        print(f"Dataset samples: {len(dataset)}")
        
        # DataLoader 생성
        batch_size = self.config.get('evaluation', {}).get('batch_size', 
                                                           self.config['data'].get('batch_size', 8))
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config['data'].get('num_workers', 0),
            collate_fn=collate_fn,
        )
        
        # 예측 실행 및 시각화
        print("\nRunning inference...")
        predictions = []
        images = []
        
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Inference"):
                if isinstance(batch, dict):
                    pixel_values = batch["pixel_values"].to(next(model.parameters()).device)
                    outputs = model(pixel_values)
                    predictions.extend(outputs)
                    if "pixel_values" in batch:
                        images.extend(batch["pixel_values"].cpu())
                else:
                    # 다른 형식 처리
                    pass
        
        # 시각화 이미지 저장
        print("\nSaving visualization images...")
        save_visualization_images(
            model=model,
            dataset=dataset,
            output_dir=self.output_dir,
            config=self.config,
            show_gt=self.show_gt,
            box_color_mode=self.box_color_mode,
            confidence_threshold=self.confidence_threshold,
            split=self.split
        )
        
        print("\n" + "="*70)
        print("TEST/INFERENCE COMPLETED!")
        print("="*70)
        print(f"Results directory: {self.output_dir}")
        print("="*70 + "\n")
        
        return {'predictions': predictions, 'output_dir': str(self.output_dir)}
    
    def _run_yolo_test(self):
        """YOLO 모델 전용 테스트/인퍼런스"""
        from ultralytics import YOLO
        
        # 모델 로드
        model = YOLO(str(self.checkpoint_path))
        
        # data.yaml 경로 설정
        data_root = Path(self.config['data']['data_root'])
        data_yaml_path = data_root / 'data.yaml'
        
        if not data_yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}")
        
        # YOLO의 split 매핑
        split_map = {
            'test': 'test',
            'val': 'val',
            'train': 'train'
        }
        yolo_split = split_map.get(self.split, 'test')
        
        # YOLO predict() 실행 (inference)
        print(f"\nRunning YOLO inference on {self.split} set...")
        print(f"Data YAML: {data_yaml_path}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        
        # YOLO predict 실행 (save=False로 설정하여 yolo_predictions 폴더 생성 방지)
        results = model.predict(
            source=str(data_root / yolo_split / "images"),
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            save=False,  # YOLO 기본 시각화 비활성화
            save_txt=False,  # YOLO 형식 텍스트 파일 저장 안 함
            save_conf=False,  # confidence 점수 포함 안 함
            save_json=False,  # JSON 형식 저장 안 함
            project=str(self.output_dir),
            name="yolo_predictions",
            exist_ok=True,
        )
        
        # 사용자 정의 색상 스키마로 시각화 
        print("\nVisualizing with custom color scheme...")
        # 원본 이미지 디렉토리 경로 전달
        original_images_dir = data_root / yolo_split / "images"
        custom_viz_dir = save_yolo_visualization_images(
            yolo_results=results,
            output_dir=self.output_dir,
            config=self.config,
            show_gt=self.show_gt,
            box_color_mode=self.box_color_mode,
            confidence_threshold=self.confidence_threshold,
            split=self.split,
            yolo_output_dir=None,  # yolo_predictions 폴더 사용 안 함
            original_images_dir=str(original_images_dir)
        )
        
        # 결과 요약
        print("\n" + "="*70)
        print("YOLO TEST/INFERENCE RESULTS")
        print("="*70)
        print(f"Total images processed: {len(results)}")
        print(f"Visualization directory: {custom_viz_dir}")
        print("="*70)
        
        # 결과 구조 정리
        results_summary = {
            'total_images': len(results),
            'output_dir': str(self.output_dir),
            'visualization_dir': str(custom_viz_dir),
        }
        
        print("\n" + "="*70)
        print("TEST/INFERENCE COMPLETED!")
        print("="*70)
        print(f"Results directory: {self.output_dir}")
        print(f"  - Visualizations: {custom_viz_dir}")
        print("="*70 + "\n")
        
        return results_summary
    
    def _run_grounding_dino_test(self):
        """Grounding DINO 모델 전용 테스트/시각화"""
        from ..models.gdino_model import GroundingDINOWrapper, GroundingDINODataset
        from ..registry import DATASET_REGISTRY
        from PIL import Image, ImageDraw, ImageFont
        from tqdm import tqdm
        import torch
        import json
        from pathlib import Path
        
        print("\nLoading Grounding DINO model...")
        
        # Wrapper 생성 및 체크포인트 로드
        wrapper = GroundingDINOWrapper(self.config)
        # GPU로 직접 로드
        checkpoint = torch.load(self.checkpoint_path, map_location=wrapper.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            wrapper.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            wrapper.model.load_state_dict(checkpoint)
        wrapper.model.eval()
        wrapper.model.to(wrapper.device)
        
        # 데이터셋 로드
        dataset_meta = DATASET_REGISTRY.get(self.config['data']['dataset_name'])
        
        # get_data_paths 메서드 사용
        paths = dataset_meta.get_data_paths(self.split, self.config)
        ann_file = Path(paths['ann_file'])
        images_dir = Path(paths['img_folder'])
        
        dataset = GroundingDINODataset(
            ann_file=str(ann_file),
            images_dir=str(images_dir),
            processor=wrapper.processor,
            class_names=self.config['data'].get('class_names', []),
            config=self.config
        )
        collate_fn = GroundingDINODataset.collate_fn
        
        # DataLoader 생성
        batch_size = self.config.get('evaluation', {}).get('batch_size', 
                                                           self.config['data'].get('batch_size', 2))
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config['data'].get('num_workers', 0),
            collate_fn=collate_fn
        )
        
        print(f"Dataset samples: {len(dataset)}")
        print(f"Evaluation batches: {len(dataloader)}")
        
        # 색상 매핑
        colors = {
            'fully-ripe': (255, 0, 0),      # 빨간색
            'semi-ripe': (255, 165, 0),     # 주황색
            'unripe': (0, 255, 0)           # 초록색
        }
        class_names = self.config['data'].get('class_names', ['fully-ripe', 'semi-ripe', 'unripe'])
        class_name_to_id = {name: i for i, name in enumerate(class_names)}
        
        # 시각화 디렉토리 생성
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # 텍스트 프롬프트 생성
        text_prompt = ". ".join([f"{name} tomato" for name in class_names]) + "."
        
        print("\nRunning inference and visualization...")
        all_predictions = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Inference")):
                pixel_values = batch['pixel_values'].to(wrapper.device)
                input_ids = batch['input_ids'].to(wrapper.device)
                attention_mask = batch['attention_mask'].to(wrapper.device)
                image_ids = batch.get('image_id', batch.get('image_ids', []))
                images = batch.get('image', [])
                
                # Forward pass
                outputs = wrapper.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # 후처리
                target_sizes = torch.tensor([[2000, 2000]] * pixel_values.shape[0], device=wrapper.device)
                results = wrapper.processor.post_process_grounded_object_detection(
                    outputs,
                    input_ids=input_ids,
                    threshold=self.confidence_threshold,
                    text_threshold=self.confidence_threshold,
                    target_sizes=target_sizes
                )
                
                # 시각화
                for i, result in enumerate(results):
                    img_id = image_ids[i] if i < len(image_ids) else batch_idx * dataloader.batch_size + i
                    
                    # 원본 이미지 가져오기 (batch에서)
                    if i < len(images):
                        image = images[i].copy() if isinstance(images[i], Image.Image) else Image.fromarray(images[i])
                    else:
                        # 폴백: 파일 경로로 로드
                        file_name = dataset.image_id_to_file.get(img_id, f"image_{img_id}.jpg")
                        img_path = dataset.images_dir / file_name
                        if not img_path.exists():
                            continue
                        image = load_and_orient_image(img_path)
                    
                    file_name = dataset.image_id_to_file.get(img_id, f"image_{img_id}.jpg")
                    draw = ImageDraw.Draw(image)
                    
                    # 예측 결과 그리기
                    boxes = result['boxes'].cpu().numpy()
                    scores = result['scores'].cpu().numpy()
                    labels = result['labels']
                    text_labels = result.get('text_labels', [])
                    
                    # 디버깅: 예측 결과 확인
                    num_detections = len(boxes)
                    if num_detections == 0:
                        if batch_idx == 0 and i == 0:
                            print(f"\nWarning: No detections for first image {img_id} (file: {file_name})")
                            print(f"  Threshold: {self.confidence_threshold}")
                            print(f"  Try lowering the threshold if you expect detections")
                        continue
                    
                    # 안전하게 처리 (boxes와 scores는 항상 같은 길이여야 함)
                    num_items = len(boxes)  # boxes와 scores는 항상 같은 길이
                    
                    # 길이 확인
                    if len(scores) != num_items:
                        print(f"  Warning: scores length ({len(scores)}) != boxes length ({num_items}) for image {img_id}")
                        num_items = min(num_items, len(scores))
                    if labels and len(labels) != num_items:
                        if batch_idx == 0 and i == 0:
                            print(f"  Warning: labels length ({len(labels)}) != boxes length ({num_items}) for image {img_id}")
                    if text_labels and len(text_labels) != num_items:
                        if batch_idx == 0 and i == 0:
                            print(f"  Warning: text_labels length ({len(text_labels)}) != boxes length ({num_items}) for image {img_id}")
                    
                    for idx in range(num_items):
                        box = boxes[idx]
                        score = float(scores[idx])
                        label = labels[idx] if labels and idx < len(labels) else ""
                        text_label = text_labels[idx] if text_labels and idx < len(text_labels) else ""
                        
                        x_min, y_min, x_max, y_max = box
                        width = x_max - x_min
                        height = y_max - y_min
                        
                        # 라벨 매핑 (text_label 사용 또는 label 인덱스 사용)
                        # text_label이 "fully - ripe tomato" 같은 형식일 수 있으므로 처리
                        category_name = None
                        if text_label:
                            # text_label에서 클래스 이름 추출
                            text_label_clean = text_label.lower().replace(" ", "").replace("-", "")
                            for class_name in class_names:
                                class_name_clean = class_name.lower().replace("-", "")
                                if class_name_clean in text_label_clean:
                                    category_name = class_name
                                    break
                        
                        if not category_name:
                            # label이 인덱스인 경우
                            if isinstance(label, int) and 0 <= label < len(class_names):
                                category_name = class_names[label]
                            elif isinstance(label, str):
                                # label이 문자열인 경우
                                label_clean = label.lower().replace(" ", "").replace("-", "")
                                for class_name in class_names:
                                    class_name_clean = class_name.lower().replace("-", "")
                                    if class_name_clean in label_clean:
                                        category_name = class_name
                                        break
                        
                        if not category_name:
                            category_name = f"class_{label}" if isinstance(label, int) else str(label)
                        
                        color = colors.get(category_name, (255, 255, 255))
                        
                        # Bbox 그리기
                        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=5)
                        
                        # 라벨 텍스트
                        label_text = f"{category_name} {score:.2f}"
                        bbox_text = draw.textbbox((x_min, y_min-20), label_text)
                        draw.rectangle(bbox_text, fill=(0, 0, 0, 200))
                        draw.text((x_min, y_min-20), label_text, fill=color)
                        
                        # 예측 저장
                        # label이 리스트인 경우 첫 번째 요소 사용 또는 인덱스로 변환
                        if isinstance(label, list) and len(label) > 0:
                            label_id = label[0] if isinstance(label[0], int) else 0
                        elif isinstance(label, int):
                            label_id = label
                        else:
                            label_id = 0
                        
                        all_predictions.append({
                            'image_id': int(img_id),
                            'file_name': file_name,
                            'category_id': int(label_id),
                            'category_name': category_name,
                            'bbox': [float(x_min), float(y_min), float(width), float(height)],
                            'score': float(score)
                        })
                    
                    # 시각화 이미지 저장
                    output_path = viz_dir / f"pred_{file_name}"
                    image.save(output_path)
        
        # 예측 결과 저장
        predictions_file = self.output_dir / "predictions.json"
        with open(predictions_file, 'w') as f:
            json.dump(all_predictions, f, indent=2)
        
        print("\n" + "="*70)
        print("GROUNDING DINO TEST/INFERENCE COMPLETED!")
        print("="*70)
        print(f"Results directory: {self.output_dir}")
        print(f"  - Visualizations: {viz_dir}")
        print(f"  - Predictions JSON: {predictions_file}")
        print(f"  - Total predictions: {len(all_predictions)}")
        print("="*70 + "\n")
        
        return {
            'predictions': all_predictions,
            'output_dir': str(self.output_dir),
            'visualizations_dir': str(viz_dir),
            'predictions_file': str(predictions_file)
        }
    
    def _organize_yolo_test_results(self, yolo_output_dir: Path, results):
        """YOLO 테스트 결과 정리"""
        if not yolo_output_dir.exists():
            print(f"  YOLO output directory not found: {yolo_output_dir}")
            return
        
        print(f"\nOrganizing YOLO test results...")
        print(f"  Output directory: {yolo_output_dir}")
        
        # 디렉토리 구조 확인
        subdirs = [d for d in yolo_output_dir.iterdir() if d.is_dir()]
        files = [f for f in yolo_output_dir.iterdir() if f.is_file()]
        
        print(f"  Subdirectories: {[d.name for d in subdirs]}")
        print(f"  Files: {[f.name for f in files[:10]]}...") 
        
        # 결과 구조 문서화
        structure = {
            'output_dir': str(yolo_output_dir),
            'structure': {
                'images': 'Predicted images with bounding boxes (saved by YOLO)',
                'labels': 'YOLO format label files (.txt) - one per image',
                'json': 'Predictions in JSON format (if save_json=True)',
            }
        }
        
        structure_path = self.output_dir / "results_structure.json"
        with open(structure_path, 'w') as f:
            json.dump(structure, f, indent=2)
        
        print(f"   Results structure saved to: {structure_path}")
    
    def _classify_florence_label(self, label: str) -> tuple:
        """
        Florence-2의 레이블을 토마토 클래스로 분류
        
        Args:
            label: Florence-2가 예측한 레이블
        
        Returns:
            (class_name, class_id) tuple
        """
        # 현재 설정의 클래스 이름 가져오기
        class_names = self.config['data'].get('class_names', ['tomato'])
        num_classes = len(class_names)
        
        # 1-class인 경우 항상 첫 번째 클래스 반환
        if num_classes == 1:
            return class_names[0], 0
        
        # 3-class인 경우 기존 로직 사용
        label_lower = label.lower()
        
        # 색상 키워드로 판단
        if any(kw in label_lower for kw in ['red', 'crimson', 'scarlet', 'ripe']):
            class_name = 'fully_ripe' if 'fully_ripe' in class_names else class_names[0]
            class_id = class_names.index(class_name) if class_name in class_names else 0
            return class_name, class_id
        elif any(kw in label_lower for kw in ['orange', 'yellow', 'amber']):
            class_name = 'semi_ripe' if 'semi_ripe' in class_names else class_names[0]
            class_id = class_names.index(class_name) if class_name in class_names else 0
            return class_name, class_id
        elif any(kw in label_lower for kw in ['green', 'unripe']):
            class_name = 'unripe' if 'unripe' in class_names else class_names[0]
            class_id = class_names.index(class_name) if class_name in class_names else 0
            return class_name, class_id
        else:
            # 기본값: 첫 번째 클래스
            return class_names[0], 0
    
    def _run_florence2_test(self):
        """Florence-2 모델 전용 테스트/시각화"""
        from ..registry import DATASET_REGISTRY
        from PIL import Image, ImageDraw, ImageFont
        from tqdm import tqdm
        import torch
        import json
        from pathlib import Path
        
        print("\nLoading Florence-2 model...")
        
        # Florence-2 모드 확인
        florence2_mode = self.config.get('florence2', {}).get('mode', 'zeroshot')
        
        # Output 디렉토리를 모드별로 구분
        original_output_dir = self.output_dir
        mode_suffix = florence2_mode  # 'zeroshot' or 'finetuned'
        
        # results/florence2/... 경로를 results/florence2/{mode}/...로 변경
        if 'florence2' in str(original_output_dir):
            parts = list(original_output_dir.parts)
            # 'florence2' 다음에 mode 삽입
            try:
                florence_idx = parts.index('florence2')
                # 이미 mode가 있는지 확인
                if florence_idx + 1 < len(parts) and parts[florence_idx + 1] not in ['zeroshot', 'finetuned']:
                    # mode가 없으면 추가
                    parts.insert(florence_idx + 1, mode_suffix)
                    self.output_dir = Path(*parts)
                elif florence_idx + 1 >= len(parts):
                    # florence2가 마지막 디렉토리면 mode 추가
                    self.output_dir = original_output_dir / mode_suffix
                # 이미 mode가 있으면 그대로 사용
            except ValueError:
                # florence2가 경로에 없으면 그대로 사용
                pass
        
        print(f"Output directory: {self.output_dir}")
        print(f"Florence-2 mode: {florence2_mode}")
        
        if florence2_mode == 'zeroshot':
            from ..models.florence2_base import Florence2Base
            # Zero-shot 모드는 checkpoint 불필요 - <OD> task 사용
            # device 우선순위: hardware.device > training.device > device > 'cuda'
            device = (self.config.get('hardware', {}).get('device') or 
                     self.config.get('training', {}).get('device') or
                     self.config.get('device', 'cuda'))
            model = Florence2Base(
                device=device,
                config=self.config  
            )
            print("Zero-shot mode: Using <OD> task (general object detection)")
        elif florence2_mode == 'finetuned':
            from ..models.florence2_finetuned import Florence2Finetuned
            # Fine-tuned 모드는 체크포인트 필요
            checkpoint_path = self.checkpoint_path or self.config.get('florence2', {}).get('checkpoint_path')
            if not checkpoint_path or not Path(checkpoint_path).exists():
                raise FileNotFoundError(
                    f"Fine-tuned mode requires valid checkpoint.\n"
                    f"  Provided: {checkpoint_path}\n"
                    f"  Please specify --checkpoint or set florence2.checkpoint_path in config"
                )
            model = Florence2Finetuned(
                checkpoint_path=checkpoint_path,
                device=self.config.get('device', 'cuda'),
                config=self.config 
            )
        else:
            raise ValueError(f"Unknown Florence-2 mode: {florence2_mode}. Use 'zeroshot' or 'finetuned'")
        
        # 데이터셋 로드
        dataset_meta = DATASET_REGISTRY.get(self.config['data']['dataset_name'])
        if dataset_meta is None:
            raise ValueError(f"Dataset '{self.config['data']['dataset_name']}' not found")
        
        # get_data_paths 메서드 사용
        paths = dataset_meta.get_data_paths(self.split, self.config)
        ann_file = Path(paths['ann_file'])
        images_dir = Path(paths['img_folder'])
        
        # COCO annotations 로드
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
        
        # 이미지 폴더에서 파일 목록을 가져와서 정렬 (test 이미지 폴더 순서대로)
        image_files = sorted([f for f in images_dir.iterdir() 
                             if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']])
        
        # COCO annotation의 images를 파일명으로 매핑
        images_dict = {img['file_name']: img for img in coco_data['images']}
        
        # 정렬된 파일 순서대로 images_list 생성
        images_list = []
        for img_file in image_files:
            file_name = img_file.name
            if file_name in images_dict:
                images_list.append(images_dict[file_name])
            else:
                # COCO annotation에 없는 파일도 처리 (file_name만으로 생성)
                print(f"  Warning: {file_name} not found in COCO annotations, skipping...")
        
        images = {img['id']: img for img in images_list}  # 빠른 조회를 위한 딕셔너리
        
        annotations = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations:
                annotations[img_id] = []
            annotations[img_id].append(ann)
        
        print(f"Dataset samples: {len(images_list)} (sorted by filename)")
        
        # 클래스 이름 매핑
        class_names = self.config['data'].get('class_names', ['fully_ripe', 'semi_ripe', 'unripe'])
        num_classes = len(class_names)
        class_id_to_name = {i: name for i, name in enumerate(class_names)}
        
        # 색상 매핑 (다른 모델과 동일한 방식)
        from ..utils.vis_utils import get_model_color, get_class_colors
        
        if self.box_color_mode == 'class':
            # 클래스별 색상 (클래스 이름을 전달하여 정확한 색상 매핑)
            class_colors = get_class_colors(num_classes, class_names)
            # 클래스 이름으로도 접근 가능하도록
            colors = {}
            for i, name in enumerate(class_names):
                colors[i] = class_colors[i]
                colors[name] = class_colors[i]
                colors[name.replace('_', '-')] = class_colors[i]  # 하이픈 버전도
        else:  # 'model'
            # 모델별 단일 색상
            model_color = get_model_color('florence2')
            colors = {i: model_color for i in range(num_classes)}
            for name in class_names:
                colors[name] = model_color
                colors[name.replace('_', '-')] = model_color
        
        # 시각화 디렉토리 (visualizations 서브폴더 없이 바로 output_dir에 저장)
        viz_dir = self.output_dir
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Florence-2 설정
        nms_threshold = self.config.get('florence2', {}).get('nms_threshold', 0.5)
        conf_threshold = self.config.get('florence2', {}).get('conf_threshold', 0.0)
        verbose = self.config.get('logging', {}).get('verbose', False)
        
        print(f"Florence-2 mode: {florence2_mode}")
        print(f"NMS threshold: {nms_threshold}")
        print(f"Confidence threshold: {conf_threshold}")
        
        print("\nRunning inference and visualization...")
        all_predictions = []
        
        # test 이미지 폴더의 정렬된 순서대로 처리
        # 각 이미지에 대해 추론
        for idx, img_info in enumerate(tqdm(images_list, desc="Inference")):
            img_id = img_info['id']
            image_path = images_dir / img_info['file_name']
            
            if not image_path.exists():
                print(f"\n  Image not found: {image_path}")
                continue
            
            try:
                # 이미지를 먼저 회전 처리 (추론 전에 회전)
                from ..utils.vis_utils import load_and_orient_image
                image = load_and_orient_image(image_path, coco_data, img_id)
                
                # 추론 (회전된 이미지로 수행)
                if florence2_mode == 'zeroshot':
                    # Zero-shot: 클래스별 프롬프트를 사용한 <CAPTION_TO_PHRASE_GROUNDING> 태스크
                    detections = []
                    
                    # 클래스별 프롬프트 정의
                    class_prompts = self._get_class_prompts(class_names, num_classes)
                    
                    if idx == 0:
                        print(f"\n   [DEBUG] Using class-specific prompts:")
                        for class_name, prompt in class_prompts.items():
                            print(f"      {class_name}: {prompt}")
                    
                    # 각 클래스별로 프롬프트를 사용해서 탐지
                    for class_name, class_id in zip(class_names, range(num_classes)):
                        # 클래스별 프롬프트 생성
                        prompt_text = class_prompts[class_name]
                        task_prompt = f"<CAPTION_TO_PHRASE_GROUNDING> {prompt_text}"
                        
                        result = model.predict(image, task=task_prompt)
                        
                        # 결과 파싱
                        result_key = task_prompt  # <CAPTION_TO_PHRASE_GROUNDING> 태스크는 전체 task_prompt를 키로 사용
                        
                        if idx == 0 and class_id == 0:
                            print(f"\n   [DEBUG] Task prompt: {task_prompt}")
                            print(f"   [DEBUG] Result keys: {list(result.keys())}")
                            print(f"   [DEBUG] Result content: {result}")
                            print(f"   [DEBUG] Using result_key: {result_key}")
                        
                        if result_key in result:
                            det = result[result_key]
                            
                            # det가 dict인 경우와 문자열인 경우 모두 처리
                            if isinstance(det, dict):
                                bboxes = det.get('bboxes', [])
                                labels = det.get('labels', [])
                            elif isinstance(det, str):
                                # 문자열 형식인 경우 파싱 시도
                                # 형식: '텍스트<loc_x1><loc_y1><loc_x2><loc_y2>텍스트<loc_x1><loc_y1><loc_x2><loc_y2>...'
                                if idx == 0 and class_id == 0:
                                    print(f"   [DEBUG] Result is string, parsing: {det[:100]}...")
                                bboxes, labels = self._parse_caption_to_phrase_grounding(det, image.width, image.height)
                                if idx == 0 and class_id == 0:
                                    print(f"   [DEBUG] Parsed {len(bboxes)} bboxes from string")
                            else:
                                bboxes = []
                                labels = []
                            
                            if idx == 0 and class_id == 0:
                                print(f"   [DEBUG] Found {len(bboxes)} bboxes for {class_name}")
                            
                            # 해당 클래스의 탐지 결과 추가
                            for bbox in bboxes:
                                detections.append({
                                    'bbox': bbox,
                                    'class': class_name,
                                    'class_id': class_id,
                                    'score': 1.0,
                                    'label': prompt_text  # 원본 프롬프트를 레이블로 저장
                                })
                        else:
                            if idx == 0 and class_id == 0:
                                print(f"   [WARNING] Result key '{result_key}' not found!")
                                print(f"   [WARNING] Available keys: {list(result.keys())}")
                    
                    # 모든 클래스 탐지 후 NMS 적용 (중복 제거)
                    if len(detections) > 0:
                        detections = self._apply_nms(detections, nms_threshold)
                        if idx == 0:
                            print(f"   [DEBUG] After NMS: {len(detections)} detections")
                
                else:  # finetuned
                    # PIL Image를 임시 파일로 저장하거나 직접 사용
                    import tempfile
                    import os
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                        image.save(tmp_file.name, quality=95)
                        tmp_path = tmp_file.name
                    
                    try:
                        detections = model.predict_finetuned(
                            tmp_path,
                            conf_threshold=conf_threshold
                        )
                    finally:
                        # 임시 파일 삭제
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                
                # 시각화 - 이미지는 이미 회전되어 있음
                draw = ImageDraw.Draw(image)
                
                # 예측 결과 그리기
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    
                    # 좌표 검증 및 정렬: x1 <= x2, y1 <= y2 보장
                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)
                    
                    # 이미지 범위로 클램핑
                    img_w, img_h = image.size
                    x1 = max(0, min(x1, img_w))
                    y1 = max(0, min(y1, img_h))
                    x2 = max(x1, min(x2, img_w))  # x2 >= x1 보장
                    y2 = max(y1, min(y2, img_h))  # y2 >= y1 보장
                    
                    # 유효한 bbox인지 확인 (너무 작거나 역순이면 건너뛰기)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    width = x2 - x1
                    height = y2 - y1
                    
                    class_name = det['class']
                    score = det['score']
                    class_id = det['class_id']
                    
                    # 색상 가져오기 (class_name 우선, 없으면 class_id, 둘 다 없으면 model_color)
                    color = colors.get(class_name)
                    if color is None:
                        color = colors.get(class_id)
                    if color is None:
                                                # box_color_mode가 'model'이면 model_color 사용
                        if self.box_color_mode == 'model':
                            from ..utils.vis_utils import get_model_color
                            color = get_model_color('florence2')
                        else:
                            color = (255, 255, 255)  # 기본값: 흰색
                    
                    # Bbox 그리기 (label/score 텍스트 없이)
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
                    
                    # COCO 형식으로 저장
                    pred_dict = {
                        'image_id': int(img_id),
                        'file_name': img_info['file_name'],
                        'category_id': int(class_id),
                        'category_name': class_name,
                        'bbox': [float(x1), float(y1), float(width), float(height)],
                        'score': float(score)
                    }
                    # 원본 레이블도 저장 (디버깅용)
                    if 'label' in det:
                        pred_dict['original_label'] = det['label']
                    all_predictions.append(pred_dict)
                
                # 시각화 저장 (번호 순서대로 파일명 생성)
                viz_filename = f"inference_{idx:04d}_test.jpg"
                viz_path = viz_dir / viz_filename
                image.save(viz_path, quality=95)
            
            except Exception as e:
                print(f"\n Error processing {image_path}: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
                continue
        
        # 예측 결과 저장
        predictions_file = self.output_dir / "predictions.json"
        with open(predictions_file, 'w') as f:
            json.dump(all_predictions, f, indent=2)
        
        print(f"\n Predictions saved to: {predictions_file}")
        print(f" Visualizations saved to: {viz_dir}")
        print(f"   Total predictions: {len(all_predictions)}")
        print(f"   Total images processed: {len(images)}")
        
        # 클래스별 통계
        class_counts = {}
        label_counts = {}  # Florence-2 원본 레이블 통계
        for pred in all_predictions:
            class_name = pred['category_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            # 원본 레이블이 있는 경우 통계
            if 'label' in pred:
                original_label = pred.get('label', 'unknown')
                label_counts[original_label] = label_counts.get(original_label, 0) + 1
        
        print("\n Detection Statistics:")
        for class_name, count in sorted(class_counts.items()):
            print(f"   {class_name}: {count}")
        
        # Florence-2 원본 레이블 통계 (디버깅용)
        if label_counts:
            print("\n Original Florence-2 Labels (first 10):")
            for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"   {label}: {count}")
        
        # 결과 구조 문서화
        structure = {
            'output_dir': str(self.output_dir),
            'mode': florence2_mode,
            'predictions_file': str(predictions_file),
            'visualizations_dir': str(viz_dir),
            'total_predictions': len(all_predictions),
            'total_images': len(images),
            'class_distribution': class_counts,
            'config': {
                'nms_threshold': nms_threshold,
                'conf_threshold': conf_threshold,
                'classes': class_names
            }
        }
        
        structure_path = self.output_dir / "results_structure.json"
        with open(structure_path, 'w') as f:
            json.dump(structure, f, indent=2)
        
        print(f"\n Results structure saved to: {structure_path}")
        
        return {
            'predictions': all_predictions,
            'output_dir': str(self.output_dir),
            'num_predictions': len(all_predictions),
            'num_images': len(images)
        }
    
    def _get_class_prompts(self, class_names: List[str], num_classes: int) -> Dict[str, str]:
        """
        클래스별 프롬프트 생성
        
        Args:
            class_names: 클래스 이름 리스트
            num_classes: 클래스 개수
        
        Returns:
            dict: 클래스 이름 -> 프롬프트 매핑
        """
        # Config에서 클래스별 프롬프트 가져오기 (있는 경우)
        florence2_config = self.config.get('florence2', {})
        class_prompts_config = florence2_config.get('class_prompts', {})
        
        # 기본 프롬프트 정의
        default_prompts = {
            'fully_ripe': 'red ripe tomato, bright red tomato, crimson red tomato',
            'semi_ripe': 'orange tomato, yellow tomato, orange yellow tomato, semi-ripe tomato',
            'unripe': 'green tomato, unripe tomato, dark green tomato, immature green tomato',
            'tomato': 'tomato'  # 1-class의 경우
        }
        
        class_prompts = {}
        for class_name in class_names:
            # Config에 정의된 프롬프트가 있으면 사용
            if class_name in class_prompts_config:
                class_prompts[class_name] = class_prompts_config[class_name]
            # 기본 프롬프트 사용
            elif class_name in default_prompts:
                class_prompts[class_name] = default_prompts[class_name]
            # 1-class의 경우
            elif num_classes == 1:
                class_prompts[class_name] = 'tomato'
            # 기본값
            else:
                class_prompts[class_name] = f'{class_name} tomato'
        
        return class_prompts
    
    def _parse_caption_to_phrase_grounding(self, result_str: str, img_width: int, img_height: int) -> Tuple[List[List[float]], List[str]]:
        """
        <CAPTION_TO_PHRASE_GROUNDING> 태스크의 문자열 결과를 파싱
        
        형식: '텍스트<loc_x1><loc_y1><loc_x2><loc_y2>텍스트<loc_x1><loc_y1><loc_x2><loc_y2>...'
        
        Args:
            result_str: 파싱할 문자열
            img_width: 이미지 너비
            img_height: 이미지 높이
        
        Returns:
            (bboxes, labels): bbox 리스트와 레이블 리스트
        """
        import re
        
        bboxes = []
        labels = []
        
        # <loc_숫자> 패턴 찾기
        # 형식: 텍스트<loc_x1><loc_y1><loc_x2><loc_y2>
        pattern = r'([^<]+)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>'
        
        matches = re.finditer(pattern, result_str)
        
        for match in matches:
            label_text = match.group(1).strip()
            x1 = int(match.group(2))
            y1 = int(match.group(3))
            x2 = int(match.group(4))
            y2 = int(match.group(5))
            
            # Florence-2는 1000x1000 좌표계를 사용하므로 이미지 크기에 맞게 스케일링
            # 좌표 검증 및 정렬
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # 1000x1000 좌표계를 이미지 크기로 스케일링
            scale_x = img_width / 1000.0
            scale_y = img_height / 1000.0
            
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            
            # 이미지 범위로 클램핑
            x1 = max(0, min(x1, img_width))
            y1 = max(0, min(y1, img_height))
            x2 = max(x1, min(x2, img_width))
            y2 = max(y1, min(y2, img_height))
            
            # 유효한 bbox인지 확인
            if x2 > x1 and y2 > y1:
                bboxes.append([float(x1), float(y1), float(x2), float(y2)])
                labels.append(label_text)
        
        return bboxes, labels
    
    def _apply_nms(self, detections: List[Dict], nms_threshold: float) -> List[Dict]:
        """
        Non-Maximum Suppression (NMS) 적용
        
        Args:
            detections: 탐지 결과 리스트
            nms_threshold: IoU 임계값
        
        Returns:
            NMS 적용된 탐지 결과 리스트
        """
        if len(detections) == 0:
            return detections
        
        # bbox를 [x1, y1, x2, y2] 형식으로 변환
        boxes = []
        scores = []
        classes = []
        for det in detections:
            bbox = det['bbox']
            if len(bbox) == 4:
                # [x1, y1, x2, y2] 또는 [x, y, w, h] 형식 확인
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    # [x1, y1, x2, y2] 형식
                    boxes.append(bbox)
                else:
                    # [x, y, w, h] 형식 -> [x1, y1, x2, y2]로 변환
                    x, y, w, h = bbox
                    boxes.append([x, y, x + w, y + h])
                scores.append(det.get('score', 1.0))
                classes.append(det.get('class_id', 0))
        
        if len(boxes) == 0:
            return detections
        
        # numpy 배열로 변환
        import numpy as np
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        classes = np.array(classes, dtype=np.int32)
        
        # 클래스별로 NMS 적용
        keep_indices = []
        unique_classes = np.unique(classes)
        
        for cls in unique_classes:
            cls_mask = classes == cls
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]
            cls_indices = np.where(cls_mask)[0]
            
            if len(cls_boxes) == 0:
                continue
            
            # IoU 계산 및 NMS
            # 간단한 NMS 구현
            sorted_indices = np.argsort(cls_scores)[::-1]  # 점수 내림차순 정렬
            keep = []
            
            while len(sorted_indices) > 0:
                # 가장 높은 점수의 박스 선택
                current = sorted_indices[0]
                keep.append(current)
                
                if len(sorted_indices) == 1:
                    break
                
                # 나머지 박스들과 IoU 계산
                current_box = cls_boxes[current]
                other_indices = sorted_indices[1:]
                other_boxes = cls_boxes[other_indices]
                
                # IoU 계산
                ious = self._calculate_iou(current_box, other_boxes)
                
                # IoU가 임계값보다 낮은 박스만 유지
                sorted_indices = other_indices[ious < nms_threshold]
            
            # 원본 인덱스로 변환
            keep_indices.extend(cls_indices[keep])
        
        # NMS 적용된 탐지 결과 반환
        return [detections[i] for i in keep_indices]
    
    def _calculate_iou(self, box1: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        IoU 계산
        
        Args:
            box1: 단일 박스 [x1, y1, x2, y2]
            boxes: 여러 박스들 [[x1, y1, x2, y2], ...]
        
        Returns:
            IoU 값들
        """
        import numpy as np
        
        # 교집합 영역 계산
        x1_min = np.maximum(box1[0], boxes[:, 0])
        y1_min = np.maximum(box1[1], boxes[:, 1])
        x2_max = np.minimum(box1[2], boxes[:, 2])
        y2_max = np.minimum(box1[3], boxes[:, 3])
        
        intersection = np.maximum(0, x2_max - x1_min) * np.maximum(0, y2_max - y1_min)
        
        # 각 박스의 넓이 계산
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # 합집합 영역 계산
        union = area1 + area2 - intersection
        
        # IoU 계산 (0으로 나누기 방지)
        iou = intersection / np.maximum(union, 1e-6)
        
        return iou
    
    def _classify_florence_label(self, label: str) -> Tuple[str, int]:
        """
        Florence-2의 레이블을 클래스로 매핑
        
        Args:
            label: Florence-2가 탐지한 레이블 (예: 'grapefruit', 'tomato', 'orange', etc.)
        
        Returns:
            (class_name, class_id): 클래스 이름과 ID
        """
        label_lower = label.lower()
        
        # 클래스 이름 가져오기
        class_names = self.config['data'].get('class_names', ['fully_ripe', 'semi_ripe', 'unripe'])
        num_classes = len(class_names)
        
        # 레이블 기반 클래스 추정 (간단한 휴리스틱)
        # 실제로는 Florence-2가 탐지한 객체가 토마토인지 확인하고,
        # 색상이나 다른 특징을 기반으로 클래스를 추정해야 하지만,
        # <OD> 태스크는 클래스 정보를 제공하지 않으므로 기본적으로 첫 번째 클래스로 매핑
        
        # 레이블에 'tomato'가 포함되어 있으면 토마토로 간주
        # 그 외에는 기본적으로 첫 번째 클래스로 매핑
        if 'tomato' in label_lower:
            # 토마토인 경우, 레이블의 색상 정보로 클래스 추정 시도
            if any(word in label_lower for word in ['red', 'ripe', 'mature', 'crimson']):
                class_name = 'fully_ripe' if 'fully_ripe' in class_names else class_names[0]
            elif any(word in label_lower for word in ['green', 'unripe', 'immature']):
                class_name = 'unripe' if 'unripe' in class_names else class_names[-1]
            elif any(word in label_lower for word in ['orange', 'yellow', 'semi']):
                class_name = 'semi_ripe' if 'semi_ripe' in class_names else class_names[1] if len(class_names) > 1 else class_names[0]
            else:
                # 기본적으로 첫 번째 클래스 (fully_ripe)
                class_name = class_names[0]
        else:
            # 토마토가 아닌 경우도 첫 번째 클래스로 매핑 (모든 객체를 토마토로 간주)
            # 또는 무시할 수도 있지만, 여기서는 첫 번째 클래스로 매핑
            class_name = class_names[0]
        
        # class_id 찾기
        class_id = class_names.index(class_name) if class_name in class_names else 0
        
        return class_name, class_id
