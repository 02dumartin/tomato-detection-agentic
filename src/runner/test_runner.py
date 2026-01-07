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
from ..utils.visualization import save_visualization_images, save_yolo_visualization_images, get_model_color, get_class_colors


class TestRunner:
    """테스트/인퍼런스 실행 클래스 - 예측 결과 생성 및 시각화"""
    
    def __init__(self, config, checkpoint_path, split='test', output_dir=None):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.split = split
        
        # 출력 디렉토리 설정
        if output_dir is None:
            # checkpoint 경로에서 exp 정보 추출
            exp_info = self._extract_exp_info_from_checkpoint(checkpoint_path, config)
            model_name = exp_info['model_name']
            exp_name = exp_info['exp_name']
            
            # results/{model}/{exp_name}/test 구조
            output_dir = Path("results") / model_name / exp_name / "test"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 테스트 설정
        self.confidence_threshold = config.get('evaluation', {}).get('score_threshold', 
                                                                     config.get('model', {}).get('confidence_threshold', 0.5))
        self.iou_threshold = config.get('evaluation', {}).get('iou_threshold', 0.5)
        
        # 시각화 옵션
        self.show_gt = config.get('test', {}).get('show_gt', False)  # 기본값: False
        self.box_color_mode = config.get('test', {}).get('box_color_mode', 'class')  # 'class' or 'model'
        
        print(f"Test/Inference output directory: {self.output_dir}")
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
                model_name = parts[exp_idx + 1]  # yolov11
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
            
            imageprocessor = DetrImageProcessor.from_pretrained(
                self.config['model']['pretrained_path']
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
        
        # YOLO predict 실행
        results = model.predict(
            source=str(data_root / yolo_split / "images"),
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            save=True,
            save_txt=True,  # YOLO 형식 텍스트 파일 저장
            save_conf=True,  # confidence 점수 포함
            save_json=True,  # JSON 형식 저장
            project=str(self.output_dir),
            name="yolo_predictions",
            exist_ok=True,
        )
        
        # YOLO는 results[0].save_dir에 실제 저장 경로를 저장
        if results and len(results) > 0 and hasattr(results[0], 'save_dir'):
            yolo_output_dir = Path(results[0].save_dir)
        else:
            yolo_output_dir = self.output_dir / "yolo_predictions"
        
        # 결과 정리
        self._organize_yolo_test_results(yolo_output_dir, results)
        
        # 사용자 정의 색상 스키마로 재시각화 
        print("\nRe-visualizing with custom color scheme...")
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
            yolo_output_dir=yolo_output_dir,
            original_images_dir=str(original_images_dir)
        )
        
        # 결과 요약
        print("\n" + "="*70)
        print("YOLO TEST/INFERENCE RESULTS")
        print("="*70)
        print(f"Total images processed: {len(results)}")
        print(f"YOLO predictions: {yolo_output_dir}")
        print(f"Custom visualization: {custom_viz_dir}")
        print("="*70)
        
        # 결과 구조 정리
        results_summary = {
            'total_images': len(results),
            'output_dir': str(yolo_output_dir),
            'predictions_dir': str(yolo_output_dir),
            'labels_dir': str(yolo_output_dir / "labels") if (yolo_output_dir / "labels").exists() else None,
            'images_dir': str(yolo_output_dir) if yolo_output_dir.exists() else None,
        }
        
        # JSON 파일이 있으면 경로 추가
        json_files = list(yolo_output_dir.glob("*.json"))
        if json_files:
            results_summary['json_file'] = str(json_files[0])
        
        print("\n" + "="*70)
        print("TEST/INFERENCE COMPLETED!")
        print("="*70)
        print(f"Results directory: {self.output_dir}")
        print(f"  - YOLO predictions: {yolo_output_dir}")
        if results_summary.get('labels_dir'):
            print(f"  - Labels (YOLO format): {results_summary['labels_dir']}")
        if results_summary.get('json_file'):
            print(f"  - Predictions JSON: {results_summary['json_file']}")
        print("="*70 + "\n")
        
        return results_summary
    
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