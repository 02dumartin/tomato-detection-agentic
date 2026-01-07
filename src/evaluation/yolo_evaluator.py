"""YOLO 모델 전용 평가"""
import json
import shutil
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import Dict
from ultralytics import YOLO

from ..metrics.model_complexity import calculate_model_complexity
from .result_saver import save_summary_metrics


class YOLOEvaluator:
    """YOLO 모델 평가 클래스"""
    
    def __init__(self, config: Dict, checkpoint_path: str, split: str = 'val', 
                 output_dir: Path = None, score_threshold: float = 0.5, 
                 iou_threshold: float = 0.5):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.split = split
        self.output_dir = output_dir
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
    
    def evaluate(self) -> Dict:
        """
        YOLO 모델 평가 실행
        
        Returns:
            results_dict: 평가 결과 딕셔너리
        """
        # 모델 로드
        model = YOLO(str(self.checkpoint_path))
        
        # data.yaml 경로 설정
        data_root = Path(self.config['data']['data_root'])
        data_yaml_path = data_root / 'data.yaml'
        
        if not data_yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}")
        
        # YOLO val() 실행
        print(f"\nRunning YOLO evaluation on {self.split} set...")
        print(f"Data YAML: {data_yaml_path}")
        
        # YOLO의 split 매핑
        split_map = {
            'val': 'val',
            'test': 'test',
            'train': 'train'
        }
        yolo_split = split_map.get(self.split, 'val')
        
        # YOLO val() 결과를 results 폴더 하위에 저장하도록 설정
        yolo_val_output_dir = self.output_dir / "yolo_val_results"
        
        results = model.val(
            data=str(data_yaml_path),
            split=yolo_split,
            batch=self.config.get('evaluation', {}).get('batch_size', 8),
            conf=self.score_threshold,
            iou=self.iou_threshold,
            plots=True,
            save_json=True,
            project=str(self.output_dir),
            name="yolo_val_results",
            exist_ok=True,
        )
        
        # YOLO가 실제로 저장한 경로 확인 및 이동
        actual_save_dir = None
        
        # 1. results 객체에서 save_dir 확인
        if hasattr(results, 'save_dir') and results.save_dir:
            actual_save_dir = Path(results.save_dir)
        elif hasattr(results, 'save_dir'):
            try:
                actual_save_dir = Path(results.save_dir) if results.save_dir else None
            except:
                pass
        
        # 2. project/name 경로 확인
        if not actual_save_dir or not actual_save_dir.exists():
            potential_dir = self.output_dir / "yolo_val_results"
            if potential_dir.exists():
                actual_save_dir = potential_dir
        
        # 3. runs/val/ 폴더 확인 (YOLO 기본 경로)
        if not actual_save_dir or not actual_save_dir.exists():
            runs_val_dir = Path("runs/val")
            if runs_val_dir.exists():
                # 가장 최근 디렉토리 찾기
                val_dirs = sorted(runs_val_dir.glob("exp*"), key=lambda x: x.stat().st_mtime, reverse=True)
                if val_dirs:
                    actual_save_dir = val_dirs[0]
        
        # 4. 실제 저장 경로가 우리가 원하는 경로와 다르면 파일 이동
        if actual_save_dir and actual_save_dir.exists() and actual_save_dir != yolo_val_output_dir:
            print(f"\n YOLO results found at: {actual_save_dir}")
            print(f" Moving results to: {yolo_val_output_dir}")
            
            # 목적지 디렉토리 생성
            yolo_val_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 모든 파일과 디렉토리 복사
            copied_count = 0
            for item in actual_save_dir.iterdir():
                dest = yolo_val_output_dir / item.name
                try:
                    if item.is_dir():
                        if dest.exists():
                            shutil.rmtree(dest)
                        shutil.copytree(item, dest)
                    else:
                        shutil.copy2(item, dest)
                    copied_count += 1
                except Exception as e:
                    print(f"  Could not copy {item.name}: {e}")
            
            print(f" Copied {copied_count} items to {yolo_val_output_dir}")
            
            # 원본 디렉토리 삭제 (runs/val/인 경우만)
            if str(actual_save_dir).startswith("runs/"):
                try:
                    shutil.rmtree(actual_save_dir)
                    print(f"  Removed original directory: {actual_save_dir}")
                except Exception as e:
                    print(f"  Could not remove original directory: {e}")
        
        # 최종 경로 확인
        if yolo_val_output_dir.exists() and any(yolo_val_output_dir.iterdir()):
            file_count = len(list(yolo_val_output_dir.iterdir()))
            print(f" YOLO validation results ({file_count} items) saved to: {yolo_val_output_dir}")
        else:
            print(f"  Warning: YOLO validation results directory is empty or not found: {yolo_val_output_dir}")
        
        # 클래스별 AP 추출
        class_names = self.config['data']['class_names']
        class_ap_50 = {}
        class_ap_50_95 = {}
        class_ap_75 = {}
        
        # YOLO results에서 클래스별 AP 추출
        # mAP@0.5:0.95 추출
        if hasattr(results.box, 'maps') and results.box.maps is not None:
            maps = results.box.maps
            if isinstance(maps, (list, np.ndarray, torch.Tensor)):
                maps = maps.tolist() if hasattr(maps, 'tolist') else list(maps)
                for i, ap in enumerate(maps):
                    if i < len(class_names):
                        class_ap_50_95[class_names[i]] = float(ap)
        
        # mAP@0.5 추출 시도
        if hasattr(results.box, 'maps50'):
            maps50 = results.box.maps50
            if maps50 is not None:
                if isinstance(maps50, (list, np.ndarray, torch.Tensor)):
                    maps50 = maps50.tolist() if hasattr(maps50, 'tolist') else list(maps50)
                    for i, ap in enumerate(maps50):
                        if i < len(class_names):
                            class_ap_50[class_names[i]] = float(ap)
        
        if not class_ap_50 and hasattr(results, 'maps50'):
            maps50 = results.maps50
            if maps50 is not None:
                if isinstance(maps50, (list, np.ndarray, torch.Tensor)):
                    maps50 = maps50.tolist() if hasattr(maps50, 'tolist') else list(maps50)
                    for i, ap in enumerate(maps50):
                        if i < len(class_names):
                            class_ap_50[class_names[i]] = float(ap)
        
        # mAP@0.75 추출 시도
        if hasattr(results.box, 'maps75'):
            maps75 = results.box.maps75
            if maps75 is not None:
                if isinstance(maps75, (list, np.ndarray, torch.Tensor)):
                    maps75 = maps75.tolist() if hasattr(maps75, 'tolist') else list(maps75)
                    for i, ap in enumerate(maps75):
                        if i < len(class_names):
                            class_ap_75[class_names[i]] = float(ap)
        
        if not class_ap_75 and hasattr(results, 'maps75'):
            maps75 = results.maps75
            if maps75 is not None:
                if isinstance(maps75, (list, np.ndarray, torch.Tensor)):
                    maps75 = maps75.tolist() if hasattr(maps75, 'tolist') else list(maps75)
                    for i, ap in enumerate(maps75):
                        if i < len(class_names):
                            class_ap_75[class_names[i]] = float(ap)
        
        # 방법 3: results.box 객체의 속성 확인 및 시도
        if not class_ap_50 or not class_ap_75:
            # 가능한 속성명들 시도
            for attr_name in ['maps50', 'ap50', 'ap_class_50', 'class_map50']:
                if hasattr(results.box, attr_name):
                    attr_value = getattr(results.box, attr_name)
                    if attr_value is not None and not class_ap_50:
                        if isinstance(attr_value, (list, np.ndarray, torch.Tensor)):
                            attr_value = attr_value.tolist() if hasattr(attr_value, 'tolist') else list(attr_value)
                            for i, ap in enumerate(attr_value):
                                if i < len(class_names):
                                    class_ap_50[class_names[i]] = float(ap)
                            break
            
            for attr_name in ['maps75', 'ap75', 'ap_class_75', 'class_map75']:
                if hasattr(results.box, attr_name):
                    attr_value = getattr(results.box, attr_name)
                    if attr_value is not None and not class_ap_75:
                        if isinstance(attr_value, (list, np.ndarray, torch.Tensor)):
                            attr_value = attr_value.tolist() if hasattr(attr_value, 'tolist') else list(attr_value)
                            for i, ap in enumerate(attr_value):
                                if i < len(class_names):
                                    class_ap_75[class_names[i]] = float(ap)
                            break
        
        # 결과 출력
        print("\n" + "="*70)
        print("YOLO EVALUATION RESULTS")
        print("="*70)
        print(f"mAP@0.50:      {results.box.map50:.4f}")
        print(f"mAP@0.50:0.95: {results.box.map:.4f}")
        if hasattr(results.box, 'map75'):
            print(f"mAP@0.75:      {results.box.map75:.4f}")
        print(f"Precision:     {results.box.mp:.4f}")
        print(f"Recall:        {results.box.mr:.4f}")
        print("="*70)
        
        # 클래스별 AP 출력
        if class_ap_50 or class_ap_50_95:
            print("\n" + "="*70)
            print("PER-CLASS AP")
            print("="*70)
            for class_name in class_names:
                ap50 = class_ap_50.get(class_name, "N/A")
                ap50_95 = class_ap_50_95.get(class_name, "N/A")
                ap75 = class_ap_75.get(class_name, "N/A")
                
                ap50_str = f"{ap50:.4f}" if isinstance(ap50, (int, float)) else str(ap50)
                ap50_95_str = f"{ap50_95:.4f}" if isinstance(ap50_95, (int, float)) else str(ap50_95)
                ap75_str = f"{ap75:.4f}" if isinstance(ap75, (int, float)) else str(ap75)
                
                print(f"  {class_name:15s}: AP@0.5={ap50_str:>6s}, "
                      f"AP@0.5:0.95={ap50_95_str:>6s}, "
                      f"AP@0.75={ap75_str:>6s}")
            print("="*70)
        
        # 모델 복잡도 계산
        complexity_metrics = {}
        try:
            complexity_metrics = calculate_model_complexity(model.model, self.config)
        except Exception as e:
            print(f"  Could not calculate model complexity: {e}")
        
        # Backbone 정보 추출
        backbone_info = self.config['model'].get('arch_name', 'unknown')
        if 'model_size' in self.config['model']:
            backbone_info = f"{backbone_info}-{self.config['model']['model_size']}"
        
        # 결과 저장
        results_dict = {
            'detection_metrics': {
                'map_50': float(results.box.map50),
                'map': float(results.box.map),
                'map_75': float(results.box.map75) if hasattr(results.box, 'map75') else None,
            },
            'detailed_statistics': {
                'total_statistics': {
                    'overall_precision': float(results.box.mp),
                    'overall_recall': float(results.box.mr),
                    'overall_f1': float(2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr) if (results.box.mp + results.box.mr) > 0 else 0.0),
                },
                'per_class_ap': {
                    'ap_50': class_ap_50,
                    'ap_50_95': class_ap_50_95,
                    'ap_75': class_ap_75,
                }
            },
            'model_complexity': complexity_metrics,
            'model_info': {
                'backbone': backbone_info,
                'arch_name': self.config['model'].get('arch_name', 'unknown'),
                'model_size': self.config['model'].get('model_size', 'unknown'),
            }
        }
        
        # 결과 저장 (공통 함수 사용)
        save_summary_metrics(results_dict, self.output_dir, self.config)
        
        # 복잡도 메트릭 출력
        if complexity_metrics:
            print("\n" + "="*70)
            print("MODEL COMPLEXITY")
            print("="*70)
            if 'params_m' in complexity_metrics:
                print(f"  Total Parameters: {complexity_metrics['params_m']:.2f}M")
            if 'model_size_mb' in complexity_metrics:
                print(f"  Model Size: {complexity_metrics['model_size_mb']:.2f} MB")
            if complexity_metrics.get('gflops') is not None:
                print(f"  GFLOPs: {complexity_metrics['gflops']:.2f}")
            print("="*70)
        
        # YOLO val() 결과 디렉토리 확인
        yolo_val_dir = self.output_dir / "yolo_val_results"
        if not yolo_val_dir.exists():
            # results 객체에서 save_dir 확인
            if hasattr(results, 'save_dir'):
                yolo_val_dir = Path(results.save_dir)
            elif hasattr(results, 'save_dir') and results.save_dir:
                yolo_val_dir = Path(results.save_dir)
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETED!")
        print("="*70)
        print(f"Results directory: {self.output_dir}")
        print(f"  - evaluation_results.json: 전체 결과")
        print(f"  - summary_metrics.csv: 성능 요약표")
        if yolo_val_dir.exists():
            print(f"  - yolo_val_results/: YOLO validation 결과 (plots, confusion matrix 등)")
        print("="*70 + "\n")
        
        return results_dict

