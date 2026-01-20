"""YOLO 모델 전용 평가"""
import shutil
import numpy as np
import torch
from pathlib import Path
from typing import Dict
from ultralytics import YOLO

from .metrics import calculate_model_complexity
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
    
    def _extract_yolo_class_ap(self, results, attr_name: str, class_names: list) -> Dict:
        """YOLO results에서 클래스별 AP 추출 헬퍼 함수"""
        class_ap = {}
        
        # results.box에서 추출 시도
        if hasattr(results, 'box') and hasattr(results.box, attr_name):
            ap_data = getattr(results.box, attr_name)
            if ap_data is not None:
                if isinstance(ap_data, (list, np.ndarray, torch.Tensor)):
                    ap_list = ap_data.tolist() if hasattr(ap_data, 'tolist') else list(ap_data)
                    for i, ap in enumerate(ap_list):
                        if i < len(class_names):
                            class_ap[class_names[i]] = float(ap)
                    return class_ap
        
        # results에서 직접 추출 시도
        if hasattr(results, attr_name):
            ap_data = getattr(results, attr_name)
            if ap_data is not None:
                if isinstance(ap_data, (list, np.ndarray, torch.Tensor)):
                    ap_list = ap_data.tolist() if hasattr(ap_data, 'tolist') else list(ap_data)
                    for i, ap in enumerate(ap_list):
                        if i < len(class_names):
                            class_ap[class_names[i]] = float(ap)
                    return class_ap
        
        # 대체 속성명 시도 (maps50의 경우)
        if attr_name == 'maps50':
            for alt_name in ['ap50', 'ap_class_50', 'class_map50']:
                if hasattr(results, 'box') and hasattr(results.box, alt_name):
                    ap_data = getattr(results.box, alt_name)
                    if ap_data is not None:
                        if isinstance(ap_data, (list, np.ndarray, torch.Tensor)):
                            ap_list = ap_data.tolist() if hasattr(ap_data, 'tolist') else list(ap_data)
                            for i, ap in enumerate(ap_list):
                                if i < len(class_names):
                                    class_ap[class_names[i]] = float(ap)
                            return class_ap
        
        # maps75의 경우
        if attr_name == 'maps75':
            for alt_name in ['ap75', 'ap_class_75', 'class_map75']:
                if hasattr(results, 'box') and hasattr(results.box, alt_name):
                    ap_data = getattr(results.box, alt_name)
                    if ap_data is not None:
                        if isinstance(ap_data, (list, np.ndarray, torch.Tensor)):
                            ap_list = ap_data.tolist() if hasattr(ap_data, 'tolist') else list(ap_data)
                            for i, ap in enumerate(ap_list):
                                if i < len(class_names):
                                    class_ap[class_names[i]] = float(ap)
                            return class_ap
        
        return class_ap
    
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
        class_ap_50 = self._extract_yolo_class_ap(results, 'maps50', class_names)
        class_ap_50_95 = self._extract_yolo_class_ap(results, 'maps', class_names)
        class_ap_75 = self._extract_yolo_class_ap(results, 'maps75', class_names)
        
        # CA-mAP 계산 (predictions.json이 있는 경우) - 먼저 초기화
        ca_metrics = {}
        yolo_val_dir = self.output_dir / "yolo_val_results"
        
        # predictions.json 찾기
        predictions_json = None
        for json_file in yolo_val_dir.rglob("predictions.json"):
            predictions_json = json_file
            break
        
        if predictions_json and predictions_json.exists():
            print(f"\n  Found predictions.json: {predictions_json}")
            try:
                from pycocotools.coco import COCO
                from pycocotools.cocoeval import COCOeval
                import json
                import copy
                import tempfile
                import os
                
                # GT 파일 경로
                # YOLO 형식 데이터셋의 경우 원본 COCO 형식 데이터셋에서 GT 파일 찾기
                data_root_str = str(self.config['data']['data_root'])
                data_root = Path(data_root_str)
                
                # data_root가 _YOLO로 끝나면 원본 데이터셋 경로로 변경
                if data_root_str.endswith('_YOLO'):
                    original_data_root = data_root_str.replace('_YOLO', '')
                    data_root = Path(original_data_root)
                    print(f"  Using original COCO dataset for GT: {data_root}")
                
                gt_file = data_root / self.split / f"custom_{self.split}.json"
                
                print(f"  Looking for GT file: {gt_file}")
                print(f"  GT file exists: {gt_file.exists()}")
                if not gt_file.exists():
                    print(f"  Warning: GT file not found at {gt_file}")
                    print(f"  Trying alternative paths...")
                    # 대체 경로 시도
                    alt_paths = [
                        Path(self.config['data']['data_root'].replace('_YOLO', '')) / self.split / f"custom_{self.split}.json",
                        Path('data') / data_root.name.replace('_YOLO', '') / self.split / f"custom_{self.split}.json",
                    ]
                    for alt_path in alt_paths:
                        if alt_path.exists():
                            gt_file = alt_path
                            print(f"  Found GT file at alternative path: {gt_file}")
                            break
                
                if gt_file.exists():
                    print("\nCalculating Class-Agnostic mAP (CA-mAP)...")
                    
                    # GT 로드
                    coco_gt = COCO(str(gt_file))
                    
                    # COCO dataset에 필수 키가 없으면 추가
                    if 'info' not in coco_gt.dataset:
                        coco_gt.dataset['info'] = {
                            'description': 'COCO format dataset',
                            'version': '1.0',
                            'year': 2024
                        }
                    if 'licenses' not in coco_gt.dataset:
                        coco_gt.dataset['licenses'] = []
                    
                    # annotations에 'iscrowd' 필드가 없으면 추가
                    for ann_id in coco_gt.anns:
                        ann = coco_gt.anns[ann_id]
                        if 'iscrowd' not in ann:
                            ann['iscrowd'] = 0
                    
                    # 예측 결과 로드
                    with open(predictions_json, 'r') as f:
                        all_predictions = json.load(f)
                    
                    print(f"  Loaded {len(all_predictions)} predictions from {predictions_json}")
                    
                    if len(all_predictions) == 0:
                        print(f"  Warning: No predictions found in {predictions_json}")
                        raise ValueError("Empty predictions list")
                    
                    # GT COCO의 images에서 file_name → image_id 매핑 생성
                    # file_name/basename/stem 모두 매핑해서 string image_id 케이스 흡수
                    file_name_to_id = {}
                    for img_id, img_info in coco_gt.imgs.items():
                        file_name = img_info.get('file_name', '')
                        if not file_name:
                            continue
                        img_id_int = int(img_id)
                        file_name_to_id[file_name] = img_id_int
                        file_name_to_id[Path(file_name).name] = img_id_int
                        file_name_to_id[Path(file_name).stem] = img_id_int
                    
                    print(f"  Created file_name → image_id mapping: {len(file_name_to_id)} images")
                    
                    # 첫 번째 예측 확인
                    if len(all_predictions) > 0:
                        first_pred = all_predictions[0]
                        pred_img_id = first_pred.get('image_id')
                        pred_file_name = first_pred.get('file_name', '')
                        print(f"  First prediction: image_id={pred_img_id} (type: {type(pred_img_id).__name__}), file_name={pred_file_name}")
                    
                    # 모든 예측의 category_id를 0으로 변경하고 image_id를 numeric id로 교체
                    ca_predictions = []
                    unmapped_count = 0
                    for pred in all_predictions:
                        ca_pred = pred.copy()
                        ca_pred['category_id'] = 0  # 모든 클래스를 0으로 통합
                        
                        # image_id를 file_name으로 찾아서 numeric id로 교체
                        file_name = ca_pred.get('file_name', '')
                        original_img_id = ca_pred.get('image_id')
                        
                        mapped_id = None
                        
                        # 1. file_name으로 직접 매핑 시도
                        if file_name and file_name in file_name_to_id:
                            mapped_id = file_name_to_id[file_name]
                        # 2. image_id가 문자열인 경우 다양한 형태로 매핑 시도
                        elif isinstance(original_img_id, str):
                            if original_img_id in file_name_to_id:
                                mapped_id = file_name_to_id[original_img_id]
                            else:
                                stem = Path(original_img_id).stem
                                if stem in file_name_to_id:
                                    mapped_id = file_name_to_id[stem]
                        # 3. 이미 numeric id인 경우
                        elif isinstance(original_img_id, int):
                            mapped_id = original_img_id
                        
                        if mapped_id is not None:
                            ca_pred['image_id'] = mapped_id
                        else:
                            # 매핑 실패
                            unmapped_count += 1
                            if unmapped_count <= 3:  # 처음 3개만 경고
                                print(f"  Warning: Could not map image_id for prediction: image_id={original_img_id}, file_name={file_name}")
                            continue  # 매핑 실패한 예측은 건너뛰기
                        
                        ca_predictions.append(ca_pred)
                    
                    if unmapped_count > 0:
                        print(f"  Warning: {unmapped_count} predictions could not be mapped to GT image_ids")
                    
                    print(f"  Converted {len(ca_predictions)} predictions to class-agnostic format (mapped to numeric image_ids)")
                    
                    # CA 예측 결과 저장
                    ca_predictions_file = yolo_val_dir / "predictions_ca.json"
                    with open(ca_predictions_file, 'w') as f:
                        json.dump(ca_predictions, f)
                    
                    # GT의 모든 category_id를 0으로 변경한 복사본 생성
                    ca_gt_dataset = copy.deepcopy(coco_gt.dataset)
                    for ann in ca_gt_dataset['annotations']:
                        ann['category_id'] = 0
                    
                    # 카테고리를 하나로 통일 (tomato로 이름 변경)
                    if len(ca_gt_dataset['categories']) > 0:
                        ca_gt_dataset['categories'] = [{
                            'id': 0,
                            'name': 'tomato',
                            'supercategory': 'none'
                        }]
                    
                    # 임시 GT 파일 생성
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                        json.dump(ca_gt_dataset, f)
                        temp_gt_file = f.name
                    
                    # CA COCO 객체 생성 및 평가
                    print(f"  Creating COCO objects for CA-mAP calculation...")
                    ca_coco_gt = COCO(temp_gt_file)
                    print(f"  GT annotations: {len(ca_coco_gt.anns)}")
                    
                    ca_coco_dt = ca_coco_gt.loadRes(str(ca_predictions_file))
                    print(f"  DT predictions: {len(ca_coco_dt.anns)}")
                    
                    ca_coco_eval = COCOeval(ca_coco_gt, ca_coco_dt, 'bbox')
                    print(f"  Running COCOeval...")
                    ca_coco_eval.evaluate()
                    ca_coco_eval.accumulate()
                    ca_coco_eval.summarize()
                    
                    print(f"  COCOeval stats: {ca_coco_eval.stats}")
                    
                    ca_metrics = {
                        'ca_map': float(ca_coco_eval.stats[0]),
                        'ca_map_50': float(ca_coco_eval.stats[1]),
                        'ca_map_75': float(ca_coco_eval.stats[2]),
                    }
                    
                    print(f"  CA-mAP@0.50:      {ca_metrics['ca_map_50']:.4f}")
                    print(f"  CA-mAP@0.50:0.95: {ca_metrics['ca_map']:.4f}")
                    print(f"  CA-mAP@0.75:      {ca_metrics['ca_map_75']:.4f}")
                    
                    # 임시 파일 삭제
                    if os.path.exists(temp_gt_file):
                        os.unlink(temp_gt_file)
            except Exception as e:
                import traceback
                print(f"  Warning: Could not calculate CA-mAP: {e}")
                print(f"  Exception details:")
                traceback.print_exc()
                ca_metrics = {
                    'ca_map': 0.0,
                    'ca_map_50': 0.0,
                    'ca_map_75': 0.0,
                }
        else:
            print(f"  Warning: predictions.json not found in {yolo_val_dir}")
            print(f"  CA-mAP calculation skipped (predictions.json required)")
        
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
        
        # CA-mAP 출력 (계산된 경우)
        if ca_metrics:
            print("\n[Class-Agnostic mAP (Localization)]")
            print(f"CA-mAP@0.50:      {ca_metrics.get('ca_map_50', 0.0):.4f}")
            print(f"CA-mAP@0.50:0.95: {ca_metrics.get('ca_map', 0.0):.4f}")
            print(f"CA-mAP@0.75:      {ca_metrics.get('ca_map_75', 0.0):.4f}")
        
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
                # CA-mAP 추가
                'ca_map_50': ca_metrics.get('ca_map_50', 0.0),
                'ca_map': ca_metrics.get('ca_map', 0.0),
                'ca_map_75': ca_metrics.get('ca_map_75', 0.0),
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
