"""Evaluation Runner"""
from pathlib import Path
from datetime import datetime
from typing import Dict
import torch

from torch.utils.data import DataLoader

from ..registry import MODEL_REGISTRY, DATASET_REGISTRY
from ..evaluation.metrics import (
    evaluate_detection_metrics,
    evaluate_classification_metrics,
    calculate_model_complexity
)
from ..evaluation import (
    YOLOEvaluator,
    DETREvaluator,
    save_evaluation_results,
    print_evaluation_results
)
from ..utils.experiment import extract_exp_info_from_checkpoint


class EvaluationRunner:
    """평가 실행 클래스 - 메트릭 계산 및 분석"""
    
    def __init__(self, config, checkpoint_path, split='val', output_dir=None):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.split = split
        
        # 출력 디렉토리 설정
        if output_dir is None:
            # checkpoint 경로에서 exp 정보 추출
            exp_info = extract_exp_info_from_checkpoint(checkpoint_path, config)
            model_name = exp_info['model_name']
            exp_name = exp_info['exp_name']
            
            # exp/{model}/{exp_name}/evaluation 구조
            # Florence-2의 경우 mode 폴더 포함
            if model_name.lower() in ['florence2', 'florence-2']:
                florence2_mode = config.get('florence2', {}).get('mode', 'zeroshot')
                mode_folder = 'zeroshot' if florence2_mode == 'zeroshot' else 'finetuned'
                exp_dir = Path("exp") / model_name / mode_folder / exp_name
            else:
                exp_dir = Path("exp") / model_name / exp_name
            output_dir = exp_dir / "evaluation"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 평가 설정
        self.score_threshold = config.get('evaluation', {}).get('score_threshold', 0.5)
        self.iou_threshold = config.get('evaluation', {}).get('iou_threshold', 0.5)
        
        print(f"Evaluation output directory: {self.output_dir}")
        print(f"Score threshold: {self.score_threshold}")
        print(f"IoU threshold: {self.iou_threshold}")
    
    def _load_model_and_data(self):
        """모델과 데이터 로드"""
        model_name = self.config['model']['arch_name']
        dataset_name = self.config['data']['dataset_name']
        
        # Dataset 메타정보
        dataset_meta = DATASET_REGISTRY.get(dataset_name)
        
        # 모델별 처리
        if model_name == "DETR" or model_name == "detr":
            from transformers import DetrImageProcessor
            from ..data.transforms.detr_transform import create_detr_dataset, DetrCocoDataset
            
            data_cfg = self.config.get('data', {})
            processor_kwargs = {}
            img_size = data_cfg.get('image_size')
            max_size = data_cfg.get('max_size')
            # shortest_edge / longest_edge를 함께 지정하여 노트북 설정과 동일하게 적용
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
            
            # 평가할 데이터셋 로드
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
            
        elif model_name.lower() in ["yolov11", "yolov12", "yolo"]:
            # YOLO 모델: 변환 함수 없이 직접 로드
            # YOLO는 data.yaml을 직접 사용하므로 dataset 생성 불필요
            # MODEL_REGISTRY에서 모델 클래스 가져오기
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
            
            # YOLO는 data.yaml을 직접 사용하므로 dataset 생성 불필요
            dataset = None
            collate_fn = None
        
        elif model_name.lower() in ["groundingdino", "gdino", "grounding_dino"]:
            # Grounding DINO 모델
            from ..models.gdino_model import GroundingDINOWrapper, GroundingDINODataset
            from pathlib import Path
            
            # Grounding DINO 래퍼 생성
            wrapper = GroundingDINOWrapper(self.config)
            
            # 체크포인트에서 가중치 로드 (.pth 파일)
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                wrapper.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                wrapper.model.load_state_dict(checkpoint)
            
            wrapper.model.eval()
            
            # 데이터셋 생성
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
            
            # 모델을 래퍼로 반환 (processor 접근 필요)
            model = wrapper
        
        elif model_name.lower() in ["florence2", "florence-2", "florence"]:
            # Florence-2 모델
            from pathlib import Path
            
            florence2_mode = self.config.get('florence2', {}).get('mode', 'zeroshot')
            
            if florence2_mode == 'zeroshot':
                from ..models.florence2_base import Florence2Base
                # Zero-shot 모드는 checkpoint 불필요
                model = Florence2Base(
                    device=self.config.get('device', 'cuda'),
                    config=self.config
                )
            elif florence2_mode == 'finetuned':
                from ..models.florence2_finetuned import Florence2Finetuned
                # Fine-tuned 모드는 체크포인트 필요
                checkpoint_path = self.checkpoint_path
                if not checkpoint_path or str(checkpoint_path).lower() == 'none':
                    checkpoint_path = self.config.get('florence2', {}).get('checkpoint_path')
                if not checkpoint_path or str(checkpoint_path).lower() == 'none':
                    raise FileNotFoundError(
                        f"Fine-tuned mode requires valid checkpoint.\n"
                        f"  Please specify --checkpoint or set florence2.checkpoint_path in config"
                    )
                model = Florence2Finetuned(
                    checkpoint_path=checkpoint_path,
                    device=self.config.get('device', 'cuda'),
                    config=self.config
                )
            else:
                raise ValueError(f"Unknown Florence-2 mode: {florence2_mode}. Use 'zeroshot' or 'finetuned'")
            
            # 데이터셋은 평가 시 직접 로드 (Florence-2는 특별한 데이터셋 클래스 불필요)
            dataset = None
            collate_fn = None
        
        else:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
        
        # YOLO와 Florence-2 모델은 eval() 호출 불필요
        # - YOLO: 별도 평가 경로 사용
        # - Florence-2: 래퍼 클래스이며 이미 __init__에서 model.eval() 호출됨
        if model_name.lower() not in ["yolov11", "yolov12", "yolo", "florence2", "florence-2", "florence_2"]:
            model.eval()
        
        return model, dataset, collate_fn
    
    def _create_dataloader(self, dataset, collate_fn):
        """DataLoader 생성"""
        batch_size = self.config.get('evaluation', {}).get('batch_size', 
                                                          self.config['data']['batch_size'])
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config['data'].get('num_workers', 0),
            collate_fn=collate_fn
        )
    
    def run(self):
        """평가 실행"""
        print("\n" + "="*70)
        print("EVALUATION STARTED")
        print("="*70)
        print(f"Model: {self.config['model']['arch_name']}")
        print(f"Dataset: {self.config['data']['dataset_name']}")
        print(f"Split: {self.split}")
        print(f"Checkpoint: {Path(self.checkpoint_path).name}")
        print("="*70)
        
        model_name = self.config['model']['arch_name'].lower()
        
        # YOLO 모델은 별도 평가 경로 사용
        if model_name in ["yolov11", "yolov12", "yolo"]:
            evaluator = YOLOEvaluator(
                config=self.config,
                checkpoint_path=self.checkpoint_path,
                split=self.split,
                output_dir=self.output_dir,
                score_threshold=self.score_threshold,
                iou_threshold=self.iou_threshold
            )
            return evaluator.evaluate()
        
        # Grounding DINO 모델은 별도 평가 경로 사용
        if model_name in ["groundingdino", "gdino", "grounding_dino"]:
            from ..models.gdino_model import GroundingDINOWrapper, GroundingDINODataset
            
            print("\nLoading Grounding DINO model and dataset...")
            
            # Wrapper 생성 및 체크포인트 로드
            wrapper = GroundingDINOWrapper(self.config)
            # GPU로 직접 로드
            checkpoint = torch.load(self.checkpoint_path, map_location=wrapper.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                wrapper.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                wrapper.model.load_state_dict(checkpoint)
            wrapper.model.eval()
            
            # 데이터셋 로드
            dataset_meta = DATASET_REGISTRY.get(self.config['data']['dataset_name'])
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
            dataloader = self._create_dataloader(dataset, collate_fn)
            
            print(f"Dataset samples: {len(dataset)}")
            print(f"Evaluation batches: {len(dataloader)}")
            
            # Grounding DINO 평가 실행
            return self._evaluate_grounding_dino(wrapper, dataloader)
        
        # Florence-2 모델은 별도 평가 경로 사용
        if model_name in ["florence2", "florence-2", "florence"]:
            return self._evaluate_florence2()
        
        # DETR 모델은 DETREvaluator 사용
        if model_name in ["detr"]:
            # 모델 및 데이터 로드
            print("\nLoading model and dataset...")
            model, dataset, collate_fn = self._load_model_and_data()
            print(f"Dataset samples: {len(dataset)}")
            
            # DataLoader 생성
            dataloader = self._create_dataloader(dataset, collate_fn)
            print(f"Evaluation batches: {len(dataloader)}")
            
            # DETREvaluator 사용
            evaluator = DETREvaluator(
                config=self.config,
                output_dir=self.output_dir,
                score_threshold=self.score_threshold,
                iou_threshold=self.iou_threshold,
                split=self.split
            )
            return evaluator.evaluate(model, dataloader)
        
        # 기타 모델은 기존 방식 (하위 호환성)
        # 1. 모델 및 데이터 로드
        print("\nLoading model and dataset...")
        model, dataset, collate_fn = self._load_model_and_data()
        print(f"Dataset samples: {len(dataset)}")
        
        # 2. DataLoader 생성
        dataloader = self._create_dataloader(dataset, collate_fn)
        print(f"Evaluation batches: {len(dataloader)}")
        
        # 3. 모델 복잡도 계산
        print("\nCalculating model complexity...")
        complexity_metrics = calculate_model_complexity(model, self.config)
        if complexity_metrics['gflops']:
            print(f"  Parameters: {complexity_metrics['params_m']:.2f}M")
            print(f"  Model Size: {complexity_metrics['model_size_mb']:.2f} MB")
            print(f"  GFLOPs: {complexity_metrics['gflops']:.2f}")
        
        # 4. 상세 Detection 메트릭 계산
        print("\nCalculating detailed detection metrics...")
        detection_results, detailed_stats, predictions, targets = evaluate_detection_metrics(
            model=model,
            dataloader=dataloader,
            config=self.config,
            score_threshold=self.score_threshold,
            iou_threshold=self.iou_threshold,
            split=self.split
        )
        
        # 5. Classification 메트릭 계산
        print("Calculating classification metrics...")
        classification_results = evaluate_classification_metrics(
            all_predictions=predictions,
            all_targets=targets,
            config=self.config
        )
        
        # 6. 결과 저장 및 시각화
        print("\nSaving results and creating visualizations...")
        saved_results = save_evaluation_results(
            detection_metrics=detection_results,
            detailed_stats=detailed_stats,
            classification_results=classification_results,
            complexity_metrics=complexity_metrics,
            output_dir=self.output_dir,
            config=self.config,
            checkpoint_path=self.checkpoint_path,
            split=self.split,
            score_threshold=self.score_threshold,
            iou_threshold=self.iou_threshold
        )
        
        # 최종 결과 통합
        final_results = {
            'detection_metrics': detection_results,
            'detailed_statistics': detailed_stats,
            'classification_metrics': classification_results,
            'model_complexity': complexity_metrics,
        }
        
        # 7. 상세 결과 출력
        print_evaluation_results(
            detection_metrics=detection_results,
            detailed_stats=detailed_stats,
            classification_results=classification_results,
            config=self.config,
            split=self.split,
            score_threshold=self.score_threshold,
            iou_threshold=self.iou_threshold
        )
        
        # 8. 복잡도 메트릭 출력
        if complexity_metrics:
            print("\n" + "="*70)
            print("MODEL COMPLEXITY")
            print("="*70)
            print(f"  Total Parameters: {complexity_metrics['params_m']:.2f}M")
            print(f"  Trainable Parameters: {complexity_metrics['trainable_params'] / 1e6:.2f}M")
            print(f"  Model Size: {complexity_metrics['model_size_mb']:.2f} MB")
            if complexity_metrics.get('gflops') is not None:
                print(f"  GFLOPs: {complexity_metrics['gflops']:.2f}")
            print("="*70)
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETED!")
        print("="*70)
        print(f"Results directory: {self.output_dir}")
        print(f"  - evaluation_results.json: 전체 결과")
        print(f"  - summary_metrics.csv: 성능 요약표")
        if classification_results:
            print(f"  - confusion_matrix.png: 혼동 행렬")
        print("="*70 + "\n")
        
        return final_results
    
    def _evaluate_grounding_dino(self, wrapper, dataloader):
        """Grounding DINO 평가 실행"""
        import json
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        from tqdm import tqdm
        import numpy as np
        
        # 예측 결과 수집
        all_predictions = []
        
        wrapper.model.eval()
        device = wrapper.device
        
        print("\nRunning inference...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
                pixel_values = batch['pixel_values'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                image_ids = batch.get('image_id', batch.get('image_ids', []))
                
                # Forward pass
                outputs = wrapper.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # 후처리
                target_sizes = torch.tensor([[2000, 2000]] * pixel_values.shape[0], device=device)
                
                # 예측 결과 후처리
                results = wrapper.processor.post_process_grounded_object_detection(
                    outputs,
                    input_ids=input_ids,
                    threshold=self.score_threshold,
                    text_threshold=self.score_threshold,
                    target_sizes=target_sizes
                )
                
                # 예측 저장
                for i, result in enumerate(results):
                    img_id = image_ids[i] if i < len(image_ids) else batch_idx * dataloader.batch_size + i
                    
                    boxes = result['boxes'].cpu().numpy()
                    scores = result['scores'].cpu().numpy()
                    labels = result['labels']
                    text_labels = result.get('text_labels', [])
                    
                    for box, score, label in zip(boxes, scores, labels):
                        x_min, y_min, x_max, y_max = box
                        width = x_max - x_min
                        height = y_max - y_min
                        
                        # label이 리스트인 경우 첫 번째 요소 사용 또는 인덱스로 변환
                        if isinstance(label, list) and len(label) > 0:
                            label_id = label[0] if isinstance(label[0], int) else 0
                        elif isinstance(label, int):
                            label_id = label
                        else:
                            label_id = 0
                        
                        all_predictions.append({
                            'image_id': int(img_id),
                            'category_id': int(label_id),
                            'bbox': [float(x_min), float(y_min), float(width), float(height)],
                            'score': float(score)
                        })
        
        # COCO 형식으로 결과 저장
        results_file = self.output_dir / "predictions.json"
        with open(results_file, 'w') as f:
            json.dump(all_predictions, f)
        
        # GT 파일 경로
        dataset_meta = DATASET_REGISTRY.get(self.config['data']['dataset_name'])
        paths = dataset_meta.get_data_paths(self.split, self.config)
        gt_file = Path(paths['ann_file'])
        
        if not gt_file.exists():
            raise FileNotFoundError(f"GT annotation file not found: {gt_file}")
        
        # COCO 평가
        print("\nCalculating COCO metrics...")
        coco_gt = COCO(str(gt_file))
        
        # COCO dataset에 필수 키가 없으면 추가 (loadRes에서 필요)
        if 'info' not in coco_gt.dataset:
            coco_gt.dataset['info'] = {
                'description': 'COCO format dataset',
                'version': '1.0',
                'year': 2024
            }
        if 'licenses' not in coco_gt.dataset:
            coco_gt.dataset['licenses'] = []
        
        # annotations에 'iscrowd' 필드가 없으면 추가 (COCOeval에서 필요)
        for ann_id in coco_gt.anns:
            ann = coco_gt.anns[ann_id]
            if 'iscrowd' not in ann:
                ann['iscrowd'] = 0
        
        coco_dt = coco_gt.loadRes(str(results_file))
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # 결과 저장
        metrics = {
            'mAP': float(coco_eval.stats[0]),
            'mAP_50': float(coco_eval.stats[1]),
            'mAP_75': float(coco_eval.stats[2]),
            'mAP_small': float(coco_eval.stats[3]),
            'mAP_medium': float(coco_eval.stats[4]),
            'mAP_large': float(coco_eval.stats[5]),
            'mAR_1': float(coco_eval.stats[6]),
            'mAR_10': float(coco_eval.stats[7]),
            'mAR_100': float(coco_eval.stats[8]),
            'mAR_small': float(coco_eval.stats[9]),
            'mAR_medium': float(coco_eval.stats[10]),
            'mAR_large': float(coco_eval.stats[11]),
        }
        
        # CA-mAP 계산 추가
        print("\nCalculating Class-Agnostic mAP (CA-mAP)...")
        import copy
        import tempfile
        import os
        
        # 예측 결과 로드
        with open(results_file, 'r') as f:
            all_predictions_ca = json.load(f)
        
        # 모든 예측의 category_id를 0으로 변경
        ca_predictions = []
        for pred in all_predictions_ca:
            ca_pred = pred.copy()
            ca_pred['category_id'] = 0  # 모든 클래스를 0으로 통합
            ca_predictions.append(ca_pred)
        
        # CA 예측 결과 저장
        ca_predictions_file = self.output_dir / "predictions_ca.json"
        with open(ca_predictions_file, 'w') as f:
            json.dump(ca_predictions, f)
        
        # GT의 모든 category_id를 0으로 변경
        ca_gt_dataset = copy.deepcopy(coco_gt.dataset)
        for ann in ca_gt_dataset['annotations']:
            ann['category_id'] = 0
        
        # 카테고리 통일
        if len(ca_gt_dataset['categories']) > 0:
            ca_gt_dataset['categories'] = [{'id': 0, 'name': 'object', 'supercategory': 'none'}]
        
        # 임시 GT 파일 생성 및 평가
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(ca_gt_dataset, f)
            temp_gt_file = f.name
        
        try:
            ca_coco_gt = COCO(temp_gt_file)
            ca_coco_dt = ca_coco_gt.loadRes(str(ca_predictions_file))
            ca_coco_eval = COCOeval(ca_coco_gt, ca_coco_dt, 'bbox')
            ca_coco_eval.evaluate()
            ca_coco_eval.accumulate()
            ca_coco_eval.summarize()
            
            # CA-mAP 메트릭 추가
            metrics['ca_mAP'] = float(ca_coco_eval.stats[0])
            metrics['ca_mAP_50'] = float(ca_coco_eval.stats[1])
            metrics['ca_mAP_75'] = float(ca_coco_eval.stats[2])
        except Exception as e:
            print(f"  Warning: Could not calculate CA-mAP: {e}")
            metrics['ca_mAP'] = 0.0
            metrics['ca_mAP_50'] = 0.0
            metrics['ca_mAP_75'] = 0.0
        finally:
            # 임시 파일 삭제
            if os.path.exists(temp_gt_file):
                os.unlink(temp_gt_file)
        
        results = {
            'metrics': metrics,
            'predictions_file': str(results_file),
            'gt_file': str(gt_file),
            'num_predictions': len(all_predictions)
        }
        
        # 결과 저장
        results_file_json = self.output_dir / "results.json"
        with open(results_file_json, 'w') as f:
            json.dump(results, f, indent=2)
        
        # 결과 출력
        print("\n" + "="*70)
        print("GROUNDING DINO 평가 결과")
        print("="*70)
        print(f"\n[Standard mAP - Class-Aware]")
        print(f"  mAP (0.5:0.95): {metrics['mAP']:.4f}")
        print(f"  mAP_50: {metrics['mAP_50']:.4f}")
        print(f"  mAP_75: {metrics['mAP_75']:.4f}")
        print(f"  mAP_small: {metrics['mAP_small']:.4f}")
        print(f"  mAP_medium: {metrics['mAP_medium']:.4f}")
        print(f"  mAP_large: {metrics['mAP_large']:.4f}")
        
        print(f"\n[CA-mAP - Class-Agnostic (Localization)]")
        print(f"  CA-mAP (0.5:0.95): {metrics.get('ca_mAP', 0.0):.4f}")
        print(f"  CA-mAP_50: {metrics.get('ca_mAP_50', 0.0):.4f}")
        print(f"  CA-mAP_75: {metrics.get('ca_mAP_75', 0.0):.4f}")
        
        print(f"\n[Recall]")
        print(f"  mAR_1: {metrics['mAR_1']:.4f}")
        print(f"  mAR_10: {metrics['mAR_10']:.4f}")
        print(f"  mAR_100: {metrics['mAR_100']:.4f}")
        print(f"  mAR_small: {metrics['mAR_small']:.4f}")
        print(f"  mAR_medium: {metrics['mAR_medium']:.4f}")
        print(f"  mAR_large: {metrics['mAR_large']:.4f}")
        print(f"\n[기타]")
        print(f"  총 예측 수: {len(all_predictions)}")
        print(f"  결과 파일: {results_file_json}")
        print("="*70 + "\n")
        
        return results
    
    def _evaluate_florence2(self):
        """Florence-2 모델 평가"""
        from ..registry import DATASET_REGISTRY
        from tqdm import tqdm
        import json
        from pathlib import Path
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        
        print("\nLoading Florence-2 model...")
        
        # 모델 로드
        model, _, _ = self._load_model_and_data()
        
        # Florence-2 모드 확인
        florence2_mode = self.config.get('florence2', {}).get('mode', 'zeroshot')
        
        # 데이터셋 로드
        dataset_meta = DATASET_REGISTRY.get(self.config['data']['dataset_name'])
        if dataset_meta is None:
            raise ValueError(f"Dataset '{self.config['data']['dataset_name']}' not found")
        
        paths = dataset_meta.get_data_paths(self.split, self.config)
        ann_file = Path(paths['ann_file'])
        images_dir = Path(paths['img_folder'])
        
        # COCO annotations 로드
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
        
        images = {img['id']: img for img in coco_data['images']}
        
        print(f"Dataset samples: {len(images)}")
        
        # 클래스 이름 매핑
        class_names = self.config['data'].get('class_names', ['tomato'])
        num_classes = len(class_names)
        name_to_id = {name: i for i, name in enumerate(class_names)}
        
        # Florence-2 설정
        nms_threshold = self.config.get('florence2', {}).get('nms_threshold', 0.5)
        conf_threshold = self.config.get('florence2', {}).get('conf_threshold', 0.0)
        
        print(f"Florence-2 mode: {florence2_mode}")
        print(f"NMS threshold: {nms_threshold}")
        print(f"Confidence threshold: {conf_threshold}")
        
        print("\nRunning inference...")
        all_predictions = []
        
        # 각 이미지에 대해 추론
        for img_id, img_info in tqdm(list(images.items()), desc="Evaluating"):
            image_path = images_dir / img_info['file_name']
            
            if not image_path.exists():
                print(f"\n  Image not found: {image_path}")
                continue
            
            try:
                # 추론
                if florence2_mode == 'zeroshot':
                    # Zero-shot: 클래스 이름을 프롬프트에 포함
                    if num_classes == 1:
                        # 1-class: 단순 tomato 검색
                        task_prompt = "<CAPTION_TO_PHRASE_GROUNDING> tomato"
                    else:
                        # 3-class: 모든 클래스를 명시
                        class_labels = ", ".join(class_names)  # "fully_ripe tomato, semi_ripe tomato, unripe tomato"
                        task_prompt = f"<CAPTION_TO_PHRASE_GROUNDING> {class_labels}"
                    
                    result = model.predict(str(image_path), task=task_prompt)
                    
                    # 결과 파싱
                    detections = []
                    # <CAPTION_TO_PHRASE_GROUNDING> 결과는 '<OD>' 키를 사용할 수도 있음
                    result_key = '<CAPTION_TO_PHRASE_GROUNDING>' if '<CAPTION_TO_PHRASE_GROUNDING>' in result else '<OD>'
                    if result_key in result:
                        det = result[result_key]
                        bboxes = det.get('bboxes', [])
                        labels = det.get('labels', [])
                        
                        for bbox, label in zip(bboxes, labels):
                            # 레이블을 클래스로 매핑 (간단한 키워드 매칭)
                            class_name = self._classify_florence_label(label, class_names)
                            if class_name in name_to_id:
                                detections.append({
                                    'bbox': bbox,
                                    'class': class_name,
                                    'class_id': name_to_id[class_name],
                                    'score': 1.0
                                })
                
                elif florence2_mode == 'finetuned':
                    # Fine-tuned: predict_finetuned 사용
                    # ✅ predict_finetuned가 이미 픽셀 좌표와 surrogate score를 반환
                    detections = model.predict_finetuned(
                        str(image_path),
                        conf_threshold=conf_threshold
                    )
                
                # NMS 적용 (간단한 구현)
                if len(detections) > 1:
                    detections = self._apply_nms(detections, nms_threshold)
                
                # COCO 형식으로 변환
                # ✅ bbox는 이미 픽셀 좌표 (predict_finetuned에서 변환됨)
                # ✅ score는 surrogate score (0.3~0.95 범위)
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Bbox가 이미지 범위 내인지 확인
                    img_w, img_h = img_info['width'], img_info['height']
                    x1 = max(0, min(img_w, x1))
                    y1 = max(0, min(img_h, y1))
                    x2 = max(0, min(img_w, x2))
                    y2 = max(0, min(img_h, y2))
                    width = x2 - x1
                    height = y2 - y1
                    
                    # 유효한 bbox만 추가
                    if width > 0 and height > 0:
                        all_predictions.append({
                            'image_id': int(img_id),
                            'category_id': int(det['class_id']),
                            'bbox': [float(x1), float(y1), float(width), float(height)],
                            'score': float(det['score'])  # Surrogate score
                        })
            
            except Exception as e:
                print(f"\n  Error processing {image_path}: {e}")
                continue
        
        # COCO 형식으로 결과 저장
        results_file = self.output_dir / "predictions.json"
        with open(results_file, 'w') as f:
            json.dump(all_predictions, f)
        
        # GT 파일 경로
        gt_file = ann_file
        
        if not gt_file.exists():
            raise FileNotFoundError(f"GT annotation file not found: {gt_file}")
        
        # COCO 평가
        print("\nCalculating COCO metrics...")
        coco_gt = COCO(str(gt_file))
        
        # COCO dataset에 필수 키가 없으면 추가 (loadRes에서 필요)
        if 'info' not in coco_gt.dataset:
            coco_gt.dataset['info'] = {
                'description': 'COCO format dataset',
                'version': '1.0',
                'year': 2024
            }
        if 'licenses' not in coco_gt.dataset:
            coco_gt.dataset['licenses'] = []
        
        # annotations에 'iscrowd' 필드가 없으면 추가 (COCOeval에서 필요)
        for ann_id in coco_gt.anns:
            ann = coco_gt.anns[ann_id]
            if 'iscrowd' not in ann:
                ann['iscrowd'] = 0
        
        coco_dt = coco_gt.loadRes(str(results_file))
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Precision, Recall 계산 (IoU=0.5 기준)
        precision = coco_eval.eval['precision'][0, :, :, 0, 2].mean()  # IoU=0.5, all classes, medium
        recall = coco_eval.eval['recall'][0, :, 0, 2].mean()  # IoU=0.5, all classes, medium
        
        # precision이나 recall이 -1이면 0으로 처리
        if precision == -1:
            precision = 0.0
        if recall == -1:
            recall = 0.0
        
        # F1 계산
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        # 클래스별 AP 계산
        per_class_ap_50 = {}
        per_class_ap_50_95 = {}
        
        for cat_id, cat_name in enumerate(class_names):
            coco_eval_class = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval_class.params.catIds = [cat_id]
            coco_eval_class.evaluate()
            coco_eval_class.accumulate()
            coco_eval_class.summarize()
            
            per_class_ap_50[cat_name] = float(coco_eval_class.stats[1])  # AP@0.5
            per_class_ap_50_95[cat_name] = float(coco_eval_class.stats[0])  # AP@0.5:0.95
        
        # CA-mAP 계산 추가
        print("\nCalculating Class-Agnostic mAP (CA-mAP)...")
        import copy
        import tempfile
        import os
        
        # 예측 결과 로드
        with open(results_file, 'r') as f:
            all_predictions_ca = json.load(f)
        
        # 모든 예측의 category_id를 0으로 변경
        ca_predictions = []
        for pred in all_predictions_ca:
            ca_pred = pred.copy()
            ca_pred['category_id'] = 0  # 모든 클래스를 0으로 통합
            ca_predictions.append(ca_pred)
        
        # CA 예측 결과 저장
        ca_predictions_file = self.output_dir / "predictions_ca.json"
        with open(ca_predictions_file, 'w') as f:
            json.dump(ca_predictions, f)
        
        # GT의 모든 category_id를 0으로 변경
        ca_gt_dataset = copy.deepcopy(coco_gt.dataset)
        for ann in ca_gt_dataset['annotations']:
            ann['category_id'] = 0
        
        # 카테고리 통일
        if len(ca_gt_dataset['categories']) > 0:
            ca_gt_dataset['categories'] = [{'id': 0, 'name': 'object', 'supercategory': 'none'}]
        
        # 임시 GT 파일 생성 및 평가
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(ca_gt_dataset, f)
            temp_gt_file = f.name
        
        ca_map_50 = 0.0
        ca_map = 0.0
        ca_map_75 = 0.0
        
        try:
            ca_coco_gt = COCO(temp_gt_file)
            ca_coco_dt = ca_coco_gt.loadRes(str(ca_predictions_file))
            ca_coco_eval = COCOeval(ca_coco_gt, ca_coco_dt, 'bbox')
            ca_coco_eval.evaluate()
            ca_coco_eval.accumulate()
            ca_coco_eval.summarize()
            
            # CA-mAP 메트릭
            ca_map_50 = float(ca_coco_eval.stats[1])
            ca_map = float(ca_coco_eval.stats[0])
            ca_map_75 = float(ca_coco_eval.stats[2])
        except Exception as e:
            print(f"  Warning: Could not calculate CA-mAP: {e}")
        finally:
            # 임시 파일 삭제
            if os.path.exists(temp_gt_file):
                os.unlink(temp_gt_file)
        
        # Model complexity 계산 (Florence2Base는 model.model에 실제 모델이 있음)
        print("\nCalculating model complexity...")
        try:
            # Florence2Base는 래퍼 클래스이므로 model.model 사용
            total_params = sum(p.numel() for p in model.model.parameters())
            trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
            
            # MB 계산 (float32 기준: 4 bytes per parameter)
            model_size_mb = total_params * 4 / (1024 ** 2)
            
            # GFLOPs 계산 (대략적으로 추정)
            # Florence-2-base는 약 232M 파라미터, ~70 GFLOPs 정도
            try:
                # 간단한 추정: 파라미터 수 기반
                gflops = total_params / 1e9 * 0.3  # 대략적 추정
            except:
                gflops = 0.0
            
            model_complexity = {
                'total_params': int(total_params),
                'trainable_params': int(trainable_params),
                'params_m': round(total_params / 1e6, 6),
                'model_size_mb': round(model_size_mb, 2),
                'gflops': round(gflops, 6),
                'gflops_formatted': f"{gflops:.3f}G"
            }
            
            print(f"  Total params: {model_complexity['params_m']:.2f}M")
            print(f"  Model size: {model_complexity['model_size_mb']:.2f} MB")
            print(f"  GFLOPs: {model_complexity['gflops_formatted']}")
        except Exception as e:
            print(f"  Warning: Could not calculate model complexity: {e}")
            model_complexity = {
                'total_params': 0,
                'trainable_params': 0,
                'params_m': 0.0,
                'model_size_mb': 0.0,
                'gflops': 0.0,
                'gflops_formatted': "0.000G"
            }
        
        # YOLO와 동일한 형식으로 결과 정리
        results = {
            'detection_metrics': {
                'map_50': float(coco_eval.stats[1]),  # mAP@0.5
                'map': float(coco_eval.stats[0]),  # mAP@0.5:0.95
                'map_75': float(coco_eval.stats[2]),  # mAP@0.75
                'ca_map_50': ca_map_50,  # CA-mAP@0.5
                'ca_map': ca_map,  # CA-mAP@0.5:0.95
                'ca_map_75': ca_map_75  # CA-mAP@0.75
            },
            'detailed_statistics': {
                'total_statistics': {
                    'overall_precision': float(precision),
                    'overall_recall': float(recall),
                    'overall_f1': float(f1)
                },
                'per_class_ap': {
                    'ap_50': per_class_ap_50,
                    'ap_50_95': per_class_ap_50_95,
                    'ap_75': {}  # 필요시 추가 가능
                }
            },
            'model_complexity': model_complexity,
            'model_info': {
                'backbone': 'florence-2-base',
                'arch_name': 'florence2',
                'mode': florence2_mode
            }
        }
        
        # 결과 저장
        results_file_json = self.output_dir / "evaluation_results.json"
        with open(results_file_json, 'w') as f:
            json.dump(results, f, indent=2)
        
        # 결과 출력
        print("\n" + "="*70)
        print("FLORENCE-2 평가 결과")
        print("="*70)
        print(f"\n Detection Metrics:")
        print(f"   mAP@0.5:        {results['detection_metrics']['map_50']:.4f}")
        print(f"   mAP@0.75:       {results['detection_metrics']['map_75']:.4f}")
        print(f"   mAP@[0.5:0.95]: {results['detection_metrics']['map']:.4f}")
        
        print(f"\n Class-Agnostic mAP (CA-mAP):")
        print(f"   CA-mAP@0.5:        {results['detection_metrics']['ca_map_50']:.4f}")
        print(f"   CA-mAP@0.75:       {results['detection_metrics']['ca_map_75']:.4f}")
        print(f"   CA-mAP@[0.5:0.95]: {results['detection_metrics']['ca_map']:.4f}")
        
        print(f"\n Overall Statistics:")
        print(f"   Precision:      {results['detailed_statistics']['total_statistics']['overall_precision']:.4f}")
        print(f"   Recall:         {results['detailed_statistics']['total_statistics']['overall_recall']:.4f}")
        print(f"   F1 Score:       {results['detailed_statistics']['total_statistics']['overall_f1']:.4f}")
        
        print(f"\n Per-class AP@0.5:")
        for class_name, ap in results['detailed_statistics']['per_class_ap']['ap_50'].items():
            print(f"   {class_name:15s}: {ap:.4f}")
        
        print(f"\n Per-class AP@[0.5:0.95]:")
        for class_name, ap in results['detailed_statistics']['per_class_ap']['ap_50_95'].items():
            print(f"   {class_name:15s}: {ap:.4f}")
        
        print(f"\n Model Complexity:")
        print(f"   Total params:   {results['model_complexity']['params_m']:.2f}M")
        print(f"   Model size:     {results['model_complexity']['model_size_mb']:.2f} MB")
        print(f"   GFLOPs:         {results['model_complexity']['gflops_formatted']}")
        
        print(f"\n[기타]")
        print(f"   총 예측 수: {len(all_predictions)}")
        print(f"   결과 파일: {results_file_json}")
        print("="*70 + "\n")
        
        return results
    
    def _classify_florence_label(self, label, class_names):
        """Florence-2 레이블을 클래스 이름으로 분류"""
        label_lower = str(label).lower()
        
        # 1-class인 경우
        if len(class_names) == 1:
            return class_names[0]
        
        # 3-class인 경우 키워드 매칭
        if 'fully' in label_lower or 'full' in label_lower or 'ripe' in label_lower:
            for name in class_names:
                if 'fully' in name.lower() or 'ripe' in name.lower():
                    return name
        elif 'semi' in label_lower or 'partial' in label_lower:
            for name in class_names:
                if 'semi' in name.lower():
                    return name
        elif 'unripe' in label_lower or 'green' in label_lower:
            for name in class_names:
                if 'unripe' in name.lower():
                    return name
        
        # 기본값: 첫 번째 클래스
        return class_names[0]
    
    def _apply_nms(self, detections, iou_threshold):
        """간단한 NMS 구현"""
        if len(detections) <= 1:
            return detections
        
        import torch
        from torchvision.ops import nms
        
        boxes = torch.tensor([d['bbox'] for d in detections], dtype=torch.float32)
        scores = torch.tensor([d['score'] for d in detections], dtype=torch.float32)
        
        keep_indices = nms(boxes, scores, iou_threshold)
        keep_indices = keep_indices.cpu().numpy()
        
        return [detections[i] for i in keep_indices]
