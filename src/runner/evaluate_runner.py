"""Evaluation Runner"""
from pathlib import Path
from datetime import datetime
from typing import Dict

from torch.utils.data import DataLoader

from ..registry import MODEL_REGISTRY, DATASET_REGISTRY
from ..metrics import (
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
            
            # results/{model}/{exp_name}/evaluation 구조
            output_dir = Path("results") / model_name / exp_name / "evaluation"
        
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
            
            imageprocessor = DetrImageProcessor.from_pretrained(
                self.config['model']['pretrained_path']
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
        
        else:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
        
        # YOLO 모델은 eval() 호출 불필요 (별도 평가 경로 사용)
        if model_name.lower() not in ["yolov11", "yolov12", "yolo"]:
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
