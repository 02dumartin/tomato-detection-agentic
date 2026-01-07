"""DETR 모델 평가 유틸리티"""
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List

from ..metrics import (
    evaluate_detection_metrics,
    evaluate_classification_metrics,
    calculate_model_complexity,
)
from .result_saver import save_summary_metrics


class DETREvaluator:
    """DETR 계열 모델 평가 클래스.

    - 모델 복잡도 (파라미터, GFLOPs)
    - Detection 메트릭 (mAP, TP/FP/FN 등)
    - Classification 메트릭
    을 한 번에 계산하고 YOLO와 동일한 형식으로 저장
    """

    def __init__(
        self,
        config: Dict,
        output_dir: Path = None,
        score_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        split: str = "val",
    ):
        self.config = config
        self.output_dir = output_dir
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.split = split

    def evaluate(
        self,
        model,
        dataloader,
    ) -> Dict:
        """DETR 모델 평가 실행.

        Args:
            model: 평가할 DETR 계열 모델 (LightningModule 등)
            dataloader: 평가용 DataLoader

        Returns:
            results_dict: 평가 결과 딕셔너리 (YOLO와 동일한 형식)
        """
        # 모델 복잡도 계산
        complexity_metrics = {}
        try:
            complexity_metrics = calculate_model_complexity(model, self.config)
        except Exception as e:
            print(f"  Could not calculate model complexity: {e}")

        # Detection 메트릭 계산
        detection_results, detailed_stats, predictions, targets = evaluate_detection_metrics(
            model=model,
            dataloader=dataloader,
            config=self.config,
            score_threshold=self.score_threshold,
            iou_threshold=self.iou_threshold,
            split=self.split,
        )

        # Classification 메트릭 계산
        classification_results = evaluate_classification_metrics(
            all_predictions=predictions,
            all_targets=targets,
            config=self.config,
        )

        # 4. 클래스별 AP 추출 (YOLO 형식과 동일하게)
        class_names = self.config['data']['class_names']
        class_ap_50 = {}
        class_ap_50_95 = {}
        class_ap_75 = {}

        # detection_results에서 클래스별 AP 추출
        if 'map_per_class' in detection_results:
            map_per_class = detection_results['map_per_class']
            if hasattr(map_per_class, 'tolist'):
                map_per_class = map_per_class.tolist()
            elif hasattr(map_per_class, '__iter__') and not isinstance(map_per_class, str):
                map_per_class = list(map_per_class)
            else:
                map_per_class = [map_per_class] if map_per_class is not None else []

            # map_per_class는 보통 mAP@0.5:0.95에 해당
            for i, ap in enumerate(map_per_class):
                if i < len(class_names):
                    class_ap_50_95[class_names[i]] = float(ap)

        # map_50_per_class가 있으면 사용
        if 'map_50_per_class' in detection_results:
            map_50_per_class = detection_results['map_50_per_class']
            if hasattr(map_50_per_class, 'tolist'):
                map_50_per_class = map_50_per_class.tolist()
            elif hasattr(map_50_per_class, '__iter__') and not isinstance(map_50_per_class, str):
                map_50_per_class = list(map_50_per_class)
            else:
                map_50_per_class = [map_50_per_class] if map_50_per_class is not None else []

            for i, ap in enumerate(map_50_per_class):
                if i < len(class_names):
                    class_ap_50[class_names[i]] = float(ap)

        # map_75_per_class가 있으면 사용
        if 'map_75_per_class' in detection_results:
            map_75_per_class = detection_results['map_75_per_class']
            if hasattr(map_75_per_class, 'tolist'):
                map_75_per_class = map_75_per_class.tolist()
            elif hasattr(map_75_per_class, '__iter__') and not isinstance(map_75_per_class, str):
                map_75_per_class = list(map_75_per_class)
            else:
                map_75_per_class = [map_75_per_class] if map_75_per_class is not None else []

            for i, ap in enumerate(map_75_per_class):
                if i < len(class_names):
                    class_ap_75[class_names[i]] = float(ap)

        # 결과 출력
        print("\n" + "="*70)
        print("DETR EVALUATION RESULTS")
        print("="*70)
        print(f"mAP@0.50:      {detection_results['map_50']:.4f}")
        print(f"mAP@0.50:0.95: {detection_results['map']:.4f}")
        print(f"mAP@0.75:      {detection_results['map_75']:.4f}")
        print(f"Precision:     {detailed_stats['total_statistics']['overall_precision']:.4f}")
        print(f"Recall:        {detailed_stats['total_statistics']['overall_recall']:.4f}")
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

        # Backbone 정보 추출
        backbone_info = self.config['model'].get('arch_name', 'unknown')
        if 'pretrained_path' in self.config['model']:
            # pretrained_path에서 모델 이름 추출 시도
            pretrained = self.config['model']['pretrained_path']
            if 'detr' in pretrained.lower():
                backbone_info = "detr"

        # 결과 딕셔너리 구성 (YOLO와 동일한 형식)
        results_dict = {
            'detection_metrics': {
                'map_50': float(detection_results['map_50']),
                'map': float(detection_results['map']),
                'map_75': float(detection_results['map_75']),
            },
            'detailed_statistics': {
                'total_statistics': {
                    'overall_precision': float(detailed_stats['total_statistics']['overall_precision']),
                    'overall_recall': float(detailed_stats['total_statistics']['overall_recall']),
                    'overall_f1': float(detailed_stats['total_statistics']['overall_f1']),
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
                'pretrained_path': self.config['model'].get('pretrained_path', 'unknown'),
            }
        }

        # 결과 저장 (공통 함수 사용)
        if self.output_dir:
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

        if self.output_dir:
            print("\n" + "="*70)
            print("EVALUATION COMPLETED!")
            print("="*70)
            print(f"Results directory: {self.output_dir}")
            print(f"  - evaluation_results.json: 전체 결과")
            print(f"  - summary_metrics.csv: 성능 요약표")
            print("="*70 + "\n")

        return results_dict
