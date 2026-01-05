"""Evaluation Runner"""
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
from tqdm import tqdm

import torch
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from ..registry import MODEL_REGISTRY, DATASET_REGISTRY
from ..utils.visualization import (
    save_detection_images, 
    plot_confusion_matrix, 
    plot_class_metrics,
    create_evaluation_report
)

def box_iou(boxes1, boxes2):
    """
    CXCYWH 형식의 박스 IoU 계산
    """
    from torchvision.ops import box_iou as tv_box_iou
    
    # CXCYWH -> XYXY 변환
    def cxcywh_to_xyxy(boxes):
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)
    
    boxes1_xyxy = cxcywh_to_xyxy(boxes1)
    boxes2_xyxy = cxcywh_to_xyxy(boxes2)
    
    return tv_box_iou(boxes1_xyxy, boxes2_xyxy)


class EvaluationRunner:
    """평가 실행 클래스 - 상세 분석 포함"""
    
    def __init__(self, config, checkpoint_path, split='test', output_dir=None):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.split = split
        
        # 출력 디렉토리 설정
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = Path(checkpoint_path).stem
            output_dir = f"results/evaluation_{checkpoint_name}_{split}_{timestamp}"
        
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
            
        elif model_name in ["YOLOv11", "YOLOv12"]:
            from ..data.transforms.yolo_transform import create_yolo_dataset
            
            dataset = create_yolo_dataset(
                dataset_meta, self.split, self.config['data']['image_size']
            )
            collate_fn = None
            
            ModelClass = MODEL_REGISTRY[model_name]
            model = ModelClass.load_from_checkpoint(
                self.checkpoint_path,
                model_size=self.config['model']['model_size'],
                num_classes=self.config['model']['num_labels'],
                lr=self.config['model']['learning_rate'],
            )
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
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
    
    def _evaluate_detection_metrics_detailed(self, model, dataloader) -> Tuple[Dict, Dict, List, List]:
        """
        상세한 Detection 메트릭 계산 (TP/FP/FN 분석 포함)
        
        Returns:
            detection_metrics: mAP 등 detection 메트릭
            detailed_stats: TP/FP/FN 상세 통계
            all_predictions: 모든 예측 결과
            all_targets: 모든 타겟 결과
        """
        device = next(model.parameters()).device
    
        # Detection metric 초기화
        map_metric = MeanAveragePrecision(
            box_format="cxcywh", 
            iou_type="bbox",
            class_metrics=True
        )
        
        # 상세 통계를 위한 변수들
        total_predictions = 0
        total_ground_truths = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        # 클래스별 통계
        num_classes = self.config['model']['num_labels']
        class_tp = [0] * num_classes
        class_fp = [0] * num_classes
        class_fn = [0] * num_classes
        class_predictions = [0] * num_classes
        class_ground_truths = [0] * num_classes
        
        all_predictions = []
        all_targets = []
        
        print(f"\n{self.split.upper()} 데이터 평가 중...")
    
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
                if self.config.get('debug') and batch_idx >= 5:
                    break
                
                pixel_values = batch["pixel_values"].to(device)
                pixel_mask = batch.get("pixel_mask")
                if pixel_mask is not None:
                    pixel_mask = pixel_mask.to(device)
                
                labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
                
                # 모델 추론
                outputs = model(pixel_values, pixel_mask)
                
                # Detection metric용 데이터 준비
                probs = outputs.logits.softmax(-1)[..., :-1]
                scores, pred_labels = probs.max(-1)
                pred_boxes = outputs.pred_boxes
                
                preds = []
                targets = []
                
                for i in range(pred_boxes.shape[0]):
                    # Prediction 필터링
                    keep = scores[i] > self.score_threshold
                    
                    pred_boxes_filtered = pred_boxes[i][keep].detach().cpu()
                    pred_scores_filtered = scores[i][keep].detach().cpu()
                    pred_labels_filtered = pred_labels[i][keep].detach().cpu()
                    
                    preds.append({
                        "boxes": pred_boxes_filtered,
                        "scores": pred_scores_filtered,
                        "labels": pred_labels_filtered,
                    })
                    
                    # Ground truth
                    gt_boxes = labels[i]["boxes"].detach().cpu()
                    gt_labels = labels[i]["class_labels"].detach().cpu()
                    
                    targets.append({
                        "boxes": gt_boxes,
                        "labels": gt_labels,
                    })
                    
                    # 상세 TP/FP/FN 계산
                    num_preds = len(pred_boxes_filtered)
                    num_gts = len(gt_boxes)
                    
                    # 클래스별 GT 카운트
                    for gt_label in gt_labels:
                        class_ground_truths[gt_label.item()] += 1
                    
                    # 클래스별 예측 카운트
                    for pred_label in pred_labels_filtered:
                        class_predictions[pred_label.item()] += 1
                    
                    if num_preds > 0 and num_gts > 0:
                        # IoU 계산
                        iou_matrix = box_iou(pred_boxes_filtered, gt_boxes)
                        
                        # 각 예측에 대해 가장 높은 IoU를 가진 GT 찾기
                        max_ious, matched_gt_indices = iou_matrix.max(dim=1)
                        
                        # TP: IoU >= threshold인 예측
                        tp_mask = max_ious >= self.iou_threshold
                        tp = tp_mask.sum().item()
                        
                        # FP: IoU < threshold인 예측
                        fp = (~tp_mask).sum().item()
                        
                        # FN: 매칭되지 않은 GT
                        matched_gt_mask = torch.zeros(num_gts, dtype=torch.bool)
                        if tp > 0:
                            matched_gt_mask[matched_gt_indices[tp_mask]] = True
                        fn = (~matched_gt_mask).sum().item()
                        
                        # 클래스별 TP/FP/FN 계산
                        for j, (pred_label, is_tp) in enumerate(zip(pred_labels_filtered, tp_mask)):
                            if is_tp:
                                class_tp[pred_label.item()] += 1
                            else:
                                class_fp[pred_label.item()] += 1
                        
                        # 매칭되지 않은 GT는 해당 클래스의 FN
                        for j, gt_label in enumerate(gt_labels):
                            if not matched_gt_mask[j]:
                                class_fn[gt_label.item()] += 1
                        
                    elif num_preds > 0 and num_gts == 0:
                        # GT가 없는데 예측이 있음 -> 모두 FP
                        tp = 0
                        fp = num_preds
                        fn = 0
                        
                        for pred_label in pred_labels_filtered:
                            class_fp[pred_label.item()] += 1
                        
                    elif num_preds == 0 and num_gts > 0:
                        # 예측이 없는데 GT가 있음 -> 모두 FN
                        tp = 0
                        fp = 0
                        fn = num_gts
                        
                        for gt_label in gt_labels:
                            class_fn[gt_label.item()] += 1
                        
                    else:
                        tp = fp = fn = 0
                    
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn
                    total_predictions += num_preds
                    total_ground_truths += num_gts
                
                all_predictions.extend(preds)
                all_targets.extend(targets)
                
                # Detection metric 업데이트
                if preds:
                    map_metric.update(preds, targets)
    
        # Detection 지표 계산
        detection_metrics = map_metric.compute()
        
        # 전체 Precision, Recall, F1 계산
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        # 클래스별 Precision, Recall, F1 계산
        class_precision = []
        class_recall = []
        class_f1 = []
        
        for i in range(num_classes):
            prec = class_tp[i] / (class_tp[i] + class_fp[i]) if (class_tp[i] + class_fp[i]) > 0 else 0
            rec = class_tp[i] / (class_tp[i] + class_fn[i]) if (class_tp[i] + class_fn[i]) > 0 else 0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            
            class_precision.append(prec)
            class_recall.append(rec)
            class_f1.append(f1)
        
        # 상세 통계 딕셔너리
        detailed_stats = {
            "total_statistics": {
                "total_ground_truths": total_ground_truths,
                "total_predictions": total_predictions,
                "total_tp": total_tp,
                "total_fp": total_fp,
                "total_fn": total_fn,
                "overall_precision": overall_precision,
                "overall_recall": overall_recall,
                "overall_f1": overall_f1
            },
            "class_statistics": {
                "class_tp": class_tp,
                "class_fp": class_fp,
                "class_fn": class_fn,
                "class_predictions": class_predictions,
                "class_ground_truths": class_ground_truths,
                "class_precision": class_precision,
                "class_recall": class_recall,
                "class_f1": class_f1
            }
        }
        
        return detection_metrics, detailed_stats, all_predictions, all_targets
    
    def _evaluate_classification_metrics(self, all_predictions, all_targets):
        """Classification 메트릭 계산"""
        all_true_labels = []
        all_pred_labels = []
        
        # Detection 결과를 Classification으로 변환
        for pred, target in zip(all_predictions, all_targets):
            gt_labels = target["labels"]
            pred_labels = pred["labels"]
        
            if len(gt_labels) > 0:
                for gt_label in gt_labels:
                    all_true_labels.append(gt_label.item())
                
                if len(pred_labels) > 0:
                    # 간단한 매칭: 예측 수만큼 GT에서 가져오기
                    for i, pred_label in enumerate(pred_labels):
                        if i < len(gt_labels):
                            all_pred_labels.append(pred_label.item())
                        else:
                            # 예측이 GT보다 많으면 마지막 GT 라벨 사용
                            all_pred_labels.append(gt_labels[-1].item())
                else:
                    # 예측이 없으면 "no detection" 클래스로 처리 (여기서는 -1)
                    for _ in gt_labels:
                        all_pred_labels.append(-1)
    
        if len(all_true_labels) == 0 or len(all_pred_labels) == 0:
            return {}
        
        # 길이 맞추기
        min_len = min(len(all_true_labels), len(all_pred_labels))
        all_true_labels = all_true_labels[:min_len]
        all_pred_labels = all_pred_labels[:min_len]
        
        # -1 (no detection) 제거
        valid_indices = [i for i, pred in enumerate(all_pred_labels) if pred >= 0]
        all_true_labels = [all_true_labels[i] for i in valid_indices]
        all_pred_labels = [all_pred_labels[i] for i in valid_indices]
        
        if len(all_true_labels) == 0:
            return {}
        
        # 메트릭 계산
        num_classes = self.config['model']['num_labels']
        class_names = self.config['data']['class_names']
        
        results = {
            "accuracy": accuracy_score(all_true_labels, all_pred_labels),
            "precision_macro": precision_score(all_true_labels, all_pred_labels, average='macro', zero_division=0),
            "recall_macro": recall_score(all_true_labels, all_pred_labels, average='macro', zero_division=0),
            "f1_macro": f1_score(all_true_labels, all_pred_labels, average='macro', zero_division=0),
            "confusion_matrix": confusion_matrix(all_true_labels, all_pred_labels, labels=list(range(num_classes))).tolist(),
            "classification_report": classification_report(
                all_true_labels, all_pred_labels,
                target_names=class_names,
                labels=list(range(num_classes)),
                output_dict=True,
                zero_division=0
            )
        }
        
        return results
    
    def _save_results_with_visualization(self, detection_metrics, detailed_stats, classification_results):
        """결과 저장 및 시각화"""
        
        # 텐서를 적절한 형태로 변환하는 함수
        def convert_tensor_to_serializable(obj):
            if torch.is_tensor(obj):
                if obj.numel() == 1:  # 스칼라 텐서
                    return obj.item()
                else:  # 배열 텐서
                    return obj.tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_tensor_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensor_to_serializable(item) for item in obj]
            else:
                return obj
        
        # 1. JSON 결과 저장
        results = {
            "evaluation_info": {
                "checkpoint": str(self.checkpoint_path),
                "split": self.split,
                "timestamp": datetime.now().isoformat(),
                "score_threshold": self.score_threshold,
                "iou_threshold": self.iou_threshold,
                "config": self.config
            },
            "detection_metrics": convert_tensor_to_serializable(detection_metrics),
            "detailed_statistics": detailed_stats,
            "classification_metrics": classification_results
        }
        
        results_file = self.output_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
            # 2. 전체 성능 요약 표 생성
        summary_data = {
            'Metric': [
                'mAP@0.5', 'mAP@0.5:0.95', 'mAP@0.75',
                'Overall Precision', 'Overall Recall', 'Overall F1-Score',
                'Total Ground Truths', 'Total Predictions', 'True Positives', 'False Positives', 'False Negatives'
            ],
            'Value': [
                f"{detection_metrics['map_50']:.4f}",
                f"{detection_metrics['map']:.4f}",
                f"{detection_metrics['map_75']:.4f}",
                f"{detailed_stats['total_statistics']['overall_precision']:.4f}",
                f"{detailed_stats['total_statistics']['overall_recall']:.4f}",
                f"{detailed_stats['total_statistics']['overall_f1']:.4f}",
                detailed_stats['total_statistics']['total_ground_truths'],
                detailed_stats['total_statistics']['total_predictions'],
                detailed_stats['total_statistics']['total_tp'],
                detailed_stats['total_statistics']['total_fp'],
                detailed_stats['total_statistics']['total_fn']
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_csv = self.output_dir / "summary_metrics.csv"
        summary_df.to_csv(summary_csv, index=False)
        
        # 3. 클래스별 상세 성능 표 생성
        class_names = self.config['data']['class_names']
        class_stats = detailed_stats['class_statistics']
        
        class_data = []
        for i, class_name in enumerate(class_names):
            if i < len(class_stats['class_tp']):
                # Detection 메트릭에서 클래스별 mAP 추출
                class_map = "N/A"
                if 'map_per_class' in detection_metrics:
                    map_per_class = convert_tensor_to_serializable(detection_metrics['map_per_class'])
                    if isinstance(map_per_class, list) and i < len(map_per_class):
                        class_map = f"{map_per_class[i]:.4f}"
                    elif not isinstance(map_per_class, list):
                        class_map = f"{map_per_class:.4f}" if i == 0 else "N/A"
                
                class_data.append({
                    'Class': class_name,
                    'Ground Truth Count': class_stats['class_ground_truths'][i],
                    'Prediction Count': class_stats['class_predictions'][i],
                    'True Positives': class_stats['class_tp'][i],
                    'False Positives': class_stats['class_fp'][i],
                    'False Negatives': class_stats['class_fn'][i],
                    'Precision': f"{class_stats['class_precision'][i]:.4f}",
                    'Recall': f"{class_stats['class_recall'][i]:.4f}",
                    'F1-Score': f"{class_stats['class_f1'][i]:.4f}",
                    'mAP@0.5:0.95': class_map
                })
        
        class_df = pd.DataFrame(class_data)
        class_csv = self.output_dir / "class_metrics.csv"
        class_df.to_csv(class_csv, index=False)
        
        # 4. Classification 결과 표 (있는 경우)
        if classification_results and 'classification_report' in classification_results:
            clf_report = classification_results['classification_report']
            clf_data = []
            
            for class_name in class_names:
                if class_name in clf_report:
                    clf_data.append({
                        'Class': class_name,
                        'Precision': f"{clf_report[class_name]['precision']:.4f}",
                        'Recall': f"{clf_report[class_name]['recall']:.4f}",
                        'F1-Score': f"{clf_report[class_name]['f1-score']:.4f}",
                        'Support': clf_report[class_name]['support']
                    })
            
            # 전체 평균 추가
            if 'macro avg' in clf_report:
                clf_data.append({
                    'Class': 'Macro Average',
                    'Precision': f"{clf_report['macro avg']['precision']:.4f}",
                    'Recall': f"{clf_report['macro avg']['recall']:.4f}",
                    'F1-Score': f"{clf_report['macro avg']['f1-score']:.4f}",
                    'Support': clf_report['macro avg']['support']
                })
            
            clf_df = pd.DataFrame(clf_data)
            clf_csv = self.output_dir / "classification_metrics.csv"
            clf_df.to_csv(clf_csv, index=False)
        
        # 5. 시각화 (Confusion Matrix)
        if classification_results and 'confusion_matrix' in classification_results:
            cm = np.array(classification_results['confusion_matrix'])
            class_names = self.config['data']['class_names']
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names
            )
            plt.title(f'Confusion Matrix - {self.split.upper()} Set')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            cm_file = self.output_dir / "confusion_matrix.png"
            plt.savefig(cm_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Confusion matrix saved to: {cm_file}")
        
        # 결과 파일 출력
        print(f"\n📁 Results saved to: {self.output_dir}")
        print(f"  📄 JSON: {results_file}")
        print(f"  📊 Summary CSV: {summary_csv}")
        print(f"  📈 Class CSV: {class_csv}")
        if 'clf_csv' in locals():
            print(f"  🎯 Classification CSV: {clf_csv}")
        
        return results
    
    def _print_detailed_results(self, detection_metrics, detailed_stats, classification_results):
        """상세 결과 출력"""
        print("\n" + "="*70)
        print("DETECTION 평가 결과")
        print("="*70)
        
        stats = detailed_stats["total_statistics"]
        class_stats = detailed_stats["class_statistics"]
        
        print(f"\n[전체 통계]")
        print(f"  총 Ground Truth 객체 수: {stats['total_ground_truths']}")
        print(f"  총 예측된 객체 수 (threshold > {self.score_threshold}): {stats['total_predictions']}")
        
        print(f"\n[Detection 성능 (IoU >= {self.iou_threshold})]")
        if stats['total_predictions'] > 0:
            print(f"  True Positives (TP):  {stats['total_tp']:4d} ({stats['total_tp']/stats['total_predictions']*100:.1f}%)")
            print(f"  False Positives (FP): {stats['total_fp']:4d} ({stats['total_fp']/stats['total_predictions']*100:.1f}%)")
        else:
            print(f"  True Positives (TP):  {stats['total_tp']:4d}")
            print(f"  False Positives (FP): {stats['total_fp']:4d}")
        
        if stats['total_ground_truths'] > 0:
            print(f"  False Negatives (FN): {stats['total_fn']:4d} ({stats['total_fn']/stats['total_ground_truths']*100:.1f}%)")
        else:
            print(f"  False Negatives (FN): {stats['total_fn']:4d}")
        
        print(f"\n[전체 Precision / Recall / F1]")
        print(f"  Precision: {stats['overall_precision']:.4f} ({stats['overall_precision']*100:.2f}%)")
        print(f"  Recall:    {stats['overall_recall']:.4f} ({stats['overall_recall']*100:.2f}%)")
        print(f"  F1-Score:  {stats['overall_f1']:.4f} ({stats['overall_f1']*100:.2f}%)")
        
        print(f"\n[mAP 지표]")
        print(f"  mAP (IoU=0.50:0.95): {detection_metrics['map']:.4f}")
        print(f"  mAP@0.50: {detection_metrics['map_50']:.4f}")
        print(f"  mAP@0.75: {detection_metrics['map_75']:.4f}")
        
        # 클래스별 상세 통계
        class_names = self.config['data']['class_names']
        print(f"\n[클래스별 상세 통계]")
        for i, class_name in enumerate(class_names):
            if i < len(class_stats['class_tp']):
                print(f"\n  {class_name}:")
                print(f"    GT 객체 수: {class_stats['class_ground_truths'][i]}")
                print(f"    예측 객체 수: {class_stats['class_predictions'][i]}")
                print(f"    TP: {class_stats['class_tp'][i]}, FP: {class_stats['class_fp'][i]}, FN: {class_stats['class_fn'][i]}")
                print(f"    Precision: {class_stats['class_precision'][i]:.4f}")
                print(f"    Recall: {class_stats['class_recall'][i]:.4f}")
                print(f"    F1-Score: {class_stats['class_f1'][i]:.4f}")
        
        # Per-class mAP
        if 'map_per_class' in detection_metrics:
            print(f"\n[클래스별 mAP (IoU=0.50:0.95)]")
            map_per_class = detection_metrics['map_per_class']
            
            if map_per_class.ndim == 0:
                if len(class_names) > 0:
                    print(f"  {class_names[0]}: {map_per_class:.4f}")
            else:
                for i, ap in enumerate(map_per_class):
                    if i < len(class_names):
                        print(f"  {class_names[i]}: {ap:.4f}")
        
        # Classification 결과
        if classification_results:
            print("\n" + "="*70)
            print("CLASSIFICATION 평가 결과")
            print("="*70)
            
            print(f"\n[전체 분류 성능]")
            print(f"  Accuracy: {classification_results['accuracy']:.4f}")
            print(f"  Precision (macro): {classification_results['precision_macro']:.4f}")
            print(f"  Recall (macro): {classification_results['recall_macro']:.4f}")
            print(f"  F1-Score (macro): {classification_results['f1_macro']:.4f}")
            
            if 'classification_report' in classification_results:
                print(f"\n[클래스별 분류 성능]")
                report = classification_results['classification_report']
                for class_name in class_names:
                    if class_name in report:
                        class_report = report[class_name]
                        print(f"  {class_name}:")
                        print(f"    Precision: {class_report['precision']:.4f}")
                        print(f"    Recall: {class_report['recall']:.4f}")
                        print(f"    F1-Score: {class_report['f1-score']:.4f}")
                        print(f"    Support: {class_report['support']}")
    
    def _save_inference_images(self, model, dataset, predictions, targets, max_images=None):
        """인퍼런스 결과 이미지 저장"""
        from PIL import Image, ImageDraw, ImageFont
        import torchvision.transforms as transforms
        
        # 인퍼런스 이미지 저장 디렉토리 생성
        inference_dir = self.output_dir / "inference_images"
        inference_dir.mkdir(exist_ok=True)
        
        device = next(model.parameters()).device
        model.eval()
        
        saved_count = 0
        class_names = self.config['data']['class_names']
        
        # max_images 처리: None이면 전체 데이터셋 크기 사용
        if max_images is None:
            max_images = len(dataset)
        
        # 실제 처리할 이미지 수
        total_images = min(len(dataset), max_images)
        
        # 색상 정의 (클래스별)
        colors = {
            0: (255, 0, 0),    # fully-ripe: 빨강
            1: (255, 165, 0),  # semi-ripe: 주황
            2: (0, 255, 0),    # unripe: 초록
        }
        
        print(f"Saving inference images to: {inference_dir}")
        print(f"Total images to process: {total_images}")
        
        with torch.no_grad():
            for idx in range(total_images):
                # 데이터셋에서 이미지와 타겟 가져오기
                sample = dataset[idx]
                
                # 이미지 처리
                if isinstance(sample, dict):
                    image_tensor = sample['pixel_values']
                    target = sample['labels'] if 'labels' in sample else None
                else:
                    image_tensor, target = sample
                
                # 배치 차원 추가
                if image_tensor.dim() == 3:
                    image_tensor = image_tensor.unsqueeze(0)
                
                # 모델 예측
                image_tensor = image_tensor.to(device)
                outputs = model(image_tensor)
                
                # 예측 결과 처리
                probs = outputs.logits.softmax(-1)[0, :, :-1]
                scores, pred_labels = probs.max(-1)
                pred_boxes = outputs.pred_boxes[0]
                
                # 임계값 적용
                keep = scores > self.score_threshold
                pred_boxes_filtered = pred_boxes[keep]
                pred_scores_filtered = scores[keep]
                pred_labels_filtered = pred_labels[keep]
                
                # 이미지를 PIL로 변환
                # 정규화 해제 (ImageNet 기준)
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                
                image_denorm = image_tensor[0].cpu() * std + mean
                image_denorm = torch.clamp(image_denorm, 0, 1)
                
                # PIL 이미지로 변환
                to_pil = transforms.ToPILImage()
                pil_image = to_pil(image_denorm)
                
                # 이미지 크기
                img_w, img_h = pil_image.size
                
                # 그리기 준비
                draw = ImageDraw.Draw(pil_image)
                
                try:
                    # 폰트 설정 (시스템에 따라 다를 수 있음)
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
                except:
                    font = ImageFont.load_default()
                
                # Ground Truth 그리기 (파란색 테두리)
                if target is not None:
                    gt_boxes = target.get('boxes', [])
                    gt_labels = target.get('class_labels', [])
                    
                    if len(gt_boxes) > 0:
                        for box, label in zip(gt_boxes, gt_labels):
                            # CXCYWH를 XYXY로 변환
                            cx, cy, w, h = box
                            x1 = (cx - w/2) * img_w
                            y1 = (cy - h/2) * img_h
                            x2 = (cx + w/2) * img_w
                            y2 = (cy + h/2) * img_h
                            
                            # 박스 그리기
                            draw.rectangle([x1, y1, x2, y2], outline="blue", width=3)
                            
                            # 라벨 텍스트
                            if label.item() < len(class_names):
                                text = f"GT: {class_names[label.item()]}"
                                draw.text((x1, y1-20), text, fill="blue", font=font)
                
                # Predictions 그리기 (클래스별 색상)
                for box, label, score in zip(pred_boxes_filtered, pred_labels_filtered, pred_scores_filtered):
                    # CXCYWH를 XYXY로 변환
                    cx, cy, w, h = box
                    x1 = (cx - w/2) * img_w
                    y1 = (cy - h/2) * img_h
                    x2 = (cx + w/2) * img_w
                    y2 = (cy + h/2) * img_h
                    
                    # 클래스별 색상
                    color = colors.get(label.item(), (255, 255, 255))
                    
                    # 박스 그리기
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                    
                    # 라벨과 스코어 텍스트
                    if label.item() < len(class_names):
                        text = f"{class_names[label.item()]}: {score:.2f}"
                        
                        # 텍스트 배경
                        bbox = draw.textbbox((x1, y2+5), text, font=font)
                        draw.rectangle(bbox, fill=color)
                        draw.text((x1, y2+5), text, fill="white", font=font)
                
                # 이미지 저장
                filename = f"inference_{idx:04d}_{self.split}.jpg"
                save_path = inference_dir / filename
                pil_image.save(save_path, quality=95)
                
                saved_count += 1
                
                if saved_count % 10 == 0:
                    print(f"  Saved {saved_count}/{total_images} images...")
        
        print(f"✅ Saved {saved_count} inference images to: {inference_dir}")
        
        # 범례 이미지 생성
        self._create_legend_image(inference_dir, class_names, colors)

    def _create_legend_image(self, inference_dir, class_names, colors):
        """범례 이미지 생성"""
        from PIL import Image, ImageDraw, ImageFont
        
        # 범례 이미지 크기
        legend_width = 400
        legend_height = 200
        
        # 범례 이미지 생성
        legend_img = Image.new('RGB', (legend_width, legend_height), 'white')
        draw = ImageDraw.Draw(legend_img)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()
        
        # 제목
        draw.text((20, 20), "Detection Legend", fill="black", font=title_font)
        
        # Ground Truth
        draw.rectangle([20, 60, 40, 80], outline="blue", width=3)
        draw.text((50, 65), "Ground Truth", fill="blue", font=font)
        
        # 각 클래스
        y_offset = 90
        for i, class_name in enumerate(class_names):
            color = colors.get(i, (255, 255, 255))
            draw.rectangle([20, y_offset, 40, y_offset+20], outline=color, width=3)
            draw.text((50, y_offset+5), f"Predicted: {class_name}", fill=color, font=font)
            y_offset += 30
        
        # 범례 저장
        legend_path = inference_dir / "legend.jpg"
        legend_img.save(legend_path, quality=95)
        print(f"📋 Legend saved to: {legend_path}")

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
        
        # 1. 모델 및 데이터 로드
        print("\nLoading model and dataset...")
        model, dataset, collate_fn = self._load_model_and_data()
        print(f"Dataset samples: {len(dataset)}")
        
        # 2. DataLoader 생성
        dataloader = self._create_dataloader(dataset, collate_fn)
        print(f"Evaluation batches: {len(dataloader)}")
        
        # 3. 상세 Detection 메트릭 계산
        print("\nCalculating detailed detection metrics...")
        detection_results, detailed_stats, predictions, targets = self._evaluate_detection_metrics_detailed(model, dataloader)
        
        # 4. Classification 메트릭 계산
        print("Calculating classification metrics...")
        classification_results = self._evaluate_classification_metrics(predictions, targets)
        
        # 5. 인퍼런스 이미지 저장 (새로 추가)
        print("Saving inference images...")
        self._save_inference_images(model, dataset, predictions, targets, max_images=None)
        
        # 6. 결과 저장 및 시각화
        print("\nSaving results and creating visualizations...")
        final_results = self._save_results_with_visualization(detection_results, detailed_stats, classification_results)
        
        # 7. 상세 결과 출력
        self._print_detailed_results(detection_results, detailed_stats, classification_results)
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETED!")
        print("="*70)
        print(f"Results directory: {self.output_dir}")
        print(f"  - evaluation_results.json: 전체 결과")
        print(f"  - summary_metrics.csv: 성능 요약표")
        print(f"  - class_metrics.csv: 클래스별 성능표")
        if classification_results:
            print(f"  - classification_metrics.csv: 분류 성능표")
            print(f"  - confusion_matrix.png: 혼동 행렬")
        print(f"  - inference_images/: 인퍼런스 결과 이미지들")
        print("="*70 + "\n")
        
        return final_results