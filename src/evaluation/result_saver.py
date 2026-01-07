"""평가 결과 저장 및 시각화"""
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


def convert_tensor_to_serializable(obj):
    """텐서를 적절한 형태로 변환하는 함수"""
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


def save_evaluation_results(
    detection_metrics: Dict,
    detailed_stats: Dict,
    classification_results: Optional[Dict],
    complexity_metrics: Optional[Dict],
    output_dir: Path,
    config: Dict,
    checkpoint_path: str,
    split: str,
    score_threshold: float,
    iou_threshold: float
) -> Dict:
    """
    평가 결과 저장 및 시각화
    
    Args:
        detection_metrics: Detection 메트릭
        detailed_stats: 상세 통계
        classification_results: Classification 메트릭 (선택적)
        complexity_metrics: 모델 복잡도 메트릭 (선택적)
        output_dir: 출력 디렉토리
        config: 설정 딕셔너리
        checkpoint_path: 체크포인트 경로
        split: 데이터셋 split
        score_threshold: Score threshold
        iou_threshold: IoU threshold
    
    Returns:
        results: 저장된 결과 딕셔너리
    """
    # 1. JSON 결과 저장
    results = {
        "evaluation_info": {
            "checkpoint": str(checkpoint_path),
            "split": split,
            "timestamp": datetime.now().isoformat(),
            "score_threshold": score_threshold,
            "iou_threshold": iou_threshold,
            "config": config
        },
        "detection_metrics": convert_tensor_to_serializable(detection_metrics),
        "detailed_statistics": detailed_stats,
        "classification_metrics": classification_results if classification_results else {},
        "model_complexity": complexity_metrics if complexity_metrics else {}
    }
    
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # 2. 클래스별 AP@0.5:0.95 추출
    class_names = config['data']['class_names']
    class_ap_columns = {}
    if 'map_per_class' in detection_metrics:
        map_per_class = convert_tensor_to_serializable(detection_metrics['map_per_class'])
        if isinstance(map_per_class, list):
            for i, class_name in enumerate(class_names):
                if i < len(map_per_class):
                    class_ap_columns[f'{class_name}_AP@0.5:0.95'] = f"{map_per_class[i]:.4f}"
                else:
                    class_ap_columns[f'{class_name}_AP@0.5:0.95'] = "N/A"
        elif map_per_class is not None:
            # 단일 값인 경우 첫 번째 클래스에만 할당
            if len(class_names) > 0:
                class_ap_columns[f'{class_names[0]}_AP@0.5:0.95'] = f"{map_per_class:.4f}"
                for class_name in class_names[1:]:
                    class_ap_columns[f'{class_name}_AP@0.5:0.95'] = "N/A"
    
    # 3. 통합 CSV 생성: mAP@0.5, 클래스별 AP@0.5:0.95, mAP@0.5:0.95, mAP@0.75, Precision, Recall, Parameter(M), GFLOPs
    metric_names = ['mAP@0.5'] + list(class_ap_columns.keys()) + ['mAP@0.5:0.95', 'mAP@0.75', 'Precision', 'Recall', 'Parameter(M)', 'GFLOPs']
    metric_values = [
        f"{detection_metrics['map_50']:.4f}",
    ] + list(class_ap_columns.values()) + [
        f"{detection_metrics['map']:.4f}",
        f"{detection_metrics['map_75']:.4f}",
        f"{detailed_stats['total_statistics']['overall_precision']:.4f}",
        f"{detailed_stats['total_statistics']['overall_recall']:.4f}",
        f"{complexity_metrics['params_m']:.2f}" if complexity_metrics and complexity_metrics.get('params_m') else "N/A",
        f"{complexity_metrics['gflops']:.2f}" if complexity_metrics and complexity_metrics.get('gflops') is not None else "N/A",
    ]
    
    summary_data = {
        'Metric': metric_names,
        'Value': metric_values
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv = output_dir / "summary_metrics.csv"
    summary_df.to_csv(summary_csv, index=False)
    
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
        clf_csv = output_dir / "classification_metrics.csv"
        clf_df.to_csv(clf_csv, index=False)
    
    # 5. 시각화 (Confusion Matrix)
    if classification_results and 'confusion_matrix' in classification_results:
        cm = np.array(classification_results['confusion_matrix'])
        class_names = config['data']['class_names']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title(f'Confusion Matrix - {split.upper()} Set')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        cm_file = output_dir / "confusion_matrix.png"
        plt.savefig(cm_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Confusion matrix saved to: {cm_file}")
    
    # 결과 파일 출력
    print(f"\n Results saved to: {output_dir}")
    print(f"   JSON: {results_file}")
    print(f"   Summary CSV: {summary_csv}")
    if classification_results and 'classification_report' in classification_results:
        print(f"   Classification CSV: {clf_csv}")
    
    return results


def print_evaluation_results(
    detection_metrics: Dict,
    detailed_stats: Dict,
    classification_results: Optional[Dict],
    config: Dict,
    split: str,
    score_threshold: float,
    iou_threshold: float
):
    """
    상세 결과 출력
    
    Args:
        detection_metrics: Detection 메트릭
        detailed_stats: 상세 통계
        classification_results: Classification 메트릭 (선택적)
        config: 설정 딕셔너리
        split: 데이터셋 split
        score_threshold: Score threshold
        iou_threshold: IoU threshold
    """
    print("\n" + "="*70)
    print("DETECTION 평가 결과")
    print("="*70)
    
    stats = detailed_stats["total_statistics"]
    class_stats = detailed_stats["class_statistics"]
    
    print(f"\n[전체 통계]")
    print(f"  총 Ground Truth 객체 수: {stats['total_ground_truths']}")
    print(f"  총 예측된 객체 수 (threshold > {score_threshold}): {stats['total_predictions']}")
    
    print(f"\n[Detection 성능 (IoU >= {iou_threshold})]")
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
    class_names = config['data']['class_names']
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


def save_summary_metrics(results_dict: Dict, output_dir: Path, config: Dict):
    """
    평가 결과 요약 메트릭 저장 (JSON 및 CSV)
    YOLO, DETR 등 모든 모델에서 공통으로 사용
    
    Args:
        results_dict: 평가 결과 딕셔너리 (YOLO/DETR 공통 형식)
        output_dir: 출력 디렉토리
        config: 설정 딕셔너리
    """
    # JSON 저장
    json_path = output_dir / "evaluation_results.json"
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f" Results saved to: {json_path}")

    # 통합 CSV 생성: mAP@0.5, 클래스별 AP@0.5:0.95, mAP@0.5:0.95, mAP@0.75, Precision, Recall, Parameter(M), GFLOPs
    per_class_ap = results_dict.get('detailed_statistics', {}).get('per_class_ap', {})
    class_names = config['data']['class_names']

    # 클래스별 AP@0.5:0.95 컬럼 생성
    class_ap_columns = {}
    if per_class_ap and 'ap_50_95' in per_class_ap:
        for class_name in class_names:
            ap_value = per_class_ap['ap_50_95'].get(class_name)
            if ap_value is not None and isinstance(ap_value, (int, float)):
                class_ap_columns[f'{class_name}_AP@0.5:0.95'] = f"{ap_value:.4f}"
            else:
                class_ap_columns[f'{class_name}_AP@0.5:0.95'] = "N/A"
    else:
        # 클래스별 AP가 없는 경우
        for class_name in class_names:
            class_ap_columns[f'{class_name}_AP@0.5:0.95'] = "N/A"

    metric_names = ['mAP@0.5'] + list(class_ap_columns.keys()) + ['mAP@0.5:0.95', 'mAP@0.75', 'Precision', 'Recall', 'Parameter(M)', 'GFLOPs']
    metric_values = [
        f"{results_dict['detection_metrics']['map_50']:.4f}",
    ] + list(class_ap_columns.values()) + [
        f"{results_dict['detection_metrics']['map']:.4f}",
        f"{results_dict['detection_metrics']['map_75']:.4f}" if results_dict['detection_metrics']['map_75'] else "N/A",
        f"{results_dict['detailed_statistics']['total_statistics']['overall_precision']:.4f}",
        f"{results_dict['detailed_statistics']['total_statistics']['overall_recall']:.4f}",
        f"{results_dict['model_complexity'].get('params_m', 0):.2f}" if results_dict.get('model_complexity') and results_dict['model_complexity'].get('params_m') else "N/A",
        f"{results_dict['model_complexity'].get('gflops', 0):.2f}" if results_dict.get('model_complexity') and results_dict['model_complexity'].get('gflops') else "N/A",
    ]

    summary_data = {
        'Metric': metric_names,
        'Value': metric_values
    }

    df = pd.DataFrame(summary_data)
    csv_path = output_dir / "summary_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f" Summary metrics saved to: {csv_path}")

