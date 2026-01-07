"""Classification 메트릭 계산"""
from typing import Dict, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


def evaluate_classification_metrics(
    all_predictions: List[Dict],
    all_targets: List[Dict],
    config: Dict
) -> Dict:
    """
    Classification 메트릭 계산
    
    Args:
        all_predictions: 모든 예측 결과 리스트
        all_targets: 모든 타겟 결과 리스트
        config: 설정 딕셔너리
    
    Returns:
        classification_results: Classification 메트릭 딕셔너리
    """
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
    num_classes = config['model']['num_labels']
    class_names = config['data']['class_names']
    
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

