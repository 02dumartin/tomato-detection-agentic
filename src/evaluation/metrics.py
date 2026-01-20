"""평가 메트릭 계산 함수들"""
import torch
from typing import Dict, List, Tuple
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


# =============================================================================
# Detection 메트릭
# =============================================================================

def box_iou(boxes1, boxes2):
    """
    CXCYWH 형식의 박스 IoU 계산
    
    Args:
        boxes1: [N, 4] 텐서 (cx, cy, w, h)
        boxes2: [M, 4] 텐서 (cx, cy, w, h)
    
    Returns:
        [N, M] IoU 행렬
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


def evaluate_detection_metrics(
    model, 
    dataloader, 
    config, 
    score_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    split: str = 'val'
) -> Tuple[Dict, Dict, List, List]:
    """
    상세한 Detection 메트릭 계산 (TP/FP/FN 분석 포함)
    
    Args:
        model: 평가할 모델
        dataloader: 평가용 DataLoader
        config: 설정 딕셔너리
        score_threshold: 예측 confidence threshold
        iou_threshold: IoU threshold for TP/FP 계산
        split: 데이터셋 split 이름 (출력용)
    
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
    
    # CA-mAP metric 초기화 (class-agnostic)
    ca_map_metric = MeanAveragePrecision(
        box_format="cxcywh", 
        iou_type="bbox",
        class_metrics=False  # 클래스 구분 없이 계산
    )
    
    def to_class_agnostic(preds, targets):
        """예측과 타겟을 class-agnostic으로 변환"""
        ca_preds = []
        ca_targets = []
        for p, t in zip(preds, targets):
            ca_preds.append({
                "boxes": p["boxes"],
                "scores": p["scores"],
                "labels": torch.zeros_like(p["labels"]),  # 모든 레이블을 0으로
            })
            ca_targets.append({
                "boxes": t["boxes"],
                "labels": torch.zeros_like(t["labels"]),  # 모든 레이블을 0으로
            })
        return ca_preds, ca_targets
    
    # 상세 통계를 위한 변수들
    total_predictions = 0
    total_ground_truths = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # 클래스별 통계
    num_classes = config['model']['num_labels']
    class_tp = [0] * num_classes
    class_fp = [0] * num_classes
    class_fn = [0] * num_classes
    class_predictions = [0] * num_classes
    class_ground_truths = [0] * num_classes
    
    all_predictions = []
    all_targets = []
    
    print(f"\n{split.upper()} 데이터 평가 중...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if config.get('debug') and batch_idx >= 5:
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
                keep = scores[i] > score_threshold
                
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
                    
                    # 매칭된 GT의 레이블 가져오기
                    matched_gt_labels = gt_labels[matched_gt_indices]
                    
                    # TP: IoU >= threshold AND 클래스 레이블이 일치하는 예측
                    class_match = pred_labels_filtered == matched_gt_labels
                    tp_mask = (max_ious >= iou_threshold) & class_match
                    tp = tp_mask.sum().item()
                    
                    # FP: IoU >= threshold이지만 클래스가 틀린 예측 OR IoU < threshold인 예측
                    # IoU는 충분하지만 클래스가 틀린 경우
                    class_mismatch = (max_ious >= iou_threshold) & (~class_match)
                    # IoU가 부족한 경우
                    low_iou = max_ious < iou_threshold
                    fp = (class_mismatch | low_iou).sum().item()
                    
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
                # CA-mAP 계산을 위한 class-agnostic 변환
                ca_preds, ca_targets = to_class_agnostic(preds, targets)
                ca_map_metric.update(ca_preds, ca_targets)
    
    # Detection 지표 계산
    detection_metrics = map_metric.compute()
    
    # CA-mAP 계산
    ca_metrics = ca_map_metric.compute()
    
    # CA-mAP 결과를 detection_metrics에 추가
    detection_metrics["ca_map"] = float(ca_metrics.get("map", 0.0))
    detection_metrics["ca_map_50"] = float(ca_metrics.get("map_50", ca_metrics.get("map", 0.0)))
    detection_metrics["ca_map_75"] = float(ca_metrics.get("map_75", ca_metrics.get("map", 0.0)))
    
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


# =============================================================================
# Classification 메트릭
# =============================================================================

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


# =============================================================================
# 모델 복잡도
# =============================================================================

def calculate_model_complexity(model, config: Dict) -> Dict:
    """
    모델 복잡도 계산 (Parameters, Size, GFLOPs)
    
    Args:
        model: 평가할 모델
        config: 설정 딕셔너리
    
    Returns:
        complexity_metrics: 복잡도 메트릭 딕셔너리
    """
    # 1. Parameters 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 2. Model Size 계산 (MB)
    model_size_mb = sum(p.numel() * 4 for p in model.parameters()) / (1024**2)  # float32 = 4 bytes
    
    # 3. GFLOPs 계산 (더미 입력 사용)
    model.eval()
    device = next(model.parameters()).device
    
    # 입력 크기 가져오기
    if 'image_size' in config.get('data', {}):
        img_size = config['data']['image_size']
    elif 'imgsz' in config.get('data', {}):
        img_size = config['data']['imgsz']
    else:
        img_size = 640  # 기본값
    
    # 더미 입력 생성 (배치 크기 1)
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
    
    gflops = None
    flops_formatted = None
    
    try:
        # thop을 사용한 GFLOPs 계산
        from thop import profile, clever_format
        
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        gflops = flops / 1e9  # GFLOPs로 변환
        
        # 또는 clever_format 사용
        flops_formatted, params_formatted = clever_format([flops, params], "%.3f")
    except ImportError:
        print("  thop not installed. GFLOPs calculation skipped.")
        print("   Install with: pip install thop")
    except Exception as e:
        print(f"  Could not calculate GFLOPs: {e}")
    
    return {
        'total_params': int(total_params),
        'trainable_params': int(trainable_params),
        'params_m': total_params / 1e6,
        'model_size_mb': model_size_mb,
        'gflops': gflops,
        'gflops_formatted': flops_formatted
    }

