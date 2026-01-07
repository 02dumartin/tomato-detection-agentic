"""Detection 메트릭 계산"""
import torch
from typing import Dict, List, Tuple
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision


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
                    
                    # TP: IoU >= threshold인 예측
                    tp_mask = max_ious >= iou_threshold
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

