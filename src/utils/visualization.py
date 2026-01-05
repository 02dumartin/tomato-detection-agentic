# =============================================================================
# 시각화 함수 정의
# =============================================================================

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

COLORS = [
    [0.850, 0.325, 0.098],  # fully-ripe (빨강)
    [0.929, 0.694, 0.125],  # semi-ripe (노랑)
    [0.466, 0.674, 0.188],  # unripe (초록)
]

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, prob, boxes, id2label):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    
    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
        cl = p.argmax()
        color = COLORS[cl % len(COLORS)]
        ax.add_patch(plt.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            fill=False, color=color, linewidth=3
        ))
        text = f'{id2label[int(cl.item())]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin - 5, text, fontsize=12, color='white',
                bbox=dict(facecolor=color, alpha=0.8, pad=2))
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_predictions(image, outputs, id2label, threshold=0.7):
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold
    bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), image.size)
    plot_results(image, probas[keep], bboxes_scaled, id2label)


# 기존 코드 유지하고 아래 함수들 추가

def save_detection_images(predictions, targets, images, output_dir, class_names, id2label, max_images=50):
    """개별 예측 이미지 저장"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, (pred, target, image) in enumerate(zip(predictions, targets, images)):
        if i >= max_images:
            break
            
        # 이미지가 tensor인 경우 PIL로 변환
        if torch.is_tensor(image):
            image = transforms.ToPILImage()(image)
        
        # 예측 결과 그리기
        draw = ImageDraw.Draw(image)
        
        # Ground truth (초록색)
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]
        
        for box, label in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = box_cxcywh_to_xyxy_pil(box, image.size)
            draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
            draw.text((x1, y1-15), f"GT: {id2label[label.item()]}", fill="green")
        
        # Predictions (빨간색)
        pred_boxes = pred["boxes"]
        pred_labels = pred["labels"]
        pred_scores = pred["scores"]
        
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            x1, y1, x2, y2 = box_cxcywh_to_xyxy_pil(box, image.size)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y2+5), f"Pred: {id2label[label.item()]} ({score:.2f})", fill="red")
        
        # 저장
        image.save(output_dir / f"prediction_{i:04d}.jpg")
    
    print(f"Saved {min(len(predictions), max_images)} prediction images to {output_dir}")


def box_cxcywh_to_xyxy_pil(box, image_size):
    """CXCYWH를 PIL 이미지용 XYXY로 변환"""
    img_w, img_h = image_size
    cx, cy, w, h = box
    
    # 정규화된 좌표를 픽셀 좌표로 변환
    cx = cx * img_w
    cy = cy * img_h
    w = w * img_w
    h = h * img_h
    
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2
    
    return int(x1), int(y1), int(x2), int(y2)


def plot_confusion_matrix(cm, class_names, output_path, title="Confusion Matrix"):
    """Confusion matrix 플롯 및 저장"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {output_path}")
    
    plt.close()


def plot_class_metrics(class_stats, class_names, output_dir):
    """클래스별 메트릭 시각화"""
    output_dir = Path(output_dir)
    
    metrics = ['precision', 'recall', 'f1']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        values = class_stats[f'class_{metric}']
        
        bars = axes[i].bar(class_names, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[i].set_title(f'Class-wise {metric.capitalize()}')
        axes[i].set_ylabel(metric.capitalize())
        axes[i].set_ylim(0, 1)
        
        # 값 표시
        for bar, value in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / "class_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Class metrics plot saved to: {output_dir / 'class_metrics.png'}")


def create_evaluation_report(results, output_path):
    """평가 결과 HTML 리포트 생성"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .metric {{ margin: 10px 0; }}
            .class-table {{ border-collapse: collapse; width: 100%; }}
            .class-table th, .class-table td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            .class-table th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Model Evaluation Report</h1>
            <p><strong>Timestamp:</strong> {results['evaluation_info']['timestamp']}</p>
            <p><strong>Checkpoint:</strong> {results['evaluation_info']['checkpoint']}</p>
            <p><strong>Split:</strong> {results['evaluation_info']['split']}</p>
        </div>
        
        <h2>Detection Metrics</h2>
        <div class="metric">mAP@0.5: {results['detection_metrics']['map_50']:.4f}</div>
        <div class="metric">mAP@0.5:0.95: {results['detection_metrics']['map']:.4f}</div>
        
        <h2>Overall Statistics</h2>
        <div class="metric">Precision: {results['detailed_statistics']['total_statistics']['overall_precision']:.4f}</div>
        <div class="metric">Recall: {results['detailed_statistics']['total_statistics']['overall_recall']:.4f}</div>
        <div class="metric">F1-Score: {results['detailed_statistics']['total_statistics']['overall_f1']:.4f}</div>
        
        <!-- 추가 내용들... -->
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"HTML report saved to: {output_path}")
