import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torchvision.transforms as transforms
import json

COLORS = [
    [1.0, 0.0, 0.0],    # fully-ripe (red) - 진한 빨강
    [1.0, 0.647, 0.0],  # semi-ripe (orange) - 진한 주황
    [0.0, 0.5, 0.0],    # unripe (green) - 초록
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


# =============================================================================
# 공통 시각화 함수 (Test/Inference용)
# =============================================================================

def get_model_color(model_name):
    """모델별 색상 반환"""
    model_colors = {
        'yolov11': (255, 165, 0),  # 주황색
        'yolov12': (255, 255, 0),  # 노란색
        'detr': (128, 0, 128),     # 보라색
        'grounding dino': (139, 69, 19),  # 갈색
        'florence2': (0, 191, 255),  # 딥스카이블루
        'florence-2': (0, 191, 255),  # 딥스카이블루
    }
    model_name_lower = model_name.lower()
    return model_colors.get(model_name_lower, (255, 255, 255))  # 기본값: 흰색


def get_class_colors(num_classes):
    """클래스별 색상 반환 (COLORS 상수 사용)"""
    # COLORS 상수 사용 (0-1 범위를 0-255로 변환)
    colors = {}
    for i in range(num_classes):
        if i < len(COLORS):
            # COLORS는 0-1 범위이므로 0-255로 변환
            colors[i] = tuple(int(c * 255) for c in COLORS[i])
        else:
            # 추가 클래스는 색상 팔레트에서 가져오기
            import colorsys
            hue = i / max(num_classes, 1)
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            colors[i] = tuple(int(c * 255) for c in rgb)
    return colors


def save_visualization_images(
    model, 
    dataset, 
    output_dir,
    config,
    show_gt=False,
    box_color_mode='class',
    confidence_threshold=0.5,
    split='test',
    max_images=None,
    show_labels=False,
    dpi=1000,
    upscale_factor=2.0
):
    """
    공통 시각화 함수 - 모든 모델에서 사용
    
    Args:
        model: 학습된 모델
        dataset: 데이터셋
        output_dir: 출력 디렉토리 (Path 객체)
        config: 설정 딕셔너리
        show_gt: GT 표시 여부 (기본값: False)
        box_color_mode: 'class' (클래스별 색상) 또는 'model' (모델별 색상)
        confidence_threshold: 신뢰도 임계값
        split: 데이터셋 split 이름
        max_images: 최대 저장 이미지 수 (None이면 전체)
        show_labels: 클래스 라벨 텍스트 표시 여부 (기본값: False)
        dpi: 저장할 이미지의 DPI (기본값: 300, matplotlib 사용 시)
        upscale_factor: 이미지 업스케일링 배율 (기본값: 2.0, 원본 해상도 유지)
    """
    if model is None or dataset is None:
        return
    
    # 인퍼런스 이미지 저장 디렉토리 생성 
    inference_dir = Path(output_dir)
    inference_dir.mkdir(parents=True, exist_ok=True)
    
    device = next(model.parameters()).device
    model.eval()
    
    saved_count = 0
    class_names = config['data']['class_names']
    model_name = config['model']['arch_name']
    
    # max_images 처리
    if max_images is None:
        max_images = len(dataset) if dataset else 0
    
    total_images = min(len(dataset) if dataset else 0, max_images)
    
    # 색상 설정
    if box_color_mode == 'model':
        # 모델별 단일 색상
        model_color = get_model_color(model_name)
        box_colors = {i: model_color for i in range(len(class_names))}
    else:
        # 클래스별 색상 (기본)
        box_colors = get_class_colors(len(class_names))
    
    print(f"Saving inference images to: {inference_dir}")
    print(f"Total images to process: {total_images}")
    print(f"Box color mode: {box_color_mode}")
    print(f"Show GT: {show_gt}")
    print(f"Show labels: {show_labels}")
    
    # 파일명 순서대로 정렬하기 위한 인덱스 매핑 생성
    # COCO 데이터셋에서 이미지 정보 가져오기
    sorted_indices = None
    image_id_to_filename = {}
    
    if hasattr(dataset, 'coco') and hasattr(dataset, 'ids'):
        coco = dataset.coco
        # 이미지 ID와 파일명 매핑
        for img_id in dataset.ids:
            img_info = coco.loadImgs(img_id)[0]
            image_id_to_filename[img_id] = img_info['file_name']
        
        # 파일명으로 정렬된 인덱스 리스트 생성
        sorted_indices = sorted(
            range(len(dataset.ids)),
            key=lambda idx: image_id_to_filename[dataset.ids[idx]]
        )
        print(f"Images will be saved in filename order (sorted by {len(sorted_indices)} filenames)")
    else:
        # COCO 데이터셋이 아닌 경우 기존 순서 유지
        sorted_indices = list(range(len(dataset)))
        print(f"Using dataset index order (COCO dataset not detected)")
    
    with torch.no_grad():
        for save_idx, dataset_idx in enumerate(sorted_indices[:total_images]):
            # 데이터셋에서 이미지와 타겟 가져오기
            sample = dataset[dataset_idx]
            
            # 이미지 처리
            if isinstance(sample, dict):
                image_tensor = sample['pixel_values']
                target = sample.get('labels') if show_gt else None
            else:
                image_tensor, target = sample if show_gt else (sample[0], None)
            
            # 배치 차원 추가
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            # 모델 예측
            image_tensor = image_tensor.to(device)
            outputs = model(image_tensor)
            
            # 예측 결과 처리 (DETR 형식)
            probs = outputs.logits.softmax(-1)[0, :, :-1]
            scores, pred_labels = probs.max(-1)
            pred_boxes = outputs.pred_boxes[0]
            
            # 임계값 적용
            keep = scores > confidence_threshold
            pred_boxes_filtered = pred_boxes[keep]
            pred_scores_filtered = scores[keep]
            pred_labels_filtered = pred_labels[keep]
            
            # 이미지를 PIL로 변환
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            
            image_denorm = image_tensor[0].cpu() * std + mean
            image_denorm = torch.clamp(image_denorm, 0, 1)
            
            to_pil = transforms.ToPILImage()
            pil_image = to_pil(image_denorm)
            
            img_w, img_h = pil_image.size
            
            draw = ImageDraw.Draw(pil_image)
            
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Ground Truth 그리기 (옵션에 따라)
            if show_gt and target is not None:
                gt_boxes = target.get('boxes', [])
                gt_labels = target.get('class_labels', [])
                
                if len(gt_boxes) > 0:
                    for box, label in zip(gt_boxes, gt_labels):
                        cx, cy, w, h = box
                        x1 = (cx - w/2) * img_w
                        y1 = (cy - h/2) * img_h
                        x2 = (cx + w/2) * img_w
                        y2 = (cy + h/2) * img_h
                        
                        draw.rectangle([x1, y1, x2, y2], outline="blue", width=4)
                        
                        if show_labels and label.item() < len(class_names):
                            text = f"GT: {class_names[label.item()]}"
                            draw.text((x1, y1-20), text, fill="blue", font=font)
            
            # Predictions 그리기
            for box, label, score in zip(pred_boxes_filtered, pred_labels_filtered, pred_scores_filtered):
                cx, cy, w, h = box
                x1 = (cx - w/2) * img_w
                y1 = (cy - h/2) * img_h
                x2 = (cx + w/2) * img_w
                y2 = (cy + h/2) * img_h
                
                color = box_colors.get(label.item(), (255, 255, 255))
                
                draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                
                if show_labels and label.item() < len(class_names):
                    text = f"{class_names[label.item()]}: {score:.2f}"
                    bbox = draw.textbbox((x1, y2+5), text, font=font)
                    draw.rectangle(bbox, fill=color)
                    draw.text((x1, y2+5), text, fill="white", font=font)
            
            # 이미지 저장 - DPI 조절 및 해상도 개선 (PNG만 저장)
            filename = f"inference_{save_idx:04d}_{split}.png"
            save_path = inference_dir / filename
            
            # 방법 1: 해상도 개선 - 업스케일링 후 저장
            if upscale_factor > 1.0:
                original_size = pil_image.size
                new_size = (int(original_size[0] * upscale_factor), 
                           int(original_size[1] * upscale_factor))
                # LANCZOS 리샘플링으로 고품질 업스케일링
                upscaled_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
                
                # 업스케일링된 이미지에 bbox 다시 그리기
                upscaled_w, upscaled_h = upscaled_image.size
                upscaled_draw = ImageDraw.Draw(upscaled_image)
                
                # 폰트 크기도 업스케일링에 맞춰 조정
                try:
                    scaled_font_size = max(12, int(16 * upscale_factor))
                    scaled_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", scaled_font_size)
                except:
                    scaled_font = font
                
                line_width = max(2, int(4 * upscale_factor))
                
                # Ground Truth 다시 그리기
                if show_gt and target is not None:
                    gt_boxes = target.get('boxes', [])
                    gt_labels = target.get('class_labels', [])
                    
                    if len(gt_boxes) > 0:
                        for box, label in zip(gt_boxes, gt_labels):
                            cx, cy, w, h = box
                            x1 = (cx - w/2) * upscaled_w
                            y1 = (cy - h/2) * upscaled_h
                            x2 = (cx + w/2) * upscaled_w
                            y2 = (cy + h/2) * upscaled_h
                            
                            upscaled_draw.rectangle([x1, y1, x2, y2], outline="blue", width=line_width)
                            
                            if show_labels and label.item() < len(class_names):
                                text = f"GT: {class_names[label.item()]}"
                                upscaled_draw.text((x1, y1-20*upscale_factor), text, fill="blue", font=scaled_font)
                
                # Predictions 다시 그리기
                for box, label, score in zip(pred_boxes_filtered, pred_labels_filtered, pred_scores_filtered):
                    cx, cy, w, h = box
                    x1 = (cx - w/2) * upscaled_w
                    y1 = (cy - h/2) * upscaled_h
                    x2 = (cx + w/2) * upscaled_w
                    y2 = (cy + h/2) * upscaled_h
                    
                    color = box_colors.get(label.item(), (255, 255, 255))
                    upscaled_draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
                    
                    if show_labels and label.item() < len(class_names):
                        text = f"{class_names[label.item()]}: {score:.2f}"
                        bbox = upscaled_draw.textbbox((x1, y2+5*upscale_factor), text, font=scaled_font)
                        upscaled_draw.rectangle(bbox, fill=color)
                        upscaled_draw.text((x1, y2+5*upscale_factor), text, fill="white", font=scaled_font)
                
                pil_image = upscaled_image
                img_w, img_h = pil_image.size
            
            # DPI 조절 - Matplotlib 사용 (고해상도 PNG 저장만)
            # PIL 이미지를 numpy 배열로 변환
            img_array = np.array(pil_image)
            
            # Matplotlib으로 PNG 저장 (DPI 완전 제어)
            fig, ax = plt.subplots(figsize=(img_w/dpi, img_h/dpi), dpi=dpi)
            ax.imshow(img_array)
            ax.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0, format='png')
            plt.close(fig)
            
            saved_count += 1
            
            if saved_count % 10 == 0:
                print(f"  Saved {saved_count}/{total_images} images...")
    
    print(f" Saved {saved_count} inference images to: {inference_dir}")
    
    # 범례 이미지 생성
    create_legend_image(inference_dir, class_names, box_colors, show_gt)
    
    return inference_dir


def create_legend_image(inference_dir, class_names, box_colors, model_name=None, show_gt=False):
    """범례 이미지 생성"""
    legend_width = 400
    legend_height = 200 + len(class_names) * 30
    legend_img = Image.new('RGB', (legend_width, legend_height), 'white')
    draw = ImageDraw.Draw(legend_img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
    
    draw.text((20, 20), "Detection Legend", fill="black", font=title_font)
    
    y_offset = 60
    if show_gt:
        draw.rectangle([20, y_offset, 40, y_offset+20], outline="blue", width=3)
        draw.text((50, y_offset+5), "Ground Truth", fill="blue", font=font)
        y_offset += 30
    
    for i, class_name in enumerate(class_names):
        color = box_colors.get(i, (255, 255, 255))
        draw.rectangle([20, y_offset, 40, y_offset+20], outline=color, width=3)
        draw.text((50, y_offset+5), f"Predicted: {class_name}", fill=color, font=font)
        y_offset += 30
    
    legend_path = Path(inference_dir) / "legend.jpg"
    legend_img.save(legend_path, quality=95)
    print(f" Legend saved to: {legend_path}")


def save_yolo_visualization_images(
    yolo_results,
    output_dir,
    config,
    show_gt=False,
    box_color_mode='class',
    confidence_threshold=0.5,
    split='test',
    max_images=None,
    show_labels=False,
    yolo_output_dir=None,
    original_images_dir=None
):
    """
    YOLO predict 결과를 우리의 색상 스키마로 재시각화
    
    Args:
        yolo_results: YOLO model.predict() 결과 리스트
        output_dir: 출력 디렉토리 (Path 객체)
        config: 설정 딕셔너리
        show_gt: GT 표시 여부 (기본값: False)
        box_color_mode: 'class' (클래스별 색상) 또는 'model' (모델별 색상)
        confidence_threshold: 신뢰도 임계값
        split: 데이터셋 split 이름
        max_images: 최대 저장 이미지 수 (None이면 전체)
        show_labels: 클래스 라벨 텍스트 표시 여부 (기본값: False)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    
    # 인퍼런스 이미지 저장 디렉토리 생성
    inference_dir = Path(output_dir) / "model"
    inference_dir.mkdir(parents=True, exist_ok=True)
    
    class_names = config['data']['class_names']
    model_name = config['model']['arch_name']
    
    # max_images 처리
    if max_images is None:
        max_images = len(yolo_results)
    
    total_images = min(len(yolo_results), max_images)
    
    # 색상 설정 (matplotlib용 - 0-1 범위로 정규화)
    # yolo_predictions는 항상 클래스별 색상 사용 (COLORS 상수)
    box_colors_rgb_class = get_class_colors(len(class_names))
    box_colors_class = {i: tuple(c / 255.0 for c in color) for i, color in box_colors_rgb_class.items()}
    
    # inference_images는 box_color_mode에 따라
    if box_color_mode == 'model':
        model_color_rgb = get_model_color(model_name)
        model_color = tuple(c / 255.0 for c in model_color_rgb)
        box_colors_inference = {i: model_color for i in range(len(class_names))}
        box_colors_rgb_inference = {i: model_color_rgb for i in range(len(class_names))}
    else:
        box_colors_inference = box_colors_class
        box_colors_rgb_inference = box_colors_rgb_class
    
    print(f"Saving YOLO inference images to: {inference_dir}")
    print(f"Total images to process: {total_images}")
    print(f"Box color mode: {box_color_mode}")
    print(f"Show GT: {show_gt}")
    print(f"Show labels: {show_labels}")
    
    saved_count = 0
    
    for idx, result in enumerate(yolo_results[:total_images]):
        # 원본 이미지 경로 찾기
        original_image_path = None
        
        # 방법 1: original_images_dir에서 파일명으로 찾기
        if original_images_dir is not None:
            if hasattr(result, 'path'):
                # result.path에서 파일명 추출
                result_path = Path(result.path)
                file_name = result_path.name
                original_image_path = Path(original_images_dir) / file_name
                if not original_image_path.exists():
                    # 확장자 제거 후 다시 찾기
                    base_name = result_path.stem
                    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                        candidate = Path(original_images_dir) / (base_name + ext)
                        if candidate.exists():
                            original_image_path = candidate
                            break
        
        # 방법 2: result.path가 원본 이미지 경로인 경우
        if original_image_path is None and hasattr(result, 'path'):
            result_path = Path(result.path)
            # yolo_predictions 폴더가 아닌 원본 이미지 폴더인지 확인
            if 'yolo_predictions' not in str(result_path):
                original_image_path = result_path
        
        # 방법 3: result에서 직접 경로 가져오기
        if original_image_path is None:
            # YOLO result 객체의 속성 확인
            if hasattr(result, 'source') or hasattr(result, 'im_file'):
                # YOLO v8+ 형식
                if hasattr(result, 'im_file'):
                    original_image_path = Path(result.im_file)
                elif hasattr(result, 'source'):
                    original_image_path = Path(result.source)
        
        # 원본 이미지 로드
        if original_image_path is not None and original_image_path.exists():
            # 원본 파일에서 직접 로드 (100% 원본 품질 보존)
            # load_and_orient_image 사용 (EXIF 및 COCO orientation 처리)
            pil_image = load_and_orient_image(original_image_path)
        elif hasattr(result, 'orig_img'):
            # orig_img 사용 (이미 처리된 이미지일 수 있음)
            img_array = result.orig_img
            if isinstance(img_array, np.ndarray):
                if img_array.dtype != np.uint8:
                    if img_array.max() <= 1.0:
                        img_array = (img_array * 255).astype(np.uint8)
                    else:
                        img_array = img_array.astype(np.uint8)
                pil_image = Image.fromarray(img_array)
            else:
                pil_image = img_array
        else:
            print(f"Warning: Could not find original image for result {idx}")
            continue
        
        # Predictions 정보 가져오기
        xyxy_filtered = []
        confidences_filtered = []
        class_ids_filtered = []
        
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            
            if hasattr(boxes, 'xyxy') and hasattr(boxes, 'conf') and hasattr(boxes, 'cls'):
                xyxy = boxes.xyxy.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy().astype(int)
                
                # 임계값 적용
                keep = confidences >= confidence_threshold
                xyxy_filtered = xyxy[keep]
                confidences_filtered = confidences[keep]
                class_ids_filtered = class_ids[keep]
        
        # inference_images용 이미지 생성 (PIL ImageDraw 사용)
        img_inference = pil_image.copy()
        draw_inference = ImageDraw.Draw(img_inference)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # inference_images용 박스 그리기
        for box, label_id, score in zip(xyxy_filtered, class_ids_filtered, confidences_filtered):
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            
            # 색상 가져오기 (RGB 튜플)
            color_rgb = box_colors_rgb_inference.get(label_id, (255, 255, 255))
            
            # 박스 그리기
            draw_inference.rectangle([x1, y1, x2, y2], outline=color_rgb, width=8)
            
            # 라벨 텍스트
            if show_labels and label_id < len(class_names):
                text = f"{class_names[label_id]}: {score:.2f}"
                bbox = draw_inference.textbbox((x1, y1 - 20), text, font=font)
                draw_inference.rectangle(bbox, fill=color_rgb)
                draw_inference.text((x1, y1 - 20), text, fill="white", font=font)
        
        # model 폴더에 저장 (원본 품질 보존)
        # 번호 순서로 파일명 지정
        filename = f"inference_{idx:04d}_{split}.jpg"
        save_path = inference_dir / filename
        img_inference.save(save_path, quality=95, optimize=False)
        
        saved_count += 1
        
        if saved_count % 10 == 0:
            print(f"  Saved {saved_count}/{total_images} images...")
    
    print(f" Saved {saved_count} YOLO inference images to: {inference_dir}")
    
    # 범례 이미지 생성
    create_legend_image(inference_dir, class_names, box_colors_rgb_class, model_name, show_gt)
    
    return inference_dir


def load_and_orient_image(image_path, coco_annotations=None, image_id=None):
    """
    이미지를 로드하고 COCO annotation의 orientation 정보에 따라 회전시킴
    
    Args:
        image_path: 이미지 파일 경로
        coco_annotations: COCO 형식의 annotations 딕셔너리 (선택)
        image_id: 이미지 ID (coco_annotations에서 찾기 위해 필요)
    
    Returns:
        PIL.Image: 회전 처리된 이미지
    """
    # 이미지 로드 및 EXIF orientation 처리
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    image = image.convert('RGB')
    
    # COCO annotation에서 orientation 정보 확인
    if coco_annotations and image_id is not None:
        # images 리스트에서 해당 image_id 찾기
        for img_info in coco_annotations.get('images', []):
            if img_info.get('id') == image_id:
                # orientation 정보가 있으면 회전
                orientation = img_info.get('orientation', 0)
                if orientation != 0:
                    # orientation 값에 따라 회전
                    # 90도: Image.ROTATE_90, 180도: Image.ROTATE_180, 270도: Image.ROTATE_270
                    if orientation == 90:
                        image = image.transpose(Image.ROTATE_90)
                    elif orientation == 180:
                        image = image.transpose(Image.ROTATE_180)
                    elif orientation == 270:
                        image = image.transpose(Image.ROTATE_270)
                break
    
    return image