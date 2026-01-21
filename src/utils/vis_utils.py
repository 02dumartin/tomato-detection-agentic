"""시각화 유틸리티 - 통합 버전"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torchvision.transforms as transforms

# 상수
COLORS = [
    [1.0, 0.0, 0.0],    # fully-ripe (red)
    [1.0, 0.647, 0.0],  # semi-ripe (orange)
    [0.0, 0.5, 0.0],    # unripe (green)
]

# =============================================================================
# 유틸리티 함수
# =============================================================================

def box_cxcywh_to_xyxy(box, image_size):
    """CXCYWH를 XYXY로 변환 (정규화된 좌표)"""
    img_w, img_h = image_size
    cx, cy, w, h = box
    x1 = (cx - w/2) * img_w
    y1 = (cy - h/2) * img_h
    x2 = (cx + w/2) * img_w
    y2 = (cy + h/2) * img_h
    return int(x1), int(y1), int(x2), int(y2)


def get_model_color(model_name):
    """모델별 색상 반환"""
    model_colors = {
        'yolov11': (255, 165, 0),
        'yolov12': (255, 255, 0),
        'detr': (128, 0, 128),
        'grounding dino': (139, 69, 19),
        'florence2': (0, 191, 255),
        'florence-2': (0, 191, 255),
    }
    return model_colors.get(model_name.lower(), (255, 255, 255))


def get_class_colors(num_classes, class_names=None):
    """
    클래스별 색상 반환
    
    Args:
        num_classes: 클래스 개수
        class_names: 클래스 이름 리스트 (선택적)
    
    Returns:
        Dict[int, Tuple[int, int, int]]: 클래스 ID -> RGB 색상 튜플 (0-255)
    
    클래스 이름별 색상 매핑:
        - fully-ripe -> red (255, 0, 0)
        - semi-ripe -> orange (255, 165, 0)
        - unripe -> green (0, 128, 0)
    """
    # 클래스 이름별 색상 매핑 정의
    class_name_to_color = {
        'fully-ripe': [1.0, 0.0, 0.0],      # red
        'semi-ripe': [1.0, 0.647, 0.0],     # orange
        'unripe': [0.0, 0.5, 0.0],          # green
    }
    
    colors = {}
    for i in range(num_classes):
        if class_names and i < len(class_names):
            # 클래스 이름으로 색상 찾기
            class_name = class_names[i]
            if class_name in class_name_to_color:
                # 클래스 이름에 해당하는 색상 사용
                colors[i] = tuple(int(c * 255) for c in class_name_to_color[class_name])
            else:
                # 기본 색상 배열 사용 (fallback)
                if i < len(COLORS):
                    colors[i] = tuple(int(c * 255) for c in COLORS[i])
                else:
                    # HSV로 자동 생성
                    import colorsys
                    hue = i / max(num_classes, 1)
                    rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
                    colors[i] = tuple(int(c * 255) for c in rgb)
        else:
            # 클래스 이름이 없으면 기본 색상 배열 사용 (하위 호환성)
        
            if i < len(COLORS):
                colors[i] = tuple(int(c * 255) for c in COLORS[i])
            else:
                import colorsys
                hue = i / max(num_classes, 1)
                rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
                colors[i] = tuple(int(c * 255) for c in rgb)
        
    return colors


def load_and_orient_image(image_path, coco_annotations=None, image_id=None):
    """
    이미지 로드 및 EXIF orientation 처리
    COCO annotation의 orientation 정보가 있으면 추가 회전 처리
    
    Args:
        image_path: 이미지 파일 경로
        coco_annotations: COCO 형식의 annotations 딕셔너리 (선택)
        image_id: 이미지 ID (coco_annotations에서 찾기 위해 필요)
    
    Returns:
        PIL.Image: 회전 처리된 이미지
    """
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
                    if orientation == 90:
                        image = image.transpose(Image.ROTATE_90)
                    elif orientation == 180:
                        image = image.transpose(Image.ROTATE_180)
                    elif orientation == 270:
                        image = image.transpose(Image.ROTATE_270)
                break
    
    return image


def _get_font(size=16):
    """폰트 가져오기"""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except:
        return ImageFont.load_default()


def _draw_boxes(draw, boxes, labels, scores, box_colors, class_names, show_labels, font, img_size=None):
    """박스 그리기 (공통 함수)"""
    if img_size is None:
        img_size = draw.im.size
    
    for box, label, score in zip(boxes, labels, scores):
        if isinstance(box, (list, tuple)) and len(box) == 4:
            # XYXY 형식 (이미 픽셀 좌표)
            x1, y1, x2, y2 = map(float, box)
        elif isinstance(box, torch.Tensor):
            # Tensor인 경우
            if box.dim() == 1 and len(box) == 4:
                # CXCYWH 형식 (정규화됨)
                x1, y1, x2, y2 = box_cxcywh_to_xyxy(box.tolist(), img_size)
            else:
                continue
        else:
            # CXCYWH 형식 (정규화됨)
            x1, y1, x2, y2 = box_cxcywh_to_xyxy(box, img_size)
        
        color = box_colors.get(int(label), (255, 255, 255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        
        if show_labels and int(label) < len(class_names):
            text = f"{class_names[int(label)]}: {score:.2f}"
            bbox = draw.textbbox((x1, y2+5), text, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x1, y2+5), text, fill="white", font=font)


def _save_image(pil_image, save_path, dpi=1000):
    """고해상도 이미지 저장"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        img_array = np.array(pil_image)
        fig, ax = plt.subplots(figsize=(pil_image.width/dpi, pil_image.height/dpi), dpi=dpi)
        ax.imshow(img_array)
        ax.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0, format='png')
        plt.close(fig)
    except Exception as e:
        print(f"Warning: Could not save image with matplotlib DPI={dpi}: {e}")
        print(f"  Falling back to PIL save with high quality...")
        # Fallback: PIL로 직접 저장 (고품질)
        pil_image.save(save_path, format='PNG', quality=100, optimize=False)


def _load_yolo_image(result, original_images_dir):
    """YOLO 결과에서 원본 이미지 로드"""
    original_image_path = None
    
    if original_images_dir and hasattr(result, 'path'):
        file_name = Path(result.path).name
        original_image_path = Path(original_images_dir) / file_name
        if not original_image_path.exists():
            base_name = Path(result.path).stem
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                candidate = Path(original_images_dir) / (base_name + ext)
                if candidate.exists():
                    original_image_path = candidate
                    break
    
    if original_image_path is None and hasattr(result, 'im_file'):
        original_image_path = Path(result.im_file)
    elif original_image_path is None and hasattr(result, 'source'):
        original_image_path = Path(result.source)
    
    if original_image_path and original_image_path.exists():
        return load_and_orient_image(original_image_path)
    elif hasattr(result, 'orig_img'):
        img_array = result.orig_img
        if isinstance(img_array, np.ndarray):
            if img_array.dtype != np.uint8:
                img_array = (img_array * 255).astype(np.uint8) if img_array.max() <= 1.0 else img_array.astype(np.uint8)
            return Image.fromarray(img_array)
        return img_array
    return None


def _extract_yolo_predictions(result, confidence_threshold):
    """YOLO 결과에서 예측 정보 추출"""
    boxes, labels, scores = [], [], []
    
    if hasattr(result, 'boxes') and result.boxes is not None:
        boxes_obj = result.boxes
        if hasattr(boxes_obj, 'xyxy') and hasattr(boxes_obj, 'conf') and hasattr(boxes_obj, 'cls'):
            xyxy = boxes_obj.xyxy.cpu().numpy()
            confidences = boxes_obj.conf.cpu().numpy()
            class_ids = boxes_obj.cls.cpu().numpy().astype(int)
            
            keep = confidences >= confidence_threshold
            boxes = xyxy[keep].tolist()
            
            # YOLO의 category_id 변환
            # 3-class 이상: category_id는 1부터 시작 (1, 2, 3...) → -1 해서 0-based로 변환
            # 1-class: category_id는 0 또는 1일 수 있음
            # 최소값을 확인해서 0-based로 변환
            filtered_class_ids = class_ids[keep]
            if len(filtered_class_ids) > 0:
                min_class_id = int(filtered_class_ids.min())
                if min_class_id == 0:
                    # 이미 0-based인 경우 (1-class 모델)
                    labels = filtered_class_ids.tolist()
                else:
                    # 1-based인 경우 (3-class 이상 모델) -1 해서 변환
                    labels = (filtered_class_ids - 1).tolist()
            else:
                labels = []
            
            scores = confidences[keep].tolist()
    
    return boxes, labels, scores


def _get_sorted_indices(dataset):
    """COCO 데이터셋 파일명 순서로 정렬"""
    if hasattr(dataset, 'coco') and hasattr(dataset, 'ids'):
        image_id_to_filename = {}
        for img_id in dataset.ids:
            img_info = dataset.coco.loadImgs(img_id)[0]
            image_id_to_filename[img_id] = img_info['file_name']
        return sorted(range(len(dataset.ids)), key=lambda idx: image_id_to_filename[dataset.ids[idx]])
    return list(range(len(dataset)))


def _tensor_to_pil(image_tensor):
    """Tensor를 PIL 이미지로 변환"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_denorm = image_tensor * std + mean
    image_denorm = torch.clamp(image_denorm, 0, 1)
    return transforms.ToPILImage()(image_denorm)


# =============================================================================
# 공통 시각화 함수 (통합)
# =============================================================================

def save_visualization_images(
    model=None,
    dataset=None,
    yolo_results=None,
    output_dir=None,
    config=None,
    show_gt=False,
    box_color_mode='class',
    confidence_threshold=0.5,
    split='test',
    max_images=None,
    show_labels=False,
    original_images_dir=None
):
    """
    통합 시각화 함수 - 모든 모델에서 사용
    
    Args:
        model: PyTorch 모델 (DETR/Florence-2용)
        dataset: 데이터셋 (DETR/Florence-2용)
        yolo_results: YOLO predict 결과 리스트 (YOLO용)
        output_dir: 출력 디렉토리
        config: 설정 딕셔너리
        show_gt: GT 표시 여부
        box_color_mode: 'class' 또는 'model'
        confidence_threshold: 신뢰도 임계값
        split: 데이터셋 split 이름
        max_images: 최대 저장 이미지 수
        show_labels: 라벨 표시 여부
        original_images_dir: 원본 이미지 디렉토리 (YOLO용)
    """
    output_dir = Path(output_dir)
    inference_dir = output_dir / ("model" if yolo_results else "inference")
    inference_dir.mkdir(parents=True, exist_ok=True)
    
    class_names = config['data']['class_names']
    model_name = config['model']['arch_name']
    
    # 색상 설정
    if box_color_mode == 'model':
        model_color = get_model_color(model_name)
        box_colors = {i: model_color for i in range(len(class_names))}
    else:
        # 클래스 이름을 전달하여 정확한 색상 매핑
        box_colors = get_class_colors(len(class_names), class_names)
    
    font = _get_font(16)
    saved_count = 0
    
    # YOLO 모드
    if yolo_results is not None:
        total_images = min(len(yolo_results), max_images or len(yolo_results))
        
        print(f"Saving YOLO inference images to: {inference_dir}")
        print(f"Total images to process: {total_images}")
        print(f"Box color mode: {box_color_mode}")
        print(f"Show GT: {show_gt}")
        print(f"Show labels: {show_labels}")
        
        for idx, result in enumerate(yolo_results[:total_images]):
            # 원본 이미지 로드
            pil_image = _load_yolo_image(result, original_images_dir)
            if pil_image is None:
                print(f"Warning: Could not find original image for result {idx}")
                continue
            
            # 예측 결과 추출
            boxes, labels, scores = _extract_yolo_predictions(result, confidence_threshold)
            
            if len(boxes) == 0:
                # 예측이 없는 경우에도 이미지 저장
                filename = f"inference_{idx:04d}_{split}.jpg"
                pil_image.save(inference_dir / filename, quality=95)
                saved_count += 1
                continue
            
            # 박스 그리기
            draw = ImageDraw.Draw(pil_image)
            _draw_boxes(draw, boxes, labels, scores, box_colors, class_names, show_labels, font)
            
            # 저장
            filename = f"inference_{idx:04d}_{split}.jpg"
            pil_image.save(inference_dir / filename, quality=95, optimize=False)
            saved_count += 1
            
            if saved_count % 10 == 0:
                print(f"  Saved {saved_count}/{total_images} images...")
    
    # DETR/Florence-2 모드
    elif model is not None and dataset is not None:
        device = next(model.parameters()).device
        model.eval()
        
        total_images = min(len(dataset), max_images or len(dataset))
        
        print(f"Saving inference images to: {inference_dir}")
        print(f"Total images to process: {total_images}")
        print(f"Box color mode: {box_color_mode}")
        print(f"Show GT: {show_gt}")
        print(f"Show labels: {show_labels}")
        
        # COCO 데이터셋 정렬
        sorted_indices = _get_sorted_indices(dataset)
        if len(sorted_indices) > 0:
            print(f"Images will be saved in filename order (sorted by {len(sorted_indices)} filenames)")
        else:
            print(f"Using dataset index order (COCO dataset not detected)")
        
        with torch.no_grad():
            for save_idx, dataset_idx in enumerate(sorted_indices[:total_images]):
                sample = dataset[dataset_idx]
                
                # 이미지 처리
                if isinstance(sample, dict):
                    image_tensor = sample['pixel_values']
                    target = sample.get('labels') if show_gt else None
                else:
                    image_tensor, target = sample if show_gt else (sample[0], None)
                
                if image_tensor.dim() == 3:
                    image_tensor = image_tensor.unsqueeze(0)
                
                # 모델 예측
                image_tensor = image_tensor.to(device)
                outputs = model(image_tensor)
                
                # 예측 결과 처리
                probs = outputs.logits.softmax(-1)[0, :, :-1]
                scores, pred_labels = probs.max(-1)
                pred_boxes = outputs.pred_boxes[0]
                
                keep = scores > confidence_threshold
                boxes = pred_boxes[keep].cpu()
                labels = pred_labels[keep].cpu()
                scores = scores[keep].cpu()
                
                # PIL 이미지로 변환
                pil_image = _tensor_to_pil(image_tensor[0].cpu())
                draw = ImageDraw.Draw(pil_image)
                
                # GT 그리기
                if show_gt and target:
                    gt_boxes = target.get('boxes', [])
                    gt_labels = target.get('class_labels', [])
                    for box, label in zip(gt_boxes, gt_labels):
                        x1, y1, x2, y2 = box_cxcywh_to_xyxy(box, pil_image.size)
                        draw.rectangle([x1, y1, x2, y2], outline="blue", width=4)
                        if show_labels and label.item() < len(class_names):
                            text = f"GT: {class_names[label.item()]}"
                            draw.text((x1, y1-20), text, fill="blue", font=font)
                
                # 예측 그리기
                _draw_boxes(draw, boxes, labels, scores, box_colors, class_names, show_labels, font)
                
                # 저장
                filename = f"inference_{save_idx:04d}_{split}.png"
                _save_image(pil_image, inference_dir / filename)
                saved_count += 1
                
                if saved_count % 10 == 0:
                    print(f"  Saved {saved_count}/{total_images} images...")
    
    print(f"Saved {saved_count} inference images to: {inference_dir}")
    
    # 범례 생성
    create_legend_image(inference_dir, class_names, box_colors, show_gt)
    return inference_dir


def create_legend_image(inference_dir, class_names, box_colors, show_gt=False):
    """범례 이미지 생성"""
    legend_width = 400
    legend_height = 200 + len(class_names) * 30
    legend_img = Image.new('RGB', (legend_width, legend_height), 'white')
    draw = ImageDraw.Draw(legend_img)
    
    font = _get_font(16)
    title_font = _get_font(20)
    
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
    
    legend_path = inference_dir / "legend.jpg"
    legend_img.save(legend_path, quality=95)
    print(f"Legend saved to: {legend_path}")


# =============================================================================
# 하위 호환성을 위한 별칭 (기존 코드 호환)
# =============================================================================

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
    YOLO 전용 시각화 함수 (하위 호환성)
    내부적으로 save_visualization_images를 호출
    """
    return save_visualization_images(
        yolo_results=yolo_results,
        output_dir=output_dir,
        config=config,
        show_gt=show_gt,
        box_color_mode=box_color_mode,
        confidence_threshold=confidence_threshold,
        split=split,
        max_images=max_images,
        show_labels=show_labels,
        original_images_dir=original_images_dir
    )

