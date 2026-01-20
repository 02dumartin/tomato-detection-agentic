"""
YOLO Data Transformation and Conversion Utilities

This module provides utilities to convert COCO format annotations to YOLO format
and organize data according to YOLO requirements.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm


def coco_to_yolo_bbox(coco_bbox: List[float], img_width: int, img_height: int) -> List[float]:
    """
    Convert COCO bbox format to YOLO format
    
    COCO format: [x_min, y_min, width, height] (absolute pixels)
    YOLO format: [x_center, y_center, width, height] (normalized 0-1)
    
    Args:
        coco_bbox: COCO format bbox [x_min, y_min, width, height]
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        YOLO format bbox [x_center, y_center, width, height] (normalized)
    """
    x_min, y_min, width, height = coco_bbox
    
    # Calculate center coordinates
    x_center = x_min + width / 2
    y_center = y_min + height / 2
    
    # Normalize to 0-1
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    # Clip to valid range [0, 1]
    x_center_norm = max(0.0, min(1.0, x_center_norm))
    y_center_norm = max(0.0, min(1.0, y_center_norm))
    width_norm = max(0.0, min(1.0, width_norm))
    height_norm = max(0.0, min(1.0, height_norm))
    
    return [x_center_norm, y_center_norm, width_norm, height_norm]


def convert_coco_to_yolo(
    coco_json_path: str,
    images_src_dir: str,
    output_dir: str,
    class_mapping: Dict[int, int] = None,
    copy_images: bool = True
) -> Tuple[int, int]:
    """
    Convert COCO format dataset to YOLO format
    
    Args:
        coco_json_path: Path to COCO format JSON file
        images_src_dir: Source directory containing images
        output_dir: Output directory for YOLO format data
        class_mapping: Optional mapping from COCO category_id to YOLO class_id
                      If None, uses category_id as-is
        copy_images: Whether to copy images to output directory
    
    Returns:
        Tuple of (total_images, total_annotations)
    """
    coco_json_path = Path(coco_json_path)
    images_src_dir = Path(images_src_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    images_out_dir = output_dir / "images"
    labels_out_dir = output_dir / "labels"
    images_out_dir.mkdir(parents=True, exist_ok=True)
    labels_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load COCO annotations
    print(f"Loading COCO annotations from: {coco_json_path}")
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Build image ID to filename mapping
    images_dict = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # Process each image
    total_images = 0
    total_annotations = 0
    skipped_images = []
    
    print(f"Converting {len(images_dict)} images...")
    for img_id, img_info in tqdm(images_dict.items(), desc="Converting"):
        file_name = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Source image path
        src_img_path = images_src_dir / file_name
        
        # Check if image exists
        if not src_img_path.exists():
            skipped_images.append(file_name)
            continue
        
        # Copy image if requested
        if copy_images:
            dst_img_path = images_out_dir / file_name
            shutil.copy2(src_img_path, dst_img_path)
        
        # Get annotations for this image
        annotations = annotations_by_image.get(img_id, [])
        
        # Create YOLO label file
        label_file_name = Path(file_name).stem + '.txt'
        label_path = labels_out_dir / label_file_name
        
        with open(label_path, 'w') as f:
            for ann in annotations:
                category_id = ann['category_id']
                
                # Apply class mapping if provided
                if class_mapping is not None:
                    if category_id not in class_mapping:
                        print(f"Warning: category_id {category_id} not in class_mapping, skipping")
                        continue
                    yolo_class_id = class_mapping[category_id]
                else:
                    yolo_class_id = category_id
                
                # Convert bbox
                coco_bbox = ann['bbox']
                yolo_bbox = coco_to_yolo_bbox(coco_bbox, img_width, img_height)
                
                # Write YOLO format: class_id x_center y_center width height
                f.write(f"{yolo_class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")
                total_annotations += 1
        
        total_images += 1
    
    if skipped_images:
        print(f"\nWarning: {len(skipped_images)} images not found in source directory:")
        for img in skipped_images[:10]:
            print(f"  - {img}")
        if len(skipped_images) > 10:
            print(f"  ... and {len(skipped_images) - 10} more")
    
    print(f"\nConversion complete!")
    print(f"  Total images: {total_images}")
    print(f"  Total annotations: {total_annotations}")
    print(f"  Output directory: {output_dir}")
    
    return total_images, total_annotations


def create_yolo_data_yaml(
    output_path: str,
    data_root: str,
    class_names: List[str],
    train_subdir: str = "train/images",
    val_subdir: str = "val/images",
    test_subdir: str = "test/images"
) -> None:
    """
    Create YOLO data.yaml configuration file
    
    Args:
        output_path: Path to save data.yaml
        data_root: Root directory of the dataset (absolute path)
        class_names: List of class names in order
        train_subdir: Subdirectory for training images
        val_subdir: Subdirectory for validation images
        test_subdir: Subdirectory for test images
    """
    output_path = Path(output_path)
    
    yaml_content = f"""names:
{chr(10).join(f'- {name}' for name in class_names)}
nc: {len(class_names)}
path: {data_root}
test: {test_subdir}
train: {train_subdir}
val: {val_subdir}
"""
    
    with open(output_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created data.yaml at: {output_path}")


def prepare_coco_format_for_detr_florence(
    coco_json_path: str,
    images_src_dir: str,
    output_dir: str,
    output_json_name: str = "custom_annotations.json",
    copy_images: bool = True
) -> None:
    """
    Prepare COCO format data for DETR and Florence-2 models
    
    Simply organizes images into images/ folder and renames annotation file
    
    Args:
        coco_json_path: Path to COCO format JSON file
        images_src_dir: Source directory containing images
        output_dir: Output directory
        output_json_name: Name for output JSON file
        copy_images: Whether to copy images to output directory
    """
    coco_json_path = Path(coco_json_path)
    images_src_dir = Path(images_src_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    images_out_dir = output_dir / "images"
    images_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load COCO annotations
    print(f"Loading COCO annotations from: {coco_json_path}")
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Copy images if requested
    if copy_images:
        print(f"Copying images...")
        for img_info in tqdm(coco_data['images'], desc="Copying images"):
            file_name = img_info['file_name']
            src_path = images_src_dir / file_name
            
            if src_path.exists():
                dst_path = images_out_dir / file_name
                shutil.copy2(src_path, dst_path)
            else:
                print(f"Warning: Image not found: {file_name}")
    
    # Save annotation file with new name
    output_json_path = output_dir / output_json_name
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"Saved annotations to: {output_json_path}")
    print(f"Total images: {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")


if __name__ == "__main__":
    # Example usage
    print("YOLO Transform Utilities")
    print("Import this module to use conversion functions")

