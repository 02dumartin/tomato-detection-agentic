#!/usr/bin/env python
"""
Data preparation script for Tomato_merge dataset

This script converts the merged_dataset to the required formats:
1. YOLO format (for YOLOv11 and YOLOv12)
2. COCO format with proper structure (for DETR and Florence-2)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.transforms.yolo_transform import (
    convert_coco_to_yolo,
    create_yolo_data_yaml,
    prepare_coco_format_for_detr_florence
)


def main():
    """Main data preparation workflow"""
    
    # Paths
    source_root = Path("/home/hyeonjin/tomato-detection-agentic/data/Tomato_merge")
    target_root = Path("/home/hyeonjin/tomato-detection-agentic/data/Tomato_merge_1")
    
    print("="*70)
    print("TOMATO_MERGE DATA PREPARATION")
    print("="*70)
    print(f"Source: {source_root}")
    print(f"Target: {target_root}")
    print("="*70)
    
    # Class mapping (merged_dataset has category_id=0 for tomato)
    class_mapping = {0: 0}  # Source category_id 0 -> YOLO class_id 0
    class_names = ["tomato"]
    
    # Process each split
    for split in ["train", "valid", "test"]:
        print(f"\n{'='*70}")
        print(f"Processing {split} split...")
        print(f"{'='*70}")
        
        # Handle split name difference (valid -> val)
        source_split = split
        target_split = "val" if split == "valid" else split
        
        # Paths for this split
        coco_json = source_root / source_split / "_annotations.coco.json"
        images_src = source_root / source_split
        
        # Check if source files exist
        if not coco_json.exists():
            print(f"Warning: {coco_json} not found, skipping {split}")
            continue
        
        # 1. Convert to YOLO format
        print(f"\n[1/2] Converting to YOLO format...")
        yolo_output = target_root / target_split
        total_imgs, total_anns = convert_coco_to_yolo(
            coco_json_path=str(coco_json),
            images_src_dir=str(images_src),
            output_dir=str(yolo_output),
            class_mapping=class_mapping,
            copy_images=True
        )
        
        print(f"✓ YOLO conversion complete: {total_imgs} images, {total_anns} annotations")
        
        # 2. Prepare COCO format for DETR/Florence-2
        print(f"\n[2/2] Preparing COCO format for DETR/Florence-2...")
        output_json_name = f"custom_{target_split}.json"
        prepare_coco_format_for_detr_florence(
            coco_json_path=str(coco_json),
            images_src_dir=str(images_src),
            output_dir=str(yolo_output),  # Same directory
            output_json_name=output_json_name,
            copy_images=False  # Already copied in step 1
        )
        
        print(f"✓ COCO format prepared: {output_json_name}")
    
    # 3. Create YOLO data.yaml
    print(f"\n{'='*70}")
    print("Creating YOLO data.yaml configuration...")
    print(f"{'='*70}")
    
    data_yaml_path = target_root / "data.yaml"
    create_yolo_data_yaml(
        output_path=str(data_yaml_path),
        data_root=str(target_root.absolute()),
        class_names=class_names,
        train_subdir="train/images",
        val_subdir="val/images",
        test_subdir="test/images"
    )
    
    print(f"✓ Created: {data_yaml_path}")
    
    # Summary
    print(f"\n{'='*70}")
    print("DATA PREPARATION COMPLETE!")
    print(f"{'='*70}")
    print(f"\nDataset location: {target_root}")
    print(f"\nDirectory structure:")
    print(f"  {target_root}/")
    print(f"    ├── train/")
    print(f"    │   ├── images/         (YOLO & DETR/Florence-2)")
    print(f"    │   ├── labels/         (YOLO)")
    print(f"    │   └── custom_train.json (DETR/Florence-2)")
    print(f"    ├── val/")
    print(f"    │   ├── images/")
    print(f"    │   ├── labels/")
    print(f"    │   └── custom_val.json")
    print(f"    ├── test/")
    print(f"    │   ├── images/")
    print(f"    │   ├── labels/")
    print(f"    │   └── custom_test.json")
    print(f"    └── data.yaml          (YOLO)")
    print(f"\nReady for training!")
    print(f"  - YOLOv11: python scripts/train.py --config configs/yolov11/yolov11_tomato_merge.yaml")
    print(f"  - YOLOv12: python scripts/train.py --config configs/yolov12/yolov12_tomato_merge.yaml")
    print(f"  - DETR:    python scripts/train.py --config configs/detr/detr_tomato_merge.yaml")
    print(f"  - Florence-2: python scripts/train.py --config configs/florence2/florence_tomato_merge_finetune.yaml")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

