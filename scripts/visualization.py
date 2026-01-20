#!/usr/bin/env python
"""
Visualization script for object detection models.

This script generates visualization images with predictions and bounding boxes.

For YOLO models:
    - Uses model.predict() instead of model.val()
    - Generates prediction images with bounding boxes
    - Saves predictions in YOLO format (.txt) and JSON
    - Results are saved to exp/{model}/{exp_name}/visualization/{mode}/

Examples:
    # Visualize with YOLO model
    python scripts/visualization.py --config configs/yolov11/yolov11_3class.yaml \
        --checkpoint exp/yolov11/.../checkpoints/yolo_weights/best.pt \
        --split test
    
    # Visualize with custom output directory
    python scripts/visualization.py --config configs/yolov11/yolov11_3class.yaml \
        --checkpoint path/to/checkpoint.pt \
        --output-dir exp/my_exp/visualization/class
"""

import warnings
import logging

# 경고 및 로그 설정
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import os
import sys
from pathlib import Path
import argparse
import torch

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

torch.set_float32_matmul_precision('medium')

from src.runner import TestRunner
from src.utils.config import load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualization for object detection models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 필수 인자
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pt, .ckpt, or .pth file)')
    
    # 테스트 설정
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to test on (default: test)')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for results (default: auto-generated)')
    
    # 테스트 옵션
    parser.add_argument('--confidence-threshold', type=float,
                       help='Confidence threshold for detections (overrides config)')
    parser.add_argument('--iou-threshold', type=float,
                       help='IoU threshold for NMS (overrides config)')
    parser.add_argument('--box-color-mode', type=str, choices=['class', 'model'],
                       help='Bounding box color mode: "class" (class-specific colors) or "model" (model-specific color)')
    
    # 디버그
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Config 로드
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    # CLI 오버라이드
    if args.confidence_threshold:
        config.setdefault('evaluation', {})['score_threshold'] = args.confidence_threshold
        config.setdefault('model', {})['confidence_threshold'] = args.confidence_threshold
    if args.iou_threshold:
        config.setdefault('evaluation', {})['iou_threshold'] = args.iou_threshold
    if args.box_color_mode:
        config.setdefault('test', {})['box_color_mode'] = args.box_color_mode
    if args.debug:
        config['debug'] = True
    
    # 체크포인트 존재 확인 (Florence-2 zero-shot은 제외)
    model_name = config.get('model', {}).get('arch_name', '').lower()
    florence2_mode = config.get('florence2', {}).get('mode', '')
    
    # Florence-2 zero-shot은 checkpoint 불필요
    skip_checkpoint_check = (model_name in ['florence2', 'florence-2'] and florence2_mode == 'zeroshot')
    
    if not skip_checkpoint_check:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print(f"Checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = args.checkpoint
        print(f"Checkpoint: Not required (Florence-2 Zero-shot mode)")
    
    print(f"Split: {args.split}")
    print(f"Mode: Visualization (predictions generation)")
    
    # 시각화 실행
    runner = TestRunner(
        config=config,
        checkpoint_path=str(checkpoint_path),
        split=args.split,
        output_dir=args.output_dir
    )
    
    results = runner.run()
    
    return results


if __name__ == '__main__':
    main()

