#!/usr/bin/env python
"""
Test/Inference script for object detection models.

This script is for running inference on test data and generating predictions.
It is separate from evaluation.py which focuses on metrics calculation.

For YOLO models:
    - Uses model.predict() instead of model.val()
    - Generates prediction images with bounding boxes
    - Saves predictions in YOLO format (.txt) and JSON
    - Results are saved to results/test_*/yolo_predictions/

Examples:
    # Test with YOLO model
    python scripts/test.py --config configs/yolov11/yolov11_3class.yaml \
        --checkpoint exp/yolov11/.../checkpoints/yolo_weights/best.pt \
        --split test
    
    # Test with custom output directory
    python scripts/test.py --config configs/yolov11/yolov11_3class.yaml \
        --checkpoint path/to/checkpoint.pt \
        --output-dir results/my_test_results
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
        description='Test/Inference for object detection models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 필수 인자
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pt or .ckpt file)')
    
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
    
    # 체크포인트 존재 확인
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Split: {args.split}")
    print(f"Mode: Test/Inference (predictions generation)")
    
    # 테스트 실행
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

