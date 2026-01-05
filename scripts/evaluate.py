#!/usr/bin/env python
"""
Config-based evaluation script.

Evaluates trained models on test/validation datasets.
Supports checkpoint loading and comprehensive metrics calculation.

Examples:
    # Evaluate with config and checkpoint
    python scripts/evaluate.py --config configs/detr/detr_3class.yaml \
        --checkpoint exp/detr/TomatOD_COCO_3_20251231_165845/checkpoints/best-epoch=40-val_loss=1.42.ckpt \
        --split test
    
    # Evaluate on validation set
    python scripts/evaluate.py --config configs/detr/detr_3class.yaml \
        --checkpoint path/to/checkpoint.ckpt --split val
    
    # Save results to specific directory
    python scripts/evaluate.py --config configs/detr/detr_3class.yaml \
        --checkpoint path/to/checkpoint.ckpt --output-dir results/evaluation
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

from src.runner import EvaluationRunner
from src.registry import list_models, list_datasets
from src.utils.config import load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate object detection models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 필수 인자
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.ckpt file)')
    
    # 평가 설정
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate on')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for results (default: auto-generated)')
    
    # 평가 옵션
    parser.add_argument('--batch-size', type=int,
                       help='Batch size for evaluation (overrides config)')
    parser.add_argument('--score-threshold', type=float,
                       help='Score threshold for detections (overrides config)')
    parser.add_argument('--iou-threshold', type=float,
                       help='IoU threshold for TP/FP classification (overrides config)')
    
    # 디버그
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode (limited samples)')
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
    if args.batch_size:
        config.setdefault('evaluation', {})['batch_size'] = args.batch_size
    if args.score_threshold:
        config.setdefault('evaluation', {})['score_threshold'] = args.score_threshold
    if args.iou_threshold:
        config.setdefault('evaluation', {})['iou_threshold'] = args.iou_threshold
    if args.debug:
        config['debug'] = True
    
    # 체크포인트 존재 확인
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Split: {args.split}")
    
    # 평가 실행
    runner = EvaluationRunner(
        config=config,
        checkpoint_path=str(checkpoint_path),
        split=args.split,
        output_dir=args.output_dir
    )
    
    results = runner.run()
    
    return results


if __name__ == '__main__':
    main()