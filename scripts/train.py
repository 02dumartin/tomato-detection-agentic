#!/usr/bin/env python
"""
Config-based training script.

Supports both YAML config files and CLI arguments.
CLI arguments override config values.

Examples:
    # Use config file only
    python scripts/train.py --config configs/DETR/detr_3class.yaml
    
    # Override with CLI arguments
    python scripts/train.py --config configs/DETR/detr_3class.yaml \
        --batch-size 16 --lr 0.0001
    
    # CLI only (uses default config)
    python scripts/train.py --model DETR --data TomatOD_3
    
    # Debug mode
    python scripts/train.py --config configs/DETR/detr_3class.yaml --debug
"""

import warnings
import logging

# 모든 경고 무시
warnings.filterwarnings("ignore")

# 로그 레벨 설정
logging.basicConfig(level=logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import os
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch

torch.set_float32_matmul_precision('medium')

from src.runner import TrainRunner
from src.registry import list_models, list_datasets
from src.utils.config import load_config, merge_config_with_args, get_default_config_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train object detection models with YAML configs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Config file (primary method)
    parser.add_argument('--config', type=str,
                       help='Path to YAML config file')
    
    # Model/Data (required if no config)
    parser.add_argument('--model', type=str,
                       help=f'Model name (overrides config). Available: {", ".join(list_models())}')
    parser.add_argument('--data', type=str,
                       help=f'Dataset name (overrides config). Available: {", ".join(list_datasets())}')
    
    # Training hyperparameters (override config)
    parser.add_argument('--epochs', type=int,
                       help='Max epochs')
    parser.add_argument('--batch-size', type=int,
                       help='Batch size')
    parser.add_argument('--lr', type=float,
                       help='Learning rate')
    parser.add_argument('--gpu', type=int,
                       help='GPU device ID')
    
    # Experiment control
    parser.add_argument('--tag', type=str,
                       help='Experiment tag')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode (2 epochs, limited batches)')
    parser.add_argument('--resume', type=str,
                       help='Resume from checkpoint')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load config
    if args.config:
        # Load from specified file
        print(f"Loading config from: {args.config}")
        config = load_config(args.config)
    elif args.model and args.data:
        # Try to find default config
        default_config_path = get_default_config_path(args.model, args.data)
        if default_config_path and default_config_path.exists():
            print(f"Using default config: {default_config_path}")
            config = load_config(str(default_config_path))
        else:
            # Build minimal config from CLI
            print("No config file found, using CLI arguments only")
            config = {
                'model': {'arch_name': args.model},
                'data': {'dataset_name': args.data},
                'trainer': {}
            }
    else:
        parser.error("Either --config or both --model and --data are required")

    # Merge CLI overrides
    config = merge_config_with_args(config, args)
    
    # Print final config
    print("\n" + "="*60)
    print("Final Configuration:")
    print("="*60)
    print(f"Model: {config['model']['arch_name']}")
    print(f"Dataset: {config['data']['dataset_name']}")
    print(f"Batch size: {config['data'].get('batch_size', 'N/A')}")
    print(f"Learning rate: {config['model'].get('learning_rate', 'N/A')}")
    print(f"Max epochs: {config['trainer'].get('max_epochs', 'N/A')}")
    if config.get('debug'):
        print("⚠️  DEBUG MODE ENABLED")
    print("="*60 + "\n")
    
    # Create runner and train
    runner = TrainRunner(config)
    runner.run()


if __name__ == '__main__':
    main()