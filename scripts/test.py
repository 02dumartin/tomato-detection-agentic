#!/usr/bin/env python
"""
Test script for evaluating object detection models.

Evaluates trained models on test/validation datasets.
Supports checkpoint loading and comprehensive metrics calculation.
Results are saved to exp/{model}/{exp_name}/test/

Examples:
    # Test with config and checkpoint
    python scripts/test.py --config configs/detr/detr_3class.yaml \
        --checkpoint exp/detr/TomatOD_COCO_3_20251231_165845/checkpoints/best-epoch=40-val_loss=1.42.ckpt \
        --split test
    
    # Test on validation set
    python scripts/test.py --config configs/detr/detr_3class.yaml \
        --checkpoint path/to/checkpoint.ckpt --split val
    
    # Save results to specific directory
    python scripts/test.py --config configs/detr/detr_3class.yaml \
        --checkpoint path/to/checkpoint.ckpt --output-dir exp/my_exp/test
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
    parser.add_argument('--checkpoint', type=str, required=False,
                       help='Path to model checkpoint (.ckpt or .pth file) - Not required for Florence-2 zero-shot')
    
    # 평가 설정
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate on')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for results (default: auto-generated)')
    parser.add_argument('--predictions', type=str,
                       help='Path to predictions.json file (if already generated)')
    
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
    
    # 체크포인트 존재 확인 (Florence-2 zero-shot은 제외)
    model_name = config.get('model', {}).get('arch_name', '').lower()
    florence2_mode = config.get('florence2', {}).get('mode', '')
    
    # Florence-2 zero-shot은 checkpoint 불필요
    skip_checkpoint_check = (model_name in ['florence2', 'florence-2'] and florence2_mode == 'zeroshot')
    
    if not skip_checkpoint_check:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for this model")
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print(f"Checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = args.checkpoint or 'none'
        print(f"Checkpoint: Not required (Florence-2 Zero-shot mode)")
    
    print(f"Split: {args.split}")
    
    # predictions 파일이 제공된 경우 직접 평가
    if args.predictions:
        print(f"\n  Using pre-generated predictions from: {args.predictions}")
        print("   Skipping model inference, only evaluating predictions...")
        results = evaluate_from_predictions(
            config=config,
            predictions_file=args.predictions,
            split=args.split,
            output_dir=args.output_dir or Path(args.predictions).parent / 'evaluation'
        )
    else:
        # 평가 실행 (모델 로드 + 추론 + 평가)
        runner = EvaluationRunner(
            config=config,
            checkpoint_path=str(checkpoint_path),
            split=args.split,
            output_dir=args.output_dir
        )
        results = runner.run()
    
    return results


def calculate_florence2_complexity(config):
    """
    Florence-2 모델의 complexity 계산
    
    Args:
        config: 설정
    
    Returns:
        dict: model_complexity 정보
    """
    try:
        import torch
        from src.models.florence2_base import Florence2Base
        
        print("\nCalculating model complexity...")
        
        # Florence-2 모델 로드 (CPU에서)
        model = Florence2Base(device='cpu')
        
        # 파라미터 수 계산
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        
        # MB 계산 (float32 기준: 4 bytes per parameter)
        model_size_mb = total_params * 4 / (1024 ** 2)
        
        # GFLOPs 계산 (대략적으로 추정)
        # Florence-2-base는 약 232M 파라미터, ~70 GFLOPs 정도
        # 실제 계산을 위해서는 fvcore나 thop 필요
        try:
            # 간단한 추정: 파라미터 수 기반
            gflops = total_params / 1e9 * 0.3  # 대략적 추정
        except:
            gflops = 0.0
        
        complexity = {
            'total_params': int(total_params),
            'trainable_params': int(trainable_params),
            'params_m': round(total_params / 1e6, 6),
            'model_size_mb': round(model_size_mb, 2),
            'gflops': round(gflops, 6),
            'gflops_formatted': f"{gflops:.3f}G"
        }
        
        print(f"  Total params: {complexity['params_m']:.2f}M")
        print(f"  Model size: {complexity['model_size_mb']:.2f} MB")
        print(f"  GFLOPs: {complexity['gflops_formatted']}")
        
        return complexity
        
    except Exception as e:
        print(f"  Warning: Could not calculate model complexity: {e}")
        # 기본값 반환
        return {
            'total_params': 0,
            'trainable_params': 0,
            'params_m': 0.0,
            'model_size_mb': 0.0,
            'gflops': 0.0,
            'gflops_formatted': "0.000G"
        }


def evaluate_from_predictions(config, predictions_file, split='test', output_dir=None):
    """
    이미 생성된 predictions.json 파일을 평가
    
    Args:
        config: 설정
        predictions_file: predictions.json 경로
        split: 데이터셋 split
        output_dir: 출력 디렉토리
    """
    import json
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    
    predictions_file = Path(predictions_file)
    if not predictions_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
    
    # 출력 디렉토리 설정
    if output_dir is None:
        # predictions.json 경로에서 exp 디렉토리 구조 추출
        parts = list(predictions_file.parent.parts)
        exp_idx = parts.index('exp') if 'exp' in parts else -1
        
        if exp_idx >= 0:
            # exp/{model}/{mode}/{exp_name}/visualization -> exp/{model}/{mode}/{exp_name}/evaluation
            # 또는 exp/{model}/{exp_name}/visualization -> exp/{model}/{exp_name}/evaluation
            
            # visualization 폴더 찾기
            viz_idx = -1
            for i, part in enumerate(parts):
                if part == 'visualization':
                    viz_idx = i
                    break
            
            if viz_idx > exp_idx:
                # visualization 이전까지가 exp_dir
                exp_dir = Path(*parts[:viz_idx])  # exp/{model}/{mode}/{exp_name} 또는 exp/{model}/{exp_name}
                output_dir = exp_dir / 'evaluation'
            else:
                # visualization이 없으면 부모 디렉토리의 evaluation
                output_dir = predictions_file.parent / 'evaluation'
        else:
            # exp가 없으면 부모 디렉토리의 evaluation
            output_dir = predictions_file.parent / 'evaluation'
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("EVALUATION FROM PREDICTIONS")
    print("="*70)
    print(f"Predictions file: {predictions_file}")
    print(f"Output directory: {output_dir}")
    print("="*70 + "\n")
    
    # Ground truth 로드
    from src.registry import DATASET_REGISTRY
    dataset_meta = DATASET_REGISTRY.get(config['data']['dataset_name'])
    if dataset_meta is None:
        raise ValueError(f"Dataset '{config['data']['dataset_name']}' not found")
    
    paths = dataset_meta.get_data_paths(split, config)
    gt_file = Path(paths['ann_file'])
    
    print(f"Loading ground truth from: {gt_file}")
    coco_gt = COCO(str(gt_file))
    
    # GT annotations에 iscrowd 필드 추가 (없는 경우)
    # COCO 객체의 dataset에 직접 추가
    if 'annotations' in coco_gt.dataset:
        for ann in coco_gt.dataset['annotations']:
            if 'iscrowd' not in ann:
                ann['iscrowd'] = 0
        # 인덱스 재생성
        coco_gt.createIndex()
    
    # Predictions 로드
    print(f"Loading predictions from: {predictions_file}")
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    print(f"  Total predictions: {len(predictions)}")
    print(f"  Total images: {len(set(p['image_id'] for p in predictions))}")
    
    # COCO 형식으로 변환
    # predictions를 COCO detection results로 변환
    # bbox 형식 변환 및 iscrowd 필드 추가
    predictions_coco_format = []
    for pred in predictions:
        pred_copy = {
            'image_id': pred['image_id'],
            'category_id': pred['category_id'],
            'score': pred.get('score', 1.0),
            'iscrowd': 0
        }
        
        # bbox 형식 확인 및 변환
        bbox = pred['bbox']
        if len(bbox) == 4:
            # [x1, y1, x2, y2] 형식인 경우 [x, y, width, height]로 변환
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
            else:
                # 이미 [x, y, width, height] 형식
                x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        else:
            x, y, w, h = 0, 0, 0, 0
        
        pred_copy['bbox'] = [x, y, w, h]  # COCO format: [x, y, width, height]
        predictions_coco_format.append(pred_copy)
    
    try:
        # 방법 1: 직접 loadRes 시도
        coco_dt = coco_gt.loadRes(predictions_coco_format)
    except (KeyError, AttributeError, TypeError) as e:
        # 방법 2: 수동으로 COCO 객체 생성
        print(f"   Warning: Using manual COCO creation due to: {e}")
        coco_dt = COCO()
        
        # GT의 기본 구조 복사
        coco_dt.dataset = {
            'images': coco_gt.dataset.get('images', []),
            'categories': coco_gt.dataset.get('categories', []),
            'annotations': []
        }
        
        # Predictions를 annotations 형식으로 변환 (이미 변환된 predictions_coco_format 사용)
        for i, pred in enumerate(predictions_coco_format):
            ann = {
                'id': i + 1,
                'image_id': pred['image_id'],
                'category_id': pred['category_id'],
                'bbox': pred['bbox'],  # 이미 COCO format: [x, y, width, height]
                'area': pred['bbox'][2] * pred['bbox'][3],  # width * height
                'score': pred.get('score', 1.0),
                'iscrowd': pred.get('iscrowd', 0)
            }
            coco_dt.dataset['annotations'].append(ann)
        
        coco_dt.createIndex()
    
    # COCO 평가
    print("\nComputing COCO metrics...")
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Precision, Recall 계산 (IoU=0.5 기준)
    # COCO의 precision/recall 값 사용
    precision = coco_eval.eval['precision'][0, :, :, 0, 2].mean()  # IoU=0.5, all classes, medium
    recall = coco_eval.eval['recall'][0, :, 0, 2].mean()  # IoU=0.5, all classes, medium
    
    # precision이나 recall이 -1이면 0으로 처리
    if precision == -1:
        precision = 0.0
    if recall == -1:
        recall = 0.0
    
    # F1 계산
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    # 클래스별 AP
    class_names = config['data'].get('class_names', ['fully_ripe', 'semi_ripe', 'unripe'])
    
    # 클래스별 AP@0.5와 AP@0.5:0.95 계산
    per_class_ap_50 = {}
    per_class_ap_50_95 = {}
    
    for cat_id, cat_name in enumerate(class_names):
        coco_eval_class = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval_class.params.catIds = [cat_id]
        coco_eval_class.evaluate()
        coco_eval_class.accumulate()
        coco_eval_class.summarize()
        
        per_class_ap_50[cat_name] = float(coco_eval_class.stats[1])  # AP@0.5
        per_class_ap_50_95[cat_name] = float(coco_eval_class.stats[0])  # AP@0.5:0.95
    
    # CA-mAP 계산 (Class-Agnostic mAP)
    print("\nComputing Class-Agnostic mAP (CA-mAP)...")
    ca_metrics = {}
    try:
        import copy
        import tempfile
        import os
        
        # 모든 예측의 category_id를 0으로 변경
        ca_predictions = []
        for pred in predictions_coco_format:
            ca_pred = pred.copy()
            ca_pred['category_id'] = 0  # 모든 클래스를 0으로 통합
            ca_predictions.append(ca_pred)
        
        print(f"  Converted {len(ca_predictions)} predictions to class-agnostic format")
        
        # GT의 모든 category_id를 0으로 변경한 복사본 생성
        ca_gt_dataset = copy.deepcopy(coco_gt.dataset)
        for ann in ca_gt_dataset['annotations']:
            ann['category_id'] = 0
        
        # 카테고리를 하나로 통일 (tomato)
        if len(ca_gt_dataset['categories']) > 0:
            ca_gt_dataset['categories'] = [{
                'id': 0,
                'name': 'tomato',
                'supercategory': 'none'
            }]
        
        # 임시 GT 파일 생성
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(ca_gt_dataset, f)
            temp_gt_file = f.name
        
        # CA COCO 객체 생성 및 평가
        ca_coco_gt = COCO(temp_gt_file)
        print(f"  GT annotations: {len(ca_coco_gt.anns)}")
        print(f"  GT images: {len(ca_coco_gt.imgs)}")
        print(f"  GT categories: {len(ca_coco_gt.cats)}")
        
        # CA predictions를 COCO 형식으로 변환
        # loadRes는 리스트를 받아야 함
        print(f"  Loading {len(ca_predictions)} CA predictions...")
        
        # 첫 번째 예측 확인
        if len(ca_predictions) > 0:
            print(f"  First CA prediction: {ca_predictions[0]}")
            print(f"  First CA prediction keys: {list(ca_predictions[0].keys())}")
        
        try:
            ca_coco_dt = ca_coco_gt.loadRes(ca_predictions)
            print(f"  DT predictions loaded: {len(ca_coco_dt.anns)}")
        except Exception as load_error:
            print(f"  Error loading predictions: {load_error}")
            # 수동으로 COCO 객체 생성 시도
            print(f"  Trying manual COCO creation...")
            ca_coco_dt = COCO()
            ca_coco_dt.dataset = {
                'images': ca_coco_gt.dataset.get('images', []),
                'categories': ca_coco_gt.dataset.get('categories', []),
                'annotations': []
            }
            # predictions를 annotations 형식으로 변환
            for i, pred in enumerate(ca_predictions):
                ann = {
                    'id': i + 1,
                    'image_id': pred['image_id'],
                    'category_id': pred['category_id'],
                    'bbox': pred['bbox'],
                    'area': pred['bbox'][2] * pred['bbox'][3],
                    'score': pred.get('score', 1.0),
                    'iscrowd': pred.get('iscrowd', 0)
                }
                ca_coco_dt.dataset['annotations'].append(ann)
            ca_coco_dt.createIndex()
            print(f"  DT predictions (manual): {len(ca_coco_dt.anns)}")
        
        # image_id 매칭 확인
        gt_img_ids = set(ca_coco_gt.imgs.keys())
        pred_img_ids = set(p['image_id'] for p in ca_predictions)
        matched_ids = gt_img_ids & pred_img_ids
        print(f"  GT image IDs: {len(gt_img_ids)}")
        print(f"  Prediction image IDs: {len(pred_img_ids)}")
        print(f"  Matched image IDs: {len(matched_ids)}")
        if len(matched_ids) == 0:
            print(f"  WARNING: No matching image IDs!")
            print(f"  Sample GT IDs: {list(gt_img_ids)[:5]}")
            print(f"  Sample Pred IDs: {list(pred_img_ids)[:5]}")
        
        ca_coco_eval = COCOeval(ca_coco_gt, ca_coco_dt, 'bbox')
        print(f"  Running COCOeval...")
        ca_coco_eval.evaluate()
        ca_coco_eval.accumulate()
        ca_coco_eval.summarize()
        
        print(f"  COCOeval stats: {ca_coco_eval.stats}")
        
        ca_metrics = {
            'ca_map': float(ca_coco_eval.stats[0]),  # CA-mAP@0.5:0.95
            'ca_map_50': float(ca_coco_eval.stats[1]),  # CA-mAP@0.5
            'ca_map_75': float(ca_coco_eval.stats[2]),  # CA-mAP@0.75
        }
        
        print(f"  CA-mAP@0.50:      {ca_metrics['ca_map_50']:.4f}")
        print(f"  CA-mAP@0.50:0.95: {ca_metrics['ca_map']:.4f}")
        print(f"  CA-mAP@0.75:      {ca_metrics['ca_map_75']:.4f}")
        
        # 임시 파일 삭제
        if os.path.exists(temp_gt_file):
            os.unlink(temp_gt_file)
            
    except Exception as e:
        import traceback
        print(f"  Warning: Could not calculate CA-mAP: {e}")
        traceback.print_exc()
        ca_metrics = {
            'ca_map': 0.0,
            'ca_map_50': 0.0,
            'ca_map_75': 0.0,
        }
    
    # Model complexity 계산 (Florence-2는 로드해서 계산)
    model_complexity = calculate_florence2_complexity(config)
    
    # 다른 모델과 동일한 형식으로 결과 정리
    results = {
        'detection_metrics': {
            'map_50': float(coco_eval.stats[1]),  # mAP@0.5
            'map': float(coco_eval.stats[0]),  # mAP@0.5:0.95
            'map_75': float(coco_eval.stats[2]),  # mAP@0.75
            'ca_map': ca_metrics.get('ca_map', 0.0),  # CA-mAP@0.5:0.95
            'ca_map_50': ca_metrics.get('ca_map_50', 0.0),  # CA-mAP@0.5
            'ca_map_75': ca_metrics.get('ca_map_75', 0.0),  # CA-mAP@0.75
        },
        'detailed_statistics': {
            'total_statistics': {
                'overall_precision': float(precision),
                'overall_recall': float(recall),
                'overall_f1': float(f1)
            },
            'per_class_ap': {
                'ap_50': per_class_ap_50,
                'ap_50_95': per_class_ap_50_95,
                'ap_75': {}  # 필요시 추가 가능
            }
        },
        'model_complexity': model_complexity,
        'model_info': {
            'backbone': 'florence-2-base',
            'arch_name': 'florence2',
            'mode': config.get('florence2', {}).get('mode', 'zeroshot')
        }
    }
    
    # 결과 출력
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"\n Detection Metrics:")
    print(f"   mAP@0.5:        {results['detection_metrics']['map_50']:.4f}")
    print(f"   mAP@0.75:       {results['detection_metrics']['map_75']:.4f}")
    print(f"   mAP@[0.5:0.95]: {results['detection_metrics']['map']:.4f}")
    
    print(f"\n Class-Agnostic mAP (CA-mAP):")
    print(f"   CA-mAP@0.5:        {results['detection_metrics']['ca_map_50']:.4f}")
    print(f"   CA-mAP@0.75:       {results['detection_metrics']['ca_map_75']:.4f}")
    print(f"   CA-mAP@[0.5:0.95]: {results['detection_metrics']['ca_map']:.4f}")
    
    print(f"\n Overall Statistics:")
    print(f"   Precision:      {results['detailed_statistics']['total_statistics']['overall_precision']:.4f}")
    print(f"   Recall:         {results['detailed_statistics']['total_statistics']['overall_recall']:.4f}")
    print(f"   F1 Score:       {results['detailed_statistics']['total_statistics']['overall_f1']:.4f}")
    
    print(f"\n Per-class AP@0.5:")
    for class_name, ap in results['detailed_statistics']['per_class_ap']['ap_50'].items():
        print(f"   {class_name:15s}: {ap:.4f}")
    
    print(f"\n Per-class AP@[0.5:0.95]:")
    for class_name, ap in results['detailed_statistics']['per_class_ap']['ap_50_95'].items():
        print(f"   {class_name:15s}: {ap:.4f}")
    
    print(f"\n Model Complexity:")
    print(f"   Total params:   {results['model_complexity']['params_m']:.2f}M")
    print(f"   Model size:     {results['model_complexity']['model_size_mb']:.2f} MB")
    print(f"   GFLOPs:         {results['model_complexity']['gflops_formatted']}")
    
    # 결과 저장
    results_file = output_dir / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n Evaluation results saved to: {results_file}")
    print("="*70 + "\n")
    
    return results


if __name__ == '__main__':
    main()