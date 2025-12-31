"""Training Runner - 실제 학습 로직"""
import sys
from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from ..registry import MODEL_REGISTRY, DATASET_REGISTRY
from ..utils.experiment import ExperimentManager


class TrainRunner:
    """학습 실행 클래스"""
    
    def __init__(self, config):
        self.config = config
        
        # Debug 모드 설정
        if config.get('debug'):
            config['trainer']['max_epochs'] = 2
            config['trainer']['limit_train_batches'] = 10
            config['trainer']['limit_val_batches'] = 5
            print("⚠️  DEBUG MODE: Limited to 2 epochs, 10 train batches, 5 val batches")
        
        # Experiment Manager
        self.exp_manager = ExperimentManager(config)
        
        # Seed 설정
        pl.seed_everything(config.get('experiment', {}).get('seed', 42))
    
    def _create_model_and_data(self):
        """Config에 따라 모델과 데이터 생성"""
        model_name = self.config['model']['arch_name']
        dataset_name = self.config['data']['dataset_name']
        
        # Dataset 메타정보
        dataset_meta = DATASET_REGISTRY.get(dataset_name)
        
        # 모델별 처리
        if model_name == "DETR" or model_name == "detr":
            from transformers import DetrImageProcessor
            from ..data.transforms.detr_transform import (
                create_detr_dataset, 
                DetrCocoDataset
            )
            
            imageprocessor = DetrImageProcessor.from_pretrained(
                self.config['model']['pretrained_path']
            )
            
            train_dataset = create_detr_dataset(dataset_meta, "train", imageprocessor, self.config)
            val_dataset = create_detr_dataset(dataset_meta, "val", imageprocessor, self.config)
            collate_fn = DetrCocoDataset.create_collate_fn(imageprocessor)
            
            ModelClass = MODEL_REGISTRY[model_name]
            model = ModelClass(
                num_labels=self.config['model']['num_labels'],
                pretrained_path=self.config['model']['pretrained_path'],
                lr=self.config['model']['learning_rate'],
                lr_backbone=self.config['model']['lr_backbone'],
                weight_decay=self.config['model']['weight_decay'],
            )
        
        elif model_name in ["YOLOv11", "YOLOv12"]:
            from ..data.transforms.yolo_transform import create_yolo_dataset
            
            train_dataset = create_yolo_dataset(
                dataset_meta, "train", self.config['data']['image_size']
            )
            val_dataset = create_yolo_dataset(
                dataset_meta, "val", self.config['data']['image_size']
            )
            collate_fn = None
            
            ModelClass = MODEL_REGISTRY[model_name]
            model = ModelClass(
                model_size=self.config['model']['model_size'],
                num_classes=self.config['model']['num_labels'],
                lr=self.config['model']['learning_rate'],
            )
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        return model, train_dataset, val_dataset, collate_fn
    
    def _create_dataloaders(self, train_dataset, val_dataset, collate_fn):
        """DataLoader 생성"""
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data'].get('num_workers', 0),
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data'].get('num_workers', 0),
            collate_fn=collate_fn
        )
        
        return train_loader, val_loader
    
    def _create_callbacks(self):
        """Callbacks 생성"""
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(self.exp_manager.checkpoint_dir),
            filename='best-{epoch:02d}-{val_loss:.2f}',
            save_top_k=self.config['trainer']['checkpoint']['save_top_k'],
            monitor=self.config['trainer']['checkpoint']['monitor'],
            mode=self.config['trainer']['checkpoint']['mode'],
            save_last=self.config['trainer']['checkpoint']['save_last']
        )
        
        early_stop_callback = EarlyStopping(
            monitor=self.config['trainer']['early_stopping']['monitor'],
            patience=self.config['trainer']['early_stopping']['patience'],
            mode=self.config['trainer']['early_stopping']['mode']
        )
        
        return [checkpoint_callback, early_stop_callback]
    
    def _create_logger(self):
        """Logger 생성"""
        return TensorBoardLogger(
            save_dir=str(self.exp_manager.tensorboard_dir),
            name="",
            version=""
        )
    
    def _create_trainer(self, callbacks, logger):
        """Trainer 생성"""
        trainer_config = self.config['trainer']
        
        trainer = pl.Trainer(
            max_epochs=trainer_config['max_epochs'],
            accelerator=trainer_config.get('accelerator', 'gpu'),
            devices=trainer_config.get('devices', 1),
            precision=trainer_config.get('precision', '16-mixed'),
            accumulate_grad_batches=trainer_config.get('accumulate_grad_batches', 1),
            gradient_clip_val=trainer_config.get('gradient_clip_val', None),
            callbacks=callbacks,
            logger=logger,
            # Debug 모드용
            limit_train_batches=trainer_config.get('limit_train_batches', 1.0),
            limit_val_batches=trainer_config.get('limit_val_batches', 1.0),
        )
        
        return trainer
    
    def run(self):
        """학습 실행"""
        print("\n" + "="*70)
        print("TRAINING STARTED")
        print("="*70)
        
        # 1. 실험 정보 출력
        self.exp_manager.print_info()
        
        print(f"Experiment ID: {self.exp_manager.timestamp}")
        sys.stdout.flush()
        
        # 2. 모델 & 데이터 생성
        print("\nCreating model and datasets...")
        model, train_dataset, val_dataset, collate_fn = self._create_model_and_data()
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")
        
        # 3. DataLoader 생성
        print("\nCreating dataloaders...")
        train_loader, val_loader = self._create_dataloaders(
            train_dataset, val_dataset, collate_fn
        )
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        
        # 4. Callbacks & Logger
        callbacks = self._create_callbacks()
        logger = self._create_logger()
        
        # 5. Trainer 생성
        trainer = self._create_trainer(callbacks, logger)
        
        # 6. Resume 처리
        resume_ckpt = self.config.get('resume')
        if resume_ckpt:
            print(f"\nResuming from: {resume_ckpt}")
        
        # 7. 학습 시작
        print("\n" + "="*70)
        print("TRAINING IN PROGRESS...")
        print("="*70 + "\n")
        
        trainer.fit(model, train_loader, val_loader, ckpt_path=resume_ckpt)
        
        # 8. 결과 저장
        checkpoint_callback = callbacks[0]  # ModelCheckpoint
        results = {
            'best_checkpoint': str(checkpoint_callback.best_model_path),
            'best_val_loss': float(checkpoint_callback.best_model_score) 
                            if checkpoint_callback.best_model_score else None,
        }
        self.exp_manager.save_final_results(results)
        
        # 9. 완료 메시지
        print("\n" + "="*70)
        print("TRAINING COMPLETED!")
        print("="*70)
        print(f"  Experiment: {self.exp_manager.exp_dir}")
        print(f"  Best checkpoint: {checkpoint_callback.best_model_path}")
        print("="*70 + "\n")