"""Training Runner - 실제 학습 로직"""
import sys
import shutil
from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import TensorBoardLogger

from ..registry import MODEL_REGISTRY, DATASET_REGISTRY
from ..utils.experiment import ExperimentManager


class Tee:
    """터미널 출력을 콘솔과 파일에 동시에 출력하는 클래스"""
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()


class SimpleEpochLogger(Callback):
    """Epoch 종료 시 한 줄로 로그를 출력하는 콜백"""
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Validation epoch 종료 시 한 줄로 로그 출력 (validation이 train 후에 실행되므로 여기서 출력)"""
        metrics = trainer.callback_metrics
        current_epoch = trainer.current_epoch
        max_epochs = trainer.max_epochs
        
        # 주요 메트릭 추출 (값이 tensor인 경우 item()으로 변환)
        def get_metric(key, default='N/A'):
            val = metrics.get(key, default)
            if hasattr(val, 'item'):
                return val.item()
            return val if val != default else default
        
        train_loss = get_metric('train_loss', get_metric('loss', 'N/A'))
        val_loss = get_metric('val_loss', 'N/A')
        val_map = get_metric('val_map', 'N/A')
        val_map_50 = get_metric('val_map_50', 'N/A')
        val_map_75 = get_metric('val_map_75', 'N/A')
        
        # 포맷팅 헬퍼 함수 (숫자면 포맷팅, 아니면 그대로)
        def format_value(val):
            if isinstance(val, (int, float)):
                return f"{val:.4f}"
            return str(val)
        
        # 한 줄로 출력
        if train_loss != 'N/A' and val_loss != 'N/A':
            log_line = (
                f"Epoch {current_epoch+1}/{max_epochs} | "
                f"train_loss: {format_value(train_loss)} | "
                f"val_loss: {format_value(val_loss)} | "
                f"val_map: {format_value(val_map)} | "
                f"val_map_50: {format_value(val_map_50)} | "
                f"val_map_75: {format_value(val_map_75)}"
            )
            print(log_line)
            sys.stdout.flush()


class TrainRunner:
    """학습 실행 클래스"""
    
    def __init__(self, config):
        self.config = config
        
        # Debug 모드 설정
        if config.get('debug'):
            config['trainer']['max_epochs'] = 2
            config['trainer']['limit_train_batches'] = 10
            config['trainer']['limit_val_batches'] = 5
            print("  DEBUG MODE: Limited to 2 epochs, 10 train batches, 5 val batches")
        
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
            
            # MODEL_REGISTRY에서 모델 찾기
            model_key = model_name if model_name in MODEL_REGISTRY else model_name.lower()
            if model_key not in MODEL_REGISTRY:
                raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")
            ModelClass = MODEL_REGISTRY[model_key]
            model = ModelClass(
                num_labels=self.config['model']['num_labels'],
                pretrained_path=self.config['model']['pretrained_path'],
                lr=self.config['model']['learning_rate'],
                lr_backbone=self.config['model']['lr_backbone'],
                weight_decay=self.config['model']['weight_decay'],
            )
            
        else:
            # YOLO 모델들은 _run_yolo_training()에서 처리되므로 여기서는 에러 발생
            model_name_lower = model_name.lower()
            if model_name_lower in ["yolov11", "yolov12", "yolo"]:
                raise RuntimeError(f"YOLO model '{model_name}' should be handled by _run_yolo_training(), not _create_model_and_data()")
            else:
                raise ValueError(f"Unknown model: {model_name}. Available models: DETR, YOLOv11, YOLOv12")
        
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
        
        # 한 줄 로그 출력 콜백 추가
        simple_logger = SimpleEpochLogger()
        
        return [checkpoint_callback, early_stop_callback, simple_logger]
    
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
            enable_progress_bar=trainer_config.get('enable_progress_bar', True),
            log_every_n_steps=trainer_config.get('log_every_n_steps', None),
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
        
        model_name = self.config['model']['arch_name']
        model_name_lower = model_name.lower()
        
        # YOLO 모델 (대소문자 구분 없이 체크)
        if model_name_lower in ["yolov11", "yolov12", "yolo"]:
            print("YOLO training mode: Using native YOLO weights (no Lightning checkpoints)")
            self._run_yolo_training()
        # Florence-2 모델
        elif model_name_lower in ["florence2", "florence-2", "florence_2"]:
            print("Florence-2 training mode: LoRA fine-tuning")
            self._run_florence2_training()
        # Grounding DINO 모델
        elif model_name_lower in ["groundingdino", "gdino", "grounding_dino"]:
            print("Grounding DINO training mode: Using official repository")
            self._run_gdino_training()
        else:
            # Lightning 기반 모델
            print(" Lightning training mode: Using PyTorch Lightning checkpoints")
            self._run_lightning_training()
    
    def _run_yolo_training(self):
        """YOLO 전용 학습 실행"""
        print("\nCreating YOLO model...")
        
        # YOLO 래퍼 사용 (대소문자 구분 없이)
        arch_name = self.config['model']['arch_name']
        model_key = arch_name if arch_name in MODEL_REGISTRY else arch_name.lower()
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {arch_name}. Available models: {list(MODEL_REGISTRY.keys())}")
        ModelClass = MODEL_REGISTRY[model_key]
        model = ModelClass(self.config)
        
        # YOLO 학습 실행
        print("Starting YOLO training...")
        print(f"YOLO results will be saved to: {self.exp_manager.exp_dir}")
        
        results = model.train(self.exp_manager.exp_dir)
        
        print(f"\n YOLO training completed!")
        print(f"   Results saved to: {results.save_dir}")
        
        # results 폴더 즉시 생성 (학습 중에도 사용 가능하도록)
        results_dir = self.exp_manager.exp_dir / "results"
        results_dir.mkdir(exist_ok=True)
        
        self._organize_yolo_results(results.save_dir)
        
        return results
    
    def _organize_yolo_results(self, yolo_save_dir: Path):
        """YOLO 학습 결과를 적절한 폴더로 정리 (YOLO 전용)"""
        import shutil
        yolo_dir = Path(yolo_save_dir)
        
        if not yolo_dir.exists():
            print(f"  YOLO results directory not found: {yolo_dir}")
            return
        
        print("\n Organizing YOLO results...")
        
        # results 폴더 생성 (YOLO 자동 생성 결과용)
        results_dir = self.exp_manager.exp_dir / "results"
        results_dir.mkdir(exist_ok=True)
        
        # 1. args.yaml -> exp_dir로 직접 이동 (config 폴더 없이)
        args_yaml = yolo_dir / "args.yaml"
        if args_yaml.exists():
            target = self.exp_manager.exp_dir / "yolo_args.yaml"
            args_yaml.rename(target)
            print(f"    Moved args.yaml to exp_dir/")
        
        # 2. weights/ -> checkpoints 폴더로 이동
        weights_dir = yolo_dir / "weights"
        if weights_dir.exists():
            target_weights = self.exp_manager.checkpoint_dir / "yolo_weights"
            if target_weights.exists():
                shutil.rmtree(target_weights)
            shutil.move(str(weights_dir), str(target_weights))
            print(f"    Moved weights/ to checkpoints/yolo_weights/")
        
        # 3. YOLO 자동 생성 결과들 -> results 폴더로 이동
        # 3-1. results 폴더가 있으면 그 안의 모든 내용 이동
        yolo_results_dir = yolo_dir / "results"
        if yolo_results_dir.exists():
            for item in yolo_results_dir.iterdir():
                target = results_dir / item.name
                if target.exists():
                    if target.is_file():
                        target.unlink()
                    else:
                        shutil.rmtree(target)
                shutil.move(str(item), str(target))
                print(f"    Moved {item.name} to results/")
        
        # 3-2. 루트에 있는 시각화 파일들 (*.png, *.jpg) -> results 폴더로 이동
        for ext in ['*.png', '*.jpg']:
            for img_file in yolo_dir.glob(ext):
                target = results_dir / img_file.name
                shutil.move(str(img_file), str(target))
                print(f"    Moved {img_file.name} to results/")
        
        # 4. results.csv -> results 폴더로 이동 (로그 파일로도 저장)
        results_csv = yolo_dir / "results.csv"
        if results_csv.exists():
            # results 폴더에 복사
            target_csv = results_dir / "yolo_results.csv"
            shutil.copy2(str(results_csv), str(target_csv))
            print(f"    Copied results.csv to results/yolo_results.csv")
        
        # 5. TensorBoard 로그 -> tensorboard 폴더로 이동
        tb_files = list(yolo_dir.glob("**/events.out.tfevents.*"))
        if tb_files:
            for tb_file in tb_files:
                target = self.exp_manager.tensorboard_dir / tb_file.name
                shutil.move(str(tb_file), str(target))
            print(f"    Moved TensorBoard logs to tensorboard/")
            
            # TensorBoard 하위 폴더가 비어있으면 삭제
            tb_subdir = yolo_dir / "tensorboard"
            if tb_subdir.exists():
                try:
                    if not any(tb_subdir.iterdir()):
                        tb_subdir.rmdir()
                        print(f"    Removed empty tensorboard/ subdirectory")
                except Exception as e:
                    print(f"    Could not remove tensorboard/ subdirectory: {e}")
        
        # 6. 빈 train 폴더 삭제 (모든 파일이 이동된 경우)
        if yolo_dir.parent == self.exp_manager.exp_dir:
            try:
                # 모든 파일과 폴더가 이동되었는지 확인
                remaining_items = list(yolo_dir.iterdir())
                if not remaining_items:
                    yolo_dir.rmdir()
                    print(f"    Removed empty {yolo_dir.name} directory")
                else:
                    print(f"      {yolo_dir.name} directory still contains: {[item.name for item in remaining_items]}")
            except OSError as e:
                # 폴더가 비어있지 않거나 권한 문제
                print(f"    Warning: Could not remove {yolo_dir.name} directory: {e}")
            except Exception as e:
                print(f"    Warning: Unexpected error removing {yolo_dir.name}: {e}")
        print(" Results organization completed!")

    def _run_gdino_training(self):
        """Grounding DINO 전용 학습 실행"""
        import sys
        import os
        from pathlib import Path
        
        # 로그 파일 비활성화 (log 폴더 생성 안 함)
        log_file = None
        
        print("\nStarting Grounding DINO training...")
        print(f"Results will be saved to: {self.exp_manager.exp_dir}")
        
        # Grounding DINO 래퍼 사용
        from ..models.gdino_model import GroundingDINOWrapper
        
        try:
                model = GroundingDINOWrapper(self.config)
                
                # 학습 실행
                results = model.train(
                    output_dir=self.exp_manager.exp_dir,
                    experiment_id=self.exp_manager.timestamp
                )
                
                # 결과 저장
                self.exp_manager.save_final_results(results)
                
                print(f"\n Training log saved to: {log_file}")
                print(f"\n Grounding DINO training completed!")
                print(f"   Best checkpoint: {results['best_checkpoint']}")
                print(f"   Best val loss: {results['best_val_loss']:.4f}")
                
                return results
                
        except Exception as e:
            # 에러 출력
            print(f"\n❌ Grounding DINO training failed: {e}\n")
            import traceback
            traceback.print_exc()
            raise

    def _run_lightning_training(self):
        """Lightning 기반 학습 실행 (DETR 등)"""
        import sys
        import os
        
        # 로그 파일 설정
        log_dir = self.exp_manager.exp_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "training.log"
        
        # 원본 stdout 저장
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        # 터미널 출력을 파일과 콘솔에 동시에 출력하도록 설정
        try:
            f = open(log_file, 'w', encoding='utf-8')
            sys.stdout = Tee(original_stdout, f)
            sys.stderr = Tee(original_stderr, f)
            log_file_handle = f
        except Exception as e:
            print(f"Warning: Could not create log file: {e}")
            log_file = None
            log_file_handle = None
        
        # 모델 & 데이터 생성
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
        
        # 학습 시작
        try:
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
            if log_file:
                print(f"  Training log: {log_file}")
            print("="*70 + "\n")
        finally:
            # 로그 파일 정리 (stdout/stderr 원래대로 복구)
            if log_file_handle:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                log_file_handle.close()
                if log_file:
                    original_stdout.write(f"\nTraining log saved to: {log_file}\n")
    
    def _run_florence2_training(self):
        """Florence-2 전용 fine-tuning 실행"""
        from ..models.florence2_finetuned import Florence2Finetuned
        
        print("\nStarting Florence-2 fine-tuning...")
        
        # Training parameters
        num_epochs = self.config['training'].get('epochs', 100)
        batch_size = self.config['data'].get('batch_size', 24)
        learning_rate = self.config['model'].get('learning_rate', 1e-5)
        weight_decay = self.config['model'].get('weight_decay', 0.0005)
        
        # LoRA parameters
        use_lora = self.config.get('florence2', {}).get('use_lora', True)
        lora_r = self.config.get('florence2', {}).get('lora_r', 8)
        lora_alpha = self.config.get('florence2', {}).get('lora_alpha', 8)
        
        # Advanced settings
        use_amp = self.config.get('training', {}).get('amp', True)
        gradient_clip_val = self.config.get('training', {}).get('gradient_clip_val', 1.0)
        accumulate_grad_batches = self.config.get('training', {}).get('accumulate_grad_batches', 1)  
        
        # Device
        device = self.config.get('device', 'cuda')
        
        # Output directory (checkpoints 폴더에 저장)
        checkpoint_dir = self.exp_manager.checkpoint_dir
        
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Weight decay: {weight_decay}")
        print(f"  AMP enabled: {use_amp}")
        print(f"  Gradient clip value: {gradient_clip_val}")      
        print(f"  Accumulate grad batches: {accumulate_grad_batches}") 
        print(f"  LoRA: {use_lora}")
        if use_lora:
            print(f"  LoRA rank: {lora_r}, alpha: {lora_alpha}")
        print(f"  Checkpoint dir: {checkpoint_dir}\n")
        
        # Run training
        Florence2Finetuned.train_model(
            config=self.config,
            output_dir=str(checkpoint_dir),
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            use_amp=use_amp,
            gradient_clip_val=gradient_clip_val,          
            accumulate_grad_batches=accumulate_grad_batches, 
            device=device
        )
        
        print(f"\n Florence-2 training completed!")
        print(f"  Checkpoints saved to: {checkpoint_dir}")
