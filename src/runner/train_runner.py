"""Training Runner - 실제 학습 로직"""
import sys
import shutil
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
            
            # MODEL_REGISTRY에서 모델 찾기 (대소문자 구분 없이)
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
            print(" YOLO training mode: Using native YOLO weights (no Lightning checkpoints)")
            self._run_yolo_training()
        else:
            # Lightning 기반 모델
            print(" Lightning training mode: Using PyTorch Lightning checkpoints")
            self._run_lightning_training()
    
    def _run_yolo_training(self):
        """YOLO 전용 학습 실행"""
        import sys
        
        # 로그 파일 설정
        log_file = self.exp_manager.log_dir / f"train_{self.exp_manager.timestamp}.log"
        
        print("\nCreating YOLO model...")
        print(f"Log file: {log_file}")

        modified_config = self.config.copy()
        modified_config['training'] = self.config['training'].copy()
        modified_config['training']['project'] = str(self.exp_manager.exp_dir)
        modified_config['training']['name'] = ''
        modified_config['training']['exist_ok'] = True
        
        # YOLO 래퍼 사용 (대소문자 구분 없이)
        arch_name = self.config['model']['arch_name']
        model_key = arch_name if arch_name in MODEL_REGISTRY else arch_name.lower()
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {arch_name}. Available models: {list(MODEL_REGISTRY.keys())}")
        ModelClass = MODEL_REGISTRY[model_key]
        model = ModelClass(modified_config)
        
        # YOLO 학습 실행
        print("Starting YOLO training...")
        print(f"YOLO results will be saved to: {self.exp_manager.exp_dir}/yolo_training")
        print(f"  (runs folder will NOT be created)")
        
        # 로그 파일에 출력 저장 (터미널에도 출력)
        import re
        
        def remove_ansi_codes(text):
            """ANSI escape 코드 제거"""
            # ANSI escape sequence 제거 (색상, 커서 제어 등)
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            return ansi_escape.sub('', text)
        
        with open(log_file, 'w', encoding='utf-8', buffering=1) as f:
            class Tee:
                def __init__(self, *files):
                    self.files = files
                    # 첫 번째 파일(터미널)의 속성들을 상속
                    if files:
                        self._original = files[0]
                
                def write(self, obj):
                    for i, file in enumerate(self.files):
                        if i == 0:
                            # 첫 번째 파일(터미널)에는 원본 출력
                        file.write(obj)
                        else:
                            # 나머지 파일(로그 파일)에는 ANSI 코드 제거
                            cleaned = remove_ansi_codes(obj)
                            file.write(cleaned)
                        # 매번 flush하여 실시간 기록
                        file.flush()
                
                def flush(self):
                    for file in self.files:
                        file.flush()
                
                def isatty(self):
                    # PyTorch Lightning이 터미널인지 확인할 때 사용
                    return self._original.isatty() if hasattr(self._original, 'isatty') else False
                
                def __getattr__(self, name):
                    # 다른 속성들은 원본에서 가져오기
                    return getattr(self._original, name) if hasattr(self._original, name) else None
            
            # stdout과 stderr를 파일과 터미널 둘 다에 출력하도록 설정
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            
            try:
                sys.stdout = Tee(original_stdout, f)
                sys.stderr = Tee(original_stderr, f)
                
                # Python의 stdout 버퍼링 비활성화
                import os
                os.environ['PYTHONUNBUFFERED'] = '1'
                
                results = model.train()
            finally:
                # 원래 stdout/stderr로 복원
                sys.stdout = original_stdout
                sys.stderr = original_stderr
        
        print(f"\n Training log saved to: {log_file}")
        
        print(f"\n YOLO training completed!")
        print(f"   Results saved to: {results.save_dir}")
        
        self._organize_yolo_results(results.save_dir)
        
        return results
    
    def _organize_yolo_results(self, yolo_save_dir: Path):
        """YOLO 학습 결과를 적절한 폴더로 정리"""
        yolo_dir = Path(yolo_save_dir)
        
        if not yolo_dir.exists():
            print(f"  YOLO results directory not found: {yolo_dir}")
            return
        
        print("\n Organizing YOLO results...")
        
        is_exp_dir = yolo_dir == self.exp_manager.exp_dir

        # 1. args.yaml -> config 폴더로 이동
        args_yaml = yolo_dir / "args.yaml"
        if args_yaml.exists():
            target = self.exp_manager.config_dir / "yolo_args.yaml"
            args_yaml.rename(target)
            print(f"    Moved args.yaml to config/")
        
        # 2. weights/ -> checkpoints 폴더로 이동
        weights_dir = yolo_dir / "weights"
        if weights_dir.exists():
            import shutil
            target_weights = self.exp_manager.checkpoint_dir / "yolo_weights"
            if target_weights.exists():
                shutil.rmtree(target_weights)
            shutil.move(str(weights_dir), str(target_weights))
            print(f"    Moved weights/ to checkpoints/yolo_weights/")
        
        # 3. 시각화 파일들 (*.png, *.jpg) -> results 폴더로 이동
        import shutil
        for ext in ['*.png', '*.jpg']:
            for img_file in yolo_dir.glob(ext):
                target = self.exp_manager.results_dir / img_file.name
                shutil.move(str(img_file), str(target))
                print(f"    Moved {img_file.name} to results/")
        
        # 4. results.csv -> results 폴더로 이동
        results_csv = yolo_dir / "results.csv"
        if results_csv.exists():
            target = self.exp_manager.results_dir / "yolo_results.csv"
            results_csv.rename(target)
            print(f"    Moved results.csv to results/")
        
        # 5. TensorBoard 로그 -> tensorboard 폴더로 이동
        tb_files = list(yolo_dir.glob("**/events.out.tfevents.*"))
        if tb_files:
            import shutil
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


    def _run_lightning_training(self):
        """Lightning 기반 학습 실행 (DETR 등)"""
        import sys
        import os
        
        # 1. 로그 파일 설정
        log_file = self.exp_manager.log_dir / f"train_{self.exp_manager.timestamp}.log"
        
        # 2. 모델 & 데이터 생성
        print("\nCreating model and datasets...")
        print(f"Log file: {log_file}")
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
        
        # 7. 학습 시작 (로그 파일에 저장)
        print("\n" + "="*70)
        print("TRAINING IN PROGRESS...")
        print("="*70 + "\n")
        
        # 로그 파일 열기 (unbuffered 모드)
        log_file_handle = open(log_file, 'w', encoding='utf-8', buffering=1)
        
        # Queue를 사용한 로그 기록 (별도 스레드)
        from threading import Thread
        from queue import Queue
        import re
        
        log_queue = Queue()
        log_running = True
        
        def remove_ansi_codes(text):
            """ANSI escape code 제거"""
            # ANSI escape sequence 제거 (색상, 커서 제어 등)
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            return ansi_escape.sub('', text)
        
        def log_writer():
            """별도 스레드에서 로그 파일에 기록 (ANSI 코드 제거)"""
            while log_running or not log_queue.empty():
                try:
                    line = log_queue.get(timeout=0.1)
                    # ANSI escape code 제거 후 파일에 기록
                    clean_line = remove_ansi_codes(line)
                    log_file_handle.write(clean_line)
                    log_file_handle.flush()
                except:
                    continue
        
        # 로그 기록 스레드 시작
        log_thread = Thread(target=log_writer, daemon=True)
        log_thread.start()
        
        # Tee 클래스: 터미널에 출력하고 로그 큐에 추가
        class Tee:
            def __init__(self, original, log_queue):
                self.original = original
                self.log_queue = log_queue
            
            def write(self, obj):
                # 터미널에 원본 출력 (색상 코드 포함)
                self.original.write(obj)
                self.original.flush()
                # 로그 큐에 추가 (나중에 ANSI 코드 제거됨)
                if obj:
                    self.log_queue.put(obj)
            
            def flush(self):
                self.original.flush()
            
            def isatty(self):
                return self.original.isatty() if hasattr(self.original, 'isatty') else False
            
            def __getattr__(self, name):
                return getattr(self.original, name)
        
        # stdout과 stderr를 Tee로 교체
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        try:
            sys.stdout = Tee(original_stdout, log_queue)
            sys.stderr = Tee(original_stderr, log_queue)
            
            # Python의 stdout 버퍼링 비활성화
            os.environ['PYTHONUNBUFFERED'] = '1'
            
            trainer.fit(model, train_loader, val_loader, ckpt_path=resume_ckpt)
        finally:
            # 원래 stdout/stderr로 복원
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            # 로그 기록 중지
            log_running = False
            log_thread.join(timeout=2)
            log_file_handle.close()
        
        print(f"\n✅ Training log saved to: {log_file}")
        
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