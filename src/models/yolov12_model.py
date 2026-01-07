"""YOLOv12 Lightning Module - Ultralytics YOLO 래퍼"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import pytorch_lightning as pl
from ultralytics import YOLO


class YOLOv12LightningModule(pl.LightningModule):
    """YOLOv12 Lightning Module (Ultralytics YOLO 래퍼)"""
    
    def __init__(
        self,
        model_size: str = "m",  # n, s, m, l, x
        pretrained_path: str = "yolo12m.pt",
        num_classes: int = 3,
        lr0: float = 0.001,
        lrf: float = 0.01,
        momentum: float = 0.937,
        weight_decay: float = 0.0005,
        box: float = 7.5,
        cls: float = 0.5,
        dfl: float = 1.5,
        data_yaml_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # YOLO 모델 초기화
        self.model = YOLO(pretrained_path)
        
        # 학습 완료 플래그
        self._training_completed = False
        self._results = None
    
    def setup_data_yaml(self, data_config: Dict[str, Any]):
        """YOLO data.yaml 파일 생성"""
        yolo_data = {
            'path': str(Path(data_config['data_root']).absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': data_config['num_classes'],
            'names': data_config['class_names']
        }
        
        data_yaml_path = Path(data_config['data_root']) / 'data.yaml'
        with open(data_yaml_path, 'w') as f:
            yaml.dump(yolo_data, f, default_flow_style=False)
        
        self.hparams.data_yaml_path = str(data_yaml_path)
        return data_yaml_path
    
    def build_train_args(self, trainer_config: Dict[str, Any], data_config: Dict[str, Any]):
        """YOLO 학습 인자 구성"""
        train_args = {
            # 데이터 설정
            'data': self.hparams.data_yaml_path,
            
            # 기본 학습 설정
            'epochs': trainer_config.get('max_epochs', 100),
            'batch': data_config.get('batch_size', 16),
            'imgsz': data_config.get('imgsz', 640),
            'device': 0,  # Lightning이 GPU 관리
            
            # 옵티마이저 설정
            'lr0': self.hparams.lr0,
            'lrf': self.hparams.lrf,
            'momentum': self.hparams.momentum,
            'weight_decay': self.hparams.weight_decay,
            
            # Loss 계수
            'box': self.hparams.box,
            'cls': self.hparams.cls,
            'dfl': self.hparams.dfl,
            
            # 학습 설정
            'patience': trainer_config.get('early_stopping', {}).get('patience', 20),
            'save': True,
            'save_period': 10,
            'cache': False,
            'val': True,
            'project': 'lightning_logs/yolo',
            'name': 'train',
            'exist_ok': True,
            'plots': True,
            'amp': True,
            
            # 데이터 증강 (기본값)
            'hsv_h': data_config.get('augmentation', {}).get('hsv_h', 0.015),
            'hsv_s': data_config.get('augmentation', {}).get('hsv_s', 0.7),
            'hsv_v': data_config.get('augmentation', {}).get('hsv_v', 0.4),
            'degrees': data_config.get('augmentation', {}).get('degrees', 0.0),
            'translate': data_config.get('augmentation', {}).get('translate', 0.1),
            'scale': data_config.get('augmentation', {}).get('scale', 0.5),
            'shear': data_config.get('augmentation', {}).get('shear', 0.0),
            'perspective': data_config.get('augmentation', {}).get('perspective', 0.0),
            'flipud': data_config.get('augmentation', {}).get('flipud', 0.0),
            'fliplr': data_config.get('augmentation', {}).get('fliplr', 0.5),
            'mosaic': data_config.get('augmentation', {}).get('mosaic', 1.0),
            'mixup': data_config.get('augmentation', {}).get('mixup', 0.0),
            
            # 기타 설정
            'seed': 42,
            'deterministic': True,
            'cos_lr': False,
            'close_mosaic': 10,
            'verbose': False,  # Lightning 로그와 충돌 방지
        }
        
        return train_args
    
    def training_step(self, batch, batch_idx):
        """Lightning training step - YOLO는 자체 학습 루프 사용"""
        return torch.tensor(0.0, requires_grad=True)
    
    def validation_step(self, batch, batch_idx):
        """Lightning validation step - YOLO는 자체 검증 사용"""
        return torch.tensor(0.0)
    
    def fit(self, trainer_config: Dict[str, Any], data_config: Dict[str, Any]):
        """YOLO 학습 실행 (Lightning 외부에서 호출)"""
        if not hasattr(self.hparams, 'data_yaml_path') or not self.hparams.data_yaml_path:
            self.setup_data_yaml(data_config)
        
        train_args = self.build_train_args(trainer_config, data_config)
        
        print("=" * 60)
        print("YOLO TRAINING STARTED")
        print("=" * 60)
        print(f"Model: YOLOv12 ({self.hparams.model_size})")
        print(f"Dataset: {data_config.get('dataset_name', 'Unknown')}")
        print(f"Epochs: {train_args['epochs']}")
        print(f"Batch size: {train_args['batch']}")
        print(f"Learning rate: {train_args['lr0']}")
        print("=" * 60)
        
        # TensorBoard 로깅을 위한 환경 설정
        import os
        
        # TensorBoard 활성화를 위한 환경 설정
        original_tensorboard = os.environ.get('TENSORBOARD', None)
        os.environ['TENSORBOARD'] = '1'
        
        print(f" TensorBoard will be saved to: {train_args['project']}/{train_args['name']}")
        
        try:
            # YOLO 학습 실행
            self._results = self.model.train(**train_args)
        finally:
            # 환경변수 복원
            if original_tensorboard is None:
                os.environ.pop('TENSORBOARD', None)
            else:
                os.environ['TENSORBOARD'] = original_tensorboard
        self._training_completed = True
        
        print("\nYOLO Training completed!")
        print(f"Results saved to: {self._results.save_dir}")
        
        # TensorBoard 로그 확인 및 생성
        self._setup_tensorboard_logs()
        
        return self._results
    
    def _setup_tensorboard_logs(self):
        """TensorBoard 로그 설정 및 생성"""
        if not self._results or not hasattr(self._results, 'save_dir'):
            return
            
        results_dir = Path(self._results.save_dir)
        
        # YOLO 결과에서 TensorBoard 로그 찾기
        tensorboard_dirs = list(results_dir.glob("**/events.out.tfevents.*"))
        
        if tensorboard_dirs:
            print(f"TensorBoard logs found: {len(tensorboard_dirs)} files")
            for tb_file in tensorboard_dirs:
                print(f"  - {tb_file}")
        else:
            print("  No TensorBoard logs found. Creating manual logs...")
            self._create_manual_tensorboard_logs()
    
    def _create_manual_tensorboard_logs(self):
        """수동으로 TensorBoard 로그 생성"""
        if not self._results or not hasattr(self._results, 'save_dir'):
            return
            
        results_dir = Path(self._results.save_dir)
        results_csv = results_dir / "results.csv"
        
        if not results_csv.exists():
            print("No results.csv found for TensorBoard conversion")
            return
            
        try:
            import pandas as pd
            from torch.utils.tensorboard import SummaryWriter
            
            # CSV 데이터 읽기
            df = pd.read_csv(results_csv)
            
            # TensorBoard 로그 디렉토리 생성
            tb_dir = results_dir / "tensorboard"
            tb_dir.mkdir(exist_ok=True)
            
            writer = SummaryWriter(str(tb_dir))
            
            # 메트릭 로깅
            for idx, row in df.iterrows():
                epoch = int(row.get('epoch', idx))
                
                # Loss 메트릭
                if 'train/box_loss' in row:
                    writer.add_scalar('Loss/train_box', row['train/box_loss'], epoch)
                if 'train/cls_loss' in row:
                    writer.add_scalar('Loss/train_cls', row['train/cls_loss'], epoch)
                if 'train/dfl_loss' in row:
                    writer.add_scalar('Loss/train_dfl', row['train/dfl_loss'], epoch)
                
                # Validation Loss
                if 'val/box_loss' in row:
                    writer.add_scalar('Loss/val_box', row['val/box_loss'], epoch)
                if 'val/cls_loss' in row:
                    writer.add_scalar('Loss/val_cls', row['val/cls_loss'], epoch)
                if 'val/dfl_loss' in row:
                    writer.add_scalar('Loss/val_dfl', row['val/dfl_loss'], epoch)
                
                # 메트릭
                if 'metrics/precision(B)' in row:
                    writer.add_scalar('Metrics/precision', row['metrics/precision(B)'], epoch)
                if 'metrics/recall(B)' in row:
                    writer.add_scalar('Metrics/recall', row['metrics/recall(B)'], epoch)
                if 'metrics/mAP50(B)' in row:
                    writer.add_scalar('Metrics/mAP50', row['metrics/mAP50(B)'], epoch)
                if 'metrics/mAP50-95(B)' in row:
                    writer.add_scalar('Metrics/mAP50-95', row['metrics/mAP50-95(B)'], epoch)
                
                # Learning Rate
                if 'lr/pg0' in row:
                    writer.add_scalar('Learning_Rate/pg0', row['lr/pg0'], epoch)
            
            writer.close()
            print(f"✅ Manual TensorBoard logs created: {tb_dir}")
            print(f"   View with: tensorboard --logdir {tb_dir}")
            
        except Exception as e:
            print(f"❌ Failed to create TensorBoard logs: {e}")
    
    def predict(self, source, **kwargs):
        """YOLO 예측"""
        return self.model.predict(source, **kwargs)
    
    def val(self, **kwargs):
        """YOLO 검증"""
        return self.model.val(**kwargs)
    
    def export(self, **kwargs):
        """YOLO 모델 내보내기"""
        return self.model.export(**kwargs)
    
    def configure_optimizers(self):
        """Lightning optimizer - YOLO는 자체 optimizer 사용"""
        # 더미 optimizer (실제로는 사용되지 않음)
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr0)
    
    @property
    def results(self):
        """학습 결과 반환"""
        return self._results
    
    @property
    def is_training_completed(self):
        """학습 완료 여부"""
        return self._training_completed


class YOLOv12Wrapper:
    """YOLOv12를 Lightning 없이 직접 사용하는 래퍼"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config['model']
        self.data_config = config['data']
        self.training_config = config.get('training', {})
        
        # YOLO 모델 초기화
        self.model = YOLO(self.model_config['pretrained_path'])
        
        # data.yaml 생성
        self.data_yaml_path = self._setup_data_yaml()
    
    def _setup_data_yaml(self):
        """YOLO data.yaml 파일 생성"""
        yolo_data = {
            'path': str(Path(self.data_config['data_root']).absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': self.data_config['num_classes'],
            'names': self.data_config['class_names']
        }
        
        data_yaml_path = Path(self.data_config['data_root']) / 'data.yaml'
        with open(data_yaml_path, 'w') as f:
            yaml.dump(yolo_data, f, default_flow_style=False)
        
        return data_yaml_path
    
    def train(self):
        """YOLO 학습 실행"""
        # project와 name 설정: runs 폴더를 생성하지 않도록 직접 경로 지정
        project = self.training_config.get('project', 'exp/yolo')
        name = self.training_config.get('name', 'train')
        
        # project가 설정되어 있으면 그대로 사용, 없으면 exp/yolo 사용
        # YOLO는 {project}/{name} 형태로 저장하므로, project를 최종 저장 경로의 부모로 설정
        train_args = {
            'data': str(self.data_yaml_path),
            'epochs': self.training_config.get('epochs', 100),
            'batch': self.data_config.get('batch_size', 16),
            'imgsz': self.data_config.get('imgsz', 640),
            'device': self.training_config.get('device', 0),
            'lr0': self.model_config.get('lr0', 0.001),
            'lrf': self.model_config.get('lrf', 0.01),
            'momentum': self.model_config.get('momentum', 0.937),
            'weight_decay': self.model_config.get('weight_decay', 0.0005),
            'box': self.model_config.get('box', 7.5),
            'cls': self.model_config.get('cls', 0.5),
            'dfl': self.model_config.get('dfl', 1.5),
            'patience': self.training_config.get('patience', 20),
            'project': project,  
            'name': name, 
            'exist_ok': self.training_config.get('exist_ok', False),
            'plots': self.training_config.get('plots', True),
            'amp': self.training_config.get('amp', True),
            'seed': self.config.get('experiment', {}).get('seed', 42),
            
            # TensorBoard 로깅 활성화
            'save_period': self.training_config.get('save_period', 10),  # 체크포인트 저장 주기
        }
        
        # 데이터 증강 설정 추가
        if 'augmentation' in self.data_config:
            aug_config = self.data_config['augmentation']
            train_args.update({
                'hsv_h': aug_config.get('hsv_h', 0.015),
                'hsv_s': aug_config.get('hsv_s', 0.7),
                'hsv_v': aug_config.get('hsv_v', 0.4),
                'degrees': aug_config.get('degrees', 0.0),
                'translate': aug_config.get('translate', 0.1),
                'scale': aug_config.get('scale', 0.5),
                'shear': aug_config.get('shear', 0.0),
                'perspective': aug_config.get('perspective', 0.0),
                'flipud': aug_config.get('flipud', 0.0),
                'fliplr': aug_config.get('fliplr', 0.5),
                'mosaic': aug_config.get('mosaic', 1.0),
                'mixup': aug_config.get('mixup', 0.0),
            })
        
        print("=" * 60)
        print("YOLO TRAINING STARTED")
        print("=" * 60)
        print(f"Experiment: {self.config.get('experiment', {}).get('name', 'yolo_exp')}")
        print(f"Model: {self.model_config.get('arch_name', 'yolov12')} ({self.model_config.get('model_size', 'm')})")
        print(f"Dataset: {self.data_config.get('dataset_name', 'Unknown')}")
        print(f"Epochs: {train_args['epochs']}")
        print(f"Batch size: {train_args['batch']}")
        print(f"Learning rate: {train_args['lr0']}")
        print("=" * 60)
        print(f"Results will be saved to: {project}/{name}")
        print(f"  TensorBoard: Enabled (real-time logging)")
        print(f"  Checkpoints: best.pt and last.pt only")
        
        # TensorBoard 활성화를 위한 환경 설정
        original_tensorboard = os.environ.get('TENSORBOARD', None)
        os.environ['TENSORBOARD'] = '1'
        
        import ultralytics
        if hasattr(ultralytics, 'settings'):
            ultralytics.settings.tensorboard = True
            
        try:
            # 학습 실행
            results = self.model.train(**train_args)
        finally:
            # 환경변수 복원
            if original_tensorboard is None:
                os.environ.pop('TENSORBOARD', None)
            else:
                os.environ['TENSORBOARD'] = original_tensorboard
        
        print("\nYOLO Training completed!")
        print(f"Results saved to: {results.save_dir}")
        
        # TensorBoard 로그 설정
        self._setup_tensorboard_logs(results)
        
        return results
    
    def _setup_tensorboard_logs(self, results):
        """TensorBoard 로그 설정"""
        if not results or not hasattr(results, 'save_dir'):
            return
            
        results_dir = Path(results.save_dir)
        results_csv = results_dir / "results.csv"
        
        if not results_csv.exists():
            print("No results.csv found for TensorBoard conversion")
            return
            
        try:
            import pandas as pd
            from torch.utils.tensorboard import SummaryWriter
            
            # CSV 데이터 읽기
            df = pd.read_csv(results_csv)
            
            # TensorBoard 로그 디렉토리 생성
            tb_dir = results_dir / "tensorboard"
            tb_dir.mkdir(exist_ok=True)
            
            writer = SummaryWriter(str(tb_dir))
            
            # 메트릭 로깅 (간소화된 버전)
            for idx, row in df.iterrows():
                epoch = int(row.get('epoch', idx))
                
                # 주요 메트릭만 로깅
                metrics_map = {
                    'train/box_loss': 'Loss/train_box',
                    'train/cls_loss': 'Loss/train_cls', 
                    'val/box_loss': 'Loss/val_box',
                    'val/cls_loss': 'Loss/val_cls',
                    'metrics/precision(B)': 'Metrics/precision',
                    'metrics/recall(B)': 'Metrics/recall',
                    'metrics/mAP50(B)': 'Metrics/mAP50',
                    'metrics/mAP50-95(B)': 'Metrics/mAP50-95',
                    'lr/pg0': 'Learning_Rate/pg0'
                }
                
                for csv_col, tb_name in metrics_map.items():
                    if csv_col in row and pd.notna(row[csv_col]):
                        writer.add_scalar(tb_name, row[csv_col], epoch)
            
            writer.close()
            print(f"✅ TensorBoard logs created: {tb_dir}")
            print(f"   View with: tensorboard --logdir {tb_dir}")
            
        except Exception as e:
            print(f"❌ Failed to create TensorBoard logs: {e}")
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, model_size='m', num_classes=3, lr=0.001, **kwargs):
        """체크포인트에서 모델 로드"""
        from ultralytics import YOLO
        
        # YOLO 모델 로드 (.pt 파일 직접 로드)
        model = YOLO(str(checkpoint_path))
        
        # 래퍼 객체 생성 (임시 config 사용)
        # 실제로는 config가 필요하지만, 평가 시에는 모델만 필요
        temp_config = {
            'model': {
                'pretrained_path': str(checkpoint_path),
                'model_size': model_size,
                'num_labels': num_classes,
            },
            'data': {
                'data_root': '',  # 평가 시 재설정됨
                'num_classes': num_classes,
                'class_names': [f'class_{i}' for i in range(num_classes)],
            },
            'training': {}
        }
        
        wrapper = cls(temp_config)
        wrapper.model = model  # 로드된 모델로 교체
        
        return wrapper
    
    def evaluate(self, **kwargs):
        """YOLO 평가"""
        eval_args = {
            'data': str(self.data_yaml_path),
            'batch': self.config.get('evaluation', {}).get('batch_size', 8),
            'conf': self.config.get('evaluation', {}).get('confidence_threshold', 0.5),
            'iou': self.config.get('evaluation', {}).get('iou_threshold', 0.5),
            'plots': True,
            'save_json': True,
        }
        eval_args.update(kwargs)
        
        return self.model.val(**eval_args)

