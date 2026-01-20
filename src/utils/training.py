"""학습 관리 통합 모듈 - 로깅 및 체크포인트 관리"""
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    """
    통합 학습 로깅 시스템
    - TensorBoard 로깅
    - CSV 메트릭 저장
    - 텍스트 로그 파일
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        config: Dict[str, Any],
        log_file_name: Optional[str] = None
    ):
        """
        Args:
            output_dir: 출력 디렉토리
            config: 설정 딕셔너리
            log_file_name: 로그 파일명 (None이면 자동 생성)
        """
        self.output_dir = Path(output_dir)
        self.config = config
        self.logging_config = config.get('logging', {})
        
        # 로깅 활성화 여부 (bool 또는 dict 모두 처리)
        tensorboard_config = self.logging_config.get('tensorboard', True)
        if isinstance(tensorboard_config, bool):
            self.tensorboard_enabled = tensorboard_config
        elif isinstance(tensorboard_config, dict):
            self.tensorboard_enabled = tensorboard_config.get('enabled', True)
        else:
            self.tensorboard_enabled = True
        
        csv_config = self.logging_config.get('csv', True)
        if isinstance(csv_config, bool):
            self.csv_enabled = csv_config
        elif isinstance(csv_config, dict):
            self.csv_enabled = csv_config.get('enabled', True)
        else:
            self.csv_enabled = True
        
        self.log_file_enabled = self.logging_config.get('save_log_file', True)
        
        # TensorBoard 설정
        self.writer = None
        if self.tensorboard_enabled:
            # 실험 폴더 내의 tensorboard 디렉토리 사용
            tb_dir = self.output_dir / "tensorboard"
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(tb_dir), flush_secs=10)
            print(f"TensorBoard logging enabled: {tb_dir}")
        
        # CSV 파일 설정
        self.csv_file = None
        if self.csv_enabled:
            self.csv_file = self.output_dir / "training_metrics.csv"
            with open(self.csv_file, 'w') as f:
                f.write("epoch,train_loss,val_loss,learning_rate,patience_counter\n")
            print(f"CSV metrics: {self.csv_file}")
        
        # 로그 파일 설정
        self.log_file = None
        if self.log_file_enabled:
            if log_file_name is None:
                log_file_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            self.log_file = self.output_dir / log_file_name
            print(f"Log file: {self.log_file}")
    
    def log_scalar(self, tag: str, value: float, step: int, flush: bool = False):
        """TensorBoard에 스칼라 로깅"""
        if self.tensorboard_enabled and self.writer:
            self.writer.add_scalar(tag, value, step)
            if flush:
                self.writer.flush()
    
    def log_losses(
        self,
        epoch: int,
        train_loss: float = None,
        val_loss: float = None,
        learning_rate: float = None,
        patience_counter: int = None,
        flush: bool = False
    ):
        """
        Loss 및 관련 메트릭 로깅 (통합)
        
        Args:
            epoch: 현재 epoch
            train_loss: 학습 loss
            val_loss: 검증 loss
            learning_rate: 학습률
            patience_counter: Early stopping patience 카운터
            flush: 즉시 flush 여부
        """
        # TensorBoard 로깅
        if self.tensorboard_enabled and self.writer:
            if train_loss is not None:
                self.log_scalar('Loss/train', train_loss, epoch, flush=flush)
            if val_loss is not None:
                self.log_scalar('Loss/val', val_loss, epoch, flush=flush)
            if learning_rate is not None:
                self.log_scalar('Learning_Rate', learning_rate, epoch, flush=flush)
            if patience_counter is not None:
                self.log_scalar('Patience_Counter', patience_counter, epoch, flush=flush)
        
        # CSV 로깅
        if self.csv_enabled and self.csv_file:
            with open(self.csv_file, 'a') as f:
                train_loss_str = f"{train_loss:.6f}" if train_loss is not None else ""
                val_loss_str = f"{val_loss:.6f}" if val_loss is not None else ""
                lr_str = f"{learning_rate:.8f}" if learning_rate is not None else ""
                patience_str = str(patience_counter) if patience_counter is not None else ""
                f.write(f"{epoch},{train_loss_str},{val_loss_str},{lr_str},{patience_str}\n")
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "Metrics", flush: bool = False):
        """메트릭 딕셔너리 로깅"""
        if self.tensorboard_enabled and self.writer:
            for key, value in metrics.items():
                tag = f"{prefix}/{key}"
                self.log_scalar(tag, float(value), step, flush=flush)
    
    def log_text(self, tag: str, text: str, step: int = 0):
        """텍스트 로깅 (설정 정보 등)"""
        if self.tensorboard_enabled and self.writer:
            self.writer.add_text(tag, text, step)
    
    def flush(self):
        """즉시 디스크에 기록"""
        if self.tensorboard_enabled and self.writer:
            self.writer.flush()
    
    def close(self):
        """로거 종료"""
        if self.tensorboard_enabled and self.writer:
            self.writer.close()
            tb_dir = self.output_dir / "tensorboard"
            print(f"TensorBoard logs saved to: {tb_dir}")
            print(f"  View with: tensorboard --logdir {tb_dir}")
    
    @classmethod
    def from_yolo_csv(
        cls,
        csv_path: Union[str, Path],
        output_dir: Union[str, Path],
        config: Dict[str, Any]
    ):
        """
        YOLO results.csv에서 TensorBoard 로그 생성
        
        Args:
            csv_path: YOLO results.csv 경로
            output_dir: 출력 디렉토리
            config: 설정 딕셔너리
        """
        csv_path = Path(csv_path)
        output_dir = Path(output_dir)
        
        if not csv_path.exists():
            print(f"CSV file not found: {csv_path}")
            return
        
        logger = cls(output_dir, config)
        
        if not logger.tensorboard_enabled:
            return
        
        try:
            df = pd.read_csv(csv_path)
            
            metric_mapping = {
                'train/box_loss': 'Loss/train_box',
                'train/cls_loss': 'Loss/train_cls',
                'train/dfl_loss': 'Loss/train_dfl',
                'val/box_loss': 'Loss/val_box',
                'val/cls_loss': 'Loss/val_cls',
                'val/dfl_loss': 'Loss/val_dfl',
                'metrics/precision(B)': 'Metrics/precision',
                'metrics/recall(B)': 'Metrics/recall',
                'metrics/mAP50(B)': 'Metrics/mAP50',
                'metrics/mAP50-95(B)': 'Metrics/mAP50-95',
                'lr/pg0': 'Learning_Rate/pg0'
            }
            
            for idx, row in df.iterrows():
                epoch = int(row.get('epoch', idx))
                for csv_col, tb_tag in metric_mapping.items():
                    if csv_col in row and pd.notna(row[csv_col]):
                        try:
                            value = float(row[csv_col])
                            logger.log_scalar(tb_tag, value, epoch)
                        except (ValueError, TypeError):
                            continue
            
            logger.close()
            print(f"Converted YOLO CSV to TensorBoard logs")
            
        except Exception as e:
            print(f"Failed to convert CSV to TensorBoard logs: {e}")
            logger.close()


class CheckpointManager:
    """
    체크포인트 저장/로드 통합 관리
    - 모델 상태 저장
    - 최적화기 상태 저장
    - 메트릭 저장
    - Best/Last 체크포인트 관리
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        model_name: str = "model",
        save_best: bool = True,
        save_last: bool = True
    ):
        """
        Args:
            checkpoint_dir: 체크포인트 저장 디렉토리
            model_name: 모델 이름 (파일명에 사용)
            save_best: Best 체크포인트 저장 여부
            save_last: Last 체크포인트 저장 여부
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.save_best = save_best
        self.save_last = save_last
        
        self.best_metric = None
        self.best_epoch = None
    
    def save_checkpoint(
        self,
        epoch: int,
        model_state_dict: Dict,
        optimizer_state_dict: Optional[Dict] = None,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        prefix: str = ""
    ) -> Path:
        """
        체크포인트 저장
        
        Args:
            epoch: 현재 epoch
            model_state_dict: 모델 상태 딕셔너리
            optimizer_state_dict: 최적화기 상태 딕셔너리
            metrics: 메트릭 딕셔너리
            is_best: Best 체크포인트 여부
            prefix: 파일명 접두사
        
        Returns:
            저장된 체크포인트 경로
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
        }
        
        if optimizer_state_dict is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state_dict
        
        if metrics is not None:
            checkpoint.update(metrics)
        
        # Best 체크포인트 저장
        if is_best and self.save_best:
            checkpoint_path = self.checkpoint_dir / f"{prefix}{self.model_name}_best.pt"
            torch.save(checkpoint, checkpoint_path)
            self.best_metric = metrics.get('val_loss') if metrics else None
            self.best_epoch = epoch
            print(f"Best checkpoint saved: {checkpoint_path}")
        
        # Last 체크포인트 저장
        if self.save_last:
            checkpoint_path = self.checkpoint_dir / f"{prefix}{self.model_name}_last.pt"
            torch.save(checkpoint, checkpoint_path)
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        model,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cuda'
    ) -> Dict[str, Any]:
        """
        체크포인트 로드
        
        Args:
            checkpoint_path: 체크포인트 경로
            model: 모델 객체
            optimizer: 최적화기 객체 (선택)
            device: 디바이스
        
        Returns:
            체크포인트 딕셔너리
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 모델 상태 로드
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # 최적화기 상태 로드
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint
    
    def get_best_checkpoint_path(self) -> Optional[Path]:
        """Best 체크포인트 경로 반환"""
        best_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
        return best_path if best_path.exists() else None
    
    def get_last_checkpoint_path(self) -> Optional[Path]:
        """Last 체크포인트 경로 반환"""
        last_path = self.checkpoint_dir / f"{self.model_name}_last.pt"
        return last_path if last_path.exists() else None


class TrainingManager:
    """
    학습 관리 통합 클래스
    - 로깅 (TensorBoard, CSV, 로그 파일)
    - 체크포인트 저장/로드
    
    사용 예시:
        manager = TrainingManager(output_dir, config, model_name="florence2")
        manager.log_losses(epoch=1, train_loss=0.5, val_loss=0.6)
        manager.save_checkpoint(epoch=1, model_state_dict=..., is_best=True)
        manager.close()
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        config: Dict[str, Any],
        model_name: str = "model",
        log_file_name: Optional[str] = None,
        save_best: bool = True,
        save_last: bool = True
    ):
        """
        Args:
            output_dir: 출력 디렉토리
            config: 설정 딕셔너리
            model_name: 모델 이름 (체크포인트 파일명에 사용)
            log_file_name: 로그 파일명 (None이면 자동 생성)
            save_best: Best 체크포인트 저장 여부
            save_last: Last 체크포인트 저장 여부
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 로거 초기화
        self.logger = TrainingLogger(output_dir, config, log_file_name)
        
        # 체크포인트 관리자 초기화
        checkpoint_dir = output_dir / "checkpoints"
        self.checkpoint = CheckpointManager(
            checkpoint_dir,
            model_name=model_name,
            save_best=save_best,
            save_last=save_last
        )
        
        self.output_dir = output_dir
        self.config = config
        self.model_name = model_name
    
    # 로깅 메서드 위임
    def log_scalar(self, tag: str, value: float, step: int, flush: bool = False):
        """TensorBoard에 스칼라 로깅"""
        return self.logger.log_scalar(tag, value, step, flush)
    
    def log_losses(
        self,
        epoch: int,
        train_loss: float = None,
        val_loss: float = None,
        learning_rate: float = None,
        patience_counter: int = None,
        flush: bool = False
    ):
        """Loss 및 관련 메트릭 로깅 (통합)"""
        return self.logger.log_losses(
            epoch, train_loss, val_loss, learning_rate, patience_counter, flush
        )
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "Metrics", flush: bool = False):
        """메트릭 딕셔너리 로깅"""
        return self.logger.log_metrics(metrics, step, prefix, flush)
    
    def log_text(self, tag: str, text: str, step: int = 0):
        """텍스트 로깅 (설정 정보 등)"""
        return self.logger.log_text(tag, text, step)
    
    def flush(self):
        """즉시 디스크에 기록"""
        return self.logger.flush()
    
    # 체크포인트 메서드 위임
    def save_checkpoint(
        self,
        epoch: int,
        model_state_dict: Dict,
        optimizer_state_dict: Optional[Dict] = None,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        prefix: str = ""
    ) -> Path:
        """체크포인트 저장"""
        return self.checkpoint.save_checkpoint(
            epoch, model_state_dict, optimizer_state_dict, metrics, is_best, prefix
        )
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        model,
        optimizer: Optional[Any] = None,
        device: str = 'cuda'
    ) -> Dict[str, Any]:
        """체크포인트 로드"""
        return self.checkpoint.load_checkpoint(checkpoint_path, model, optimizer, device)
    
    def get_best_checkpoint_path(self) -> Optional[Path]:
        """Best 체크포인트 경로 반환"""
        return self.checkpoint.get_best_checkpoint_path()
    
    def get_last_checkpoint_path(self) -> Optional[Path]:
        """Last 체크포인트 경로 반환"""
        return self.checkpoint.get_last_checkpoint_path()
    
    def close(self):
        """로거 종료"""
        self.logger.close()
    
    # 편의 속성
    @property
    def tensorboard_enabled(self) -> bool:
        """TensorBoard 활성화 여부"""
        return self.logger.tensorboard_enabled
    
    @property
    def csv_enabled(self) -> bool:
        """CSV 활성화 여부"""
        return self.logger.csv_enabled
    
    @property
    def csv_file(self) -> Optional[Path]:
        """CSV 파일 경로"""
        return self.logger.csv_file
    
    @property
    def log_file(self) -> Optional[Path]:
        """로그 파일 경로"""
        return self.logger.log_file

