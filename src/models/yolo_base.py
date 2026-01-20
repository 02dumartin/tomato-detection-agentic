"""YOLO 모델 공통 베이스 클래스"""
from abc import ABC
from pathlib import Path
from typing import Dict, Any
import re
import sys
import os
import yaml
from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO

from ..utils.training import TrainingLogger


def create_data_yaml(
    data_config: Dict[str, Any],
    output_path: Path = None,
    train_subdir: str = None,
    val_subdir: str = None,
    test_subdir: str = None
) -> Path:
    """
    YOLO data.yaml 파일 생성
    
    Args:
        data_config: 데이터 설정 딕셔너리
        output_path: 출력 경로 (None이면 data_root/data.yaml)
        train_subdir: 학습 이미지 서브디렉토리 (None이면 config에서 추출)
        val_subdir: 검증 이미지 서브디렉토리 (None이면 config에서 추출)
        test_subdir: 테스트 이미지 서브디렉토리 (None이면 config에서 추출)
    
    Returns:
        생성된 data.yaml 파일 경로
    """
    if output_path is None:
        output_path = Path(data_config['data_root']) / 'data.yaml'
    
    # config에서 경로 추출 (train_dir, val_dir, test_dir 사용)
    data_root = Path(data_config['data_root']).absolute()
    
    # train_subdir 추출
    if train_subdir is None:
        if 'train_dir' in data_config:
            train_dir = Path(data_config['train_dir']).absolute()
            # data_root 기준 상대 경로 계산
            try:
                train_subdir = str(train_dir.relative_to(data_root))
            except ValueError:
                # 절대 경로인 경우
                train_subdir = "train/images"  # 기본값
        else:
            train_subdir = "train/images"  # 기본값
    
    # val_subdir 추출
    if val_subdir is None:
        if 'val_dir' in data_config:
            val_dir = Path(data_config['val_dir']).absolute()
            try:
                val_subdir = str(val_dir.relative_to(data_root))
            except ValueError:
                val_subdir = "val/images"  # 기본값
        else:
            val_subdir = "val/images"  # 기본값
    
    # test_subdir 추출
    if test_subdir is None:
        if 'test_dir' in data_config:
            test_dir = Path(data_config['test_dir']).absolute()
            try:
                test_subdir = str(test_dir.relative_to(data_root))
            except ValueError:
                test_subdir = "test/images"  # 기본값
        else:
            test_subdir = "test/images"  # 기본값
    
    yolo_data = {
        'path': str(data_root),
        'train': train_subdir,
        'val': val_subdir,
        'test': test_subdir,
        'nc': data_config['num_classes'],
        'names': data_config['class_names']
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(yolo_data, f, default_flow_style=False)
    
    return output_path


def build_yolo_train_args(
    config: Dict[str, Any],
    data_yaml_path: Path
) -> Dict[str, Any]:
    """
    YOLO 학습 인자 구성
    
    Args:
        config: 전체 설정 딕셔너리
        data_yaml_path: data.yaml 파일 경로
    
    Returns:
        YOLO model.train()에 전달할 인자 딕셔너리
    """
    model_config = config['model']
    data_config = config['data']
    training_config = config.get('training', {})
    
    train_args = {
        'data': str(data_yaml_path),
        'epochs': training_config.get('epochs', 100),
        'batch': data_config.get('batch_size', 16),
        'imgsz': data_config.get('imgsz', 640),
        'device': training_config.get('device', 0),
        'lr0': model_config.get('lr0', 0.001),
        'lrf': model_config.get('lrf', 0.01),
        'momentum': model_config.get('momentum', 0.937),
        'weight_decay': model_config.get('weight_decay', 0.0005),
        'box': model_config.get('box', 7.5),
        'cls': model_config.get('cls', 0.5),
        'dfl': model_config.get('dfl', 1.5),
        'patience': training_config.get('patience', 20),
        'project': training_config.get('project', 'exp/yolo'),
        'name': training_config.get('name', 'train'),
        'exist_ok': training_config.get('exist_ok', False),
        'plots': training_config.get('plots', True),
        'amp': training_config.get('amp', True),
        'seed': config.get('experiment', {}).get('seed', 42),
        'save_period': training_config.get('save_period', 10),
    }
    
    # 로깅 설정은 환경 변수로 처리 (YOLO는 train_args에 직접 전달 불가)
    # 환경 변수는 train() 메서드에서 설정됨
    
    # 데이터 증강 설정
    if 'augmentation' in data_config:
        aug_config = data_config['augmentation']
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
    
    return train_args


class BaseYOLOWrapper(ABC):
    """YOLO 모델 공통 기능을 제공하는 베이스 클래스"""
    
    def __init__(self, config: Dict[str, Any], model_version: str = "yolo12"):
        """
        Args:
            config: 전체 설정 딕셔너리
            model_version: "yolo11" 또는 "yolo12"
        """
        self.config = config
        self.model_config = config['model']
        self.data_config = config['data']
        self.training_config = config.get('training', {})
        self.model_version = model_version
        
        # YOLO 모델 초기화 - config에 적힌 경로를 그대로 사용
        pretrained_path = self.model_config['pretrained_path']
        
        # 파일이 존재하는지 확인 (상대 경로인 경우 프로젝트 루트 기준)
        pretrained_path_obj = Path(pretrained_path)
        if not pretrained_path_obj.is_absolute():
            # 프로젝트 루트를 찾기 위해 현재 파일 위치 기준으로 계산
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent  # src/models/yolo_base.py -> project root
            full_path = (project_root / pretrained_path).resolve()
            
            # 파일이 존재하는지 확인
            if not full_path.exists():
                raise FileNotFoundError(
                    f"Pretrained model not found: {full_path}\n"
                    f"Config path: {pretrained_path}\n"
                    f"Please check the path in config file."
                )
        else:
            # 절대 경로인 경우
            if not pretrained_path_obj.exists():
                raise FileNotFoundError(
                    f"Pretrained model not found: {pretrained_path_obj}\n"
                    f"Please check the path in config file."
                )
        
        # YOLO 자동 다운로드 방지 (AMP 체크용 다운로드 막기)
        import os
        original_download = os.environ.get('YOLO_WEIGHTS_AUTO_DOWNLOAD', None)
        os.environ['YOLO_WEIGHTS_AUTO_DOWNLOAD'] = 'False'
        
        try:
            # config에 적힌 경로를 그대로 사용 (YOLO가 상대 경로를 처리할 수 있음)
            self.model = YOLO(pretrained_path)
        finally:
            # 환경변수 복원
            if original_download is None:
                os.environ.pop('YOLO_WEIGHTS_AUTO_DOWNLOAD', None)
            else:
                os.environ['YOLO_WEIGHTS_AUTO_DOWNLOAD'] = original_download
        
        # data.yaml 생성
        self.data_yaml_path = create_data_yaml(self.data_config)
    
    def train(self, output_dir: Path):
        """YOLO 학습 실행"""
        # 학습 인자 구성
        train_args = build_yolo_train_args(self.config, self.data_yaml_path)
        
        # 출력 디렉토리 설정 - YOLO가 정확히 이 디렉토리를 사용하도록 설정
        train_args['project'] = str(output_dir.parent)
        train_args['name'] = output_dir.name
        train_args['exist_ok'] = True  # 기존 디렉토리 덮어쓰기 허용 (중복 폴더 생성 방지)
        
        # 실시간 로그 파일 생성
        from datetime import datetime
        log_file = output_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        print("=" * 60)
        print("YOLO TRAINING STARTED")
        print("=" * 60)
        print(f"Model: {self.model_version.upper()}")
        print(f"Dataset: {self.data_config.get('dataset_name', 'Unknown')}")
        print(f"Epochs: {train_args['epochs']}")
        print(f"Batch size: {train_args['batch']}")
        print(f"Learning rate: {train_args['lr0']}")
        print(f"Log file: {log_file}")
        print("=" * 60)
        
        # 로깅 설정을 환경 변수로 처리 (YOLO는 환경 변수로 로깅 제어)
        logging_config = self.config.get('logging', {})
        
        # 환경 변수 백업
        original_env = {}
        original_cwd = os.getcwd()
        
        # 프로젝트 루트로 작업 디렉토리 변경 (YOLO가 config에 적힌 가중치 파일을 찾을 수 있도록)
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        os.chdir(str(project_root))
        
        env_vars = {
            'TENSORBOARD': '1' if logging_config.get('tensorboard', True) else '0',
            'WANDB': '1' if logging_config.get('wandb', False) else '0',
            'CLEARML': '1' if logging_config.get('clearml', False) else '0',
            'COMET': '1' if logging_config.get('comet', False) else '0',
            'YOLO_WEIGHTS_AUTO_DOWNLOAD': 'False',  # 자동 다운로드 비활성화 (AMP 체크용 다운로드도 방지)
        }
        
        for key, value in env_vars.items():
            original_env[key] = os.environ.get(key, None)
            os.environ[key] = value
        
        # wandb 추가 설정
        if logging_config.get('wandb', False):
            if 'wandb_project' in logging_config:
                original_env['WANDB_PROJECT'] = os.environ.get('WANDB_PROJECT', None)
                os.environ['WANDB_PROJECT'] = str(logging_config['wandb_project'])
            if 'wandb_entity' in logging_config and logging_config['wandb_entity']:
                original_env['WANDB_ENTITY'] = os.environ.get('WANDB_ENTITY', None)
                os.environ['WANDB_ENTITY'] = str(logging_config['wandb_entity'])
        
        ansi_escape = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

        class Tee:
            def __init__(self, console_stream, file_stream):
                self.console_stream = console_stream
                self.file_stream = file_stream

            def _clean(self, data: str) -> str:
                cleaned = ansi_escape.sub("", data)
                return cleaned.replace("\r", "\n")

            def write(self, data):
                self.console_stream.write(data)
                self.console_stream.flush()
                if self.file_stream:
                    self.file_stream.write(self._clean(data))
                    self.file_stream.flush()

            def flush(self):
                self.console_stream.flush()
                if self.file_stream:
                    self.file_stream.flush()

        log_fh = open(log_file, 'a', buffering=1)
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = Tee(original_stdout, log_fh)
        sys.stderr = Tee(original_stderr, log_fh)

        import threading
        import time
        import shutil
        from pathlib import Path as PathLib

        monitoring = {'active': True}
        yolo_results_dir = PathLib(train_args['project']) / train_args['name']
        tb_target_dir = output_dir / "tensorboard"
        tb_target_dir.mkdir(parents=True, exist_ok=True)

        tb_writer = None
        if logging_config.get('tensorboard', True):
            tb_writer = SummaryWriter(log_dir=str(tb_target_dir), flush_secs=10)

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

        def monitor_tensorboard_events():
            """학습 중 생성되는 TensorBoard 이벤트 파일을 실험 폴더로 복사"""
            copied = set()
            while monitoring['active']:
                try:
                    tb_files = yolo_results_dir.glob("**/events.out.tfevents.*")
                    for tb_file in tb_files:
                        if tb_file in copied:
                            continue
                        target = tb_target_dir / tb_file.name
                        if not target.exists():
                            shutil.copy2(str(tb_file), str(target))
                        copied.add(tb_file)
                except Exception:
                    pass
                time.sleep(2)

        def monitor_csv_to_tensorboard():
            """YOLO results.csv를 실시간으로 읽어 TensorBoard에 기록"""
            if tb_writer is None:
                return
            last_line_count = 0
            while monitoring['active']:
                csv_path = yolo_results_dir / "results.csv"
                if not csv_path.exists():
                    time.sleep(2)
                    continue
                try:
                    with open(csv_path, 'r', newline='') as f:
                        lines = f.readlines()
                    if len(lines) <= 1:
                        time.sleep(2)
                        continue
                    header = [h.strip() for h in lines[0].strip().split(',')]
                    rows = lines[1:]
                    if len(rows) <= last_line_count:
                        time.sleep(2)
                        continue
                    new_rows = rows[last_line_count:]
                    for row in new_rows:
                        values = [v.strip() for v in row.strip().split(',')]
                        if len(values) != len(header):
                            continue
                        row_dict = dict(zip(header, values))
                        epoch_str = row_dict.get('epoch')
                        try:
                            epoch = int(float(epoch_str)) if epoch_str is not None else last_line_count
                        except ValueError:
                            epoch = last_line_count
                        for csv_col, tb_tag in metric_mapping.items():
                            if csv_col in row_dict and row_dict[csv_col] not in ("", "nan", "NaN"):
                                try:
                                    value = float(row_dict[csv_col])
                                    tb_writer.add_scalar(tb_tag, value, epoch)
                                except ValueError:
                                    continue
                    tb_writer.flush()
                    last_line_count = len(rows)
                except Exception:
                    pass
                time.sleep(2)

        monitor_thread = threading.Thread(target=monitor_tensorboard_events, daemon=True)
        monitor_thread.start()
        csv_monitor_thread = threading.Thread(target=monitor_csv_to_tensorboard, daemon=True)
        csv_monitor_thread.start()

        try:
            results = self.model.train(**train_args)
        finally:
            monitoring['active'] = False
            monitor_thread.join(timeout=5)
            csv_monitor_thread.join(timeout=5)
            if tb_writer:
                tb_writer.close()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_fh.close()
            # 환경 변수 복원
            for key, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value
            
            # 작업 디렉토리 복원
            os.chdir(original_cwd)
        
        print("\nYOLO Training completed!")
        print(f"Results saved to: {results.save_dir}")
        return results
    
    def evaluate(self, **kwargs):
        """평가 실행"""
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
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, **kwargs):
        """체크포인트에서 모델 로드"""
        model = YOLO(str(checkpoint_path))
        temp_config = {
            'model': {'pretrained_path': str(checkpoint_path)},
            'data': {
                'data_root': '',
                'num_classes': kwargs.get('num_classes', 1),
                'class_names': kwargs.get('class_names', ['class_0']),
            },
            'training': {}
        }
        wrapper = cls(temp_config, **kwargs)
        wrapper.model = model
        return wrapper
