"""실험 관리 및 자동 로그 저장"""
import os
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from .config import save_config


# Git 관련 함수들
def get_git_commit() -> str:
    """현재 Git commit hash"""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], 
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
    except:
        return 'N/A'


def get_git_branch() -> str:
    """현재 Git branch"""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
    except:
        return 'N/A'


def get_git_status() -> str:
    """Git status"""
    try:
        return subprocess.check_output(
            ['git', 'status', '--short'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
    except:
        return 'N/A'


def save_git_info(output_path: Path):
    """Git 정보를 파일로 저장"""
    try:
        git_info = []
        git_info.append(f"Commit: {get_git_commit()}")
        git_info.append(f"Branch: {get_git_branch()}")
        git_info.append(f"Status:\n{get_git_status()}")
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(git_info))
    except:
        pass


# 실험 이름 생성 함수들
def get_model_directory_florence2(base_save_dir: Path, config: Dict[str, Any]) -> Path:
    """
    모델별 디렉토리 경로 생성
    
    Args:
        base_save_dir: 기본 저장 디렉토리
        config: 설정 딕셔너리
    
    Returns:
        모델 디렉토리 경로
    """
    model_name = config['model']['arch_name']
    
    # Florence-2 모델의 경우 mode에 따라 하위 폴더 생성
    if model_name.lower() in ['florence2', 'florence-2']:
        florence2_mode = config.get('florence2', {}).get('mode', 'zeroshot')
        mode_folder = 'zeroshot' if florence2_mode == 'zeroshot' else 'finetuned'
        return base_save_dir / model_name / mode_folder
    else:
        return base_save_dir / model_name


def get_experiment_name(base_dir: Path, config: Dict[str, Any], exp_base_name: Optional[str] = None) -> str:
    """
    실험 이름 생성: config의 experiment.name을 우선 사용, 없으면 자동 생성
    
    Args:
        base_dir: 실험 디렉토리 기본 경로
        config: 설정 딕셔너리 (exp_base_name이 None일 때 사용)
        exp_base_name: 실험 기본 이름 (선택적, None이면 config에서 생성)
    
    Returns:
        실험 이름 (예: "TomatOD_YOLO_3_yolov11_260106_1" 또는 config의 name)
    """
    # config에 experiment.name이 있으면 그대로 사용 (고정된 이름)
    experiment_config = config.get('experiment', {})
    if 'name' in experiment_config and experiment_config['name']:
        fixed_name = experiment_config['name']
        # 항상 같은 이름 사용 (기존 디렉토리 있으면 덮어쓰기)
        return fixed_name
    
    # 기본 이름 생성 (제공되지 않은 경우)
    if exp_base_name is None:
        date_str = datetime.now().strftime("%y%m%d")
        model_name = config['model']['arch_name']
        # data_root에서 실제 데이터셋 폴더 이름 추출
        data_config = config.get('data', {})
        if 'data_root' in data_config:
            data_root = Path(data_config['data_root'])
            # data/TomatOD_3_YOLO -> TomatOD_3_YOLO
            dataset_name = data_root.name
        else:
            # fallback: dataset_name 사용
            dataset_name = data_config.get('dataset_name', 'unknown')
        exp_base_name = f"{dataset_name}_{model_name}_{date_str}"
    
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # 같은 날짜/조합의 실험이 있는지 확인
    existing_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith(exp_base_name)]
    
    # 번호 찾기: 기존 실험들의 번호를 확인
    numbers = []
    for d in existing_dirs:
        name = d.name
        if name == exp_base_name:
            # 번호 없는 것도 1로 간주 (첫 번째 실험)
            numbers.append(1)
        elif name.startswith(exp_base_name + "_"):
            suffix = name[len(exp_base_name) + 1:]
            try:
                num = int(suffix)
                numbers.append(num)
            except ValueError:
                pass
    
    # 다음 번호 결정 (첫 번째 실험부터 1부터 시작)
    if numbers:
        next_num = max(numbers) + 1
    else:
        next_num = 1  # 첫 번째 실험도 1부터 시작
    
    return f"{exp_base_name}_{next_num}"


# 모델별 디렉토리 생성 전략
def get_required_directories(config: Dict[str, Any], exp_dir: Path) -> List[Path]:
    """
    모델별로 필요한 디렉토리 목록 반환
    
    Args:
        config: 설정 딕셔너리
        exp_dir: 실험 디렉토리
    
    Returns:
        생성할 디렉토리 목록
    """
    model_name = config['model']['arch_name'].lower()
    directories = []
    
    # Florence-2 모델
    if model_name in ['florence2', 'florence-2']:
        florence2_mode = config.get('florence2', {}).get('mode', 'zeroshot')
        if florence2_mode == 'zeroshot':
            # Zero-shot은 학습이 없으므로 checkpoint와 tensorboard 불필요
            pass
        else:
            # Fine-tuning 모드는 checkpoint와 tensorboard 필요
            directories.extend([
                exp_dir / "checkpoints",
                exp_dir / "tensorboard"
            ])
    
    # YOLO 모델
    elif model_name in ["yolov11", "yolov12", "yolo"]:
        # YOLO는 자체 checkpoint 시스템 사용, tensorboard만 필요
        directories.append(exp_dir / "tensorboard")
    
    # DETR 등 다른 모델
    else:
        # DETR 등 다른 모델은 checkpoint와 tensorboard 필요
        directories.extend([
            exp_dir / "checkpoints",
            exp_dir / "tensorboard"
        ])
    
    return directories


class ExperimentManager:    
    """
    실험 관리 클래스
    - 실험 디렉토리 생성 및 관리
    - 메타데이터 저장 (config, git info, summary)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        
        # 모델 디렉토리 생성
        base_save_dir = Path(config['experiment']['save_dir'])
        model_dir = get_model_directory_florence2(base_save_dir, config)
        
        # 실험 이름 생성 (내부에서 기본 이름 생성)
        exp_name = get_experiment_name(model_dir, config)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = exp_name
        self.exp_dir = model_dir / exp_name
        
        # 하위 디렉토리 경로 설정
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.tensorboard_dir = self.exp_dir / "tensorboard"
        
        # 디렉토리 생성
        self._create_directories()
        
        # Config와 metadata 저장
        self._save_experiment_info()
    
    def _create_directories(self):
        """실험 폴더 생성"""
        directories = get_required_directories(self.config, self.exp_dir)
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        # 정보 출력
        model_name = self.config['model']['arch_name'].lower()
        if model_name in ['florence2', 'florence-2']:
            florence2_mode = self.config.get('florence2', {}).get('mode', 'zeroshot')
            if florence2_mode == 'zeroshot':
                print("Florence-2 Zero-shot mode: Skipping checkpoint and tensorboard directory creation")
        elif model_name in ["yolov11", "yolov12", "yolo"]:
            print("YOLO model detected: Skipping checkpoint directory creation")
    
    def _save_experiment_info(self):
        """Config 및 메타데이터 저장"""
        # Config 저장
        config_path = self.exp_dir / "config.yaml"
        save_config(self.config, config_path)
        
        # Git 정보 저장
        git_info_path = self.exp_dir / "git_info.txt"
        save_git_info(git_info_path)
        
        # Summary JSON 초기화
        summary = {
            'experiment_id': self.experiment_id,
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model': self.config['model']['arch_name'],
            'dataset': self.config['data']['dataset_name'],
            'num_classes': self.config['data']['num_classes'],
            'status': 'running'
        }
        self._save_summary(summary)
    
    def _save_summary(self, summary: Dict[str, Any]):
        """Summary JSON 저장"""
        summary_path = self.exp_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def save_final_results(self, results: Dict[str, Any]):
        """
        실험 최종 결과 저장
        
        Args:
            results: 실험 결과 딕셔너리
        """
        summary_path = self.exp_dir / "summary.json"
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        summary.update({
            'status': 'completed',
            'completed_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'results': results
        })
        
        self._save_summary(summary)
    
    def print_info(self):
        """실험 정보 출력"""
        print(f"\nEXPERIMENT SETUP")
        print(f"{'='*70}")
        print(f"  Experiment ID:   {self.experiment_id}")
        print(f"  Name:            {self.config['experiment']['name']}")
        print(f"  Model:           {self.config['model']['arch_name']}")
        print(f"  Dataset:         {self.config['data']['dataset_name']} "
              f"({self.config['data']['num_classes']} classes)")
        print(f"  Classes:         {', '.join(self.config['data']['class_names'])}")
        print(f"")
        print(f"  Directories:")
        print(f"     Root:         {self.exp_dir}")
        print(f"     Checkpoints:  {self.checkpoint_dir}")
        print(f"     TensorBoard:  {self.tensorboard_dir}")
        print(f"{'='*70}\n")


def extract_exp_info_from_checkpoint(checkpoint_path: str, config: Dict[str, Any]) -> Dict[str, str]:
    """
    Checkpoint 경로에서 실험 정보 추출 (실험 디렉토리 구조 파싱)
    
    Checkpoint 경로 형식: exp/{model_name}/{exp_name}/... 또는
    exp/{model_name}/{mode}/{exp_name}/... (Florence-2의 경우)
    
    Args:
        checkpoint_path: 체크포인트 파일 경로
        config: 설정 딕셔너리 (경로 파싱 실패 시 fallback으로 사용)
    
    Returns:
        {'model_name': str, 'exp_name': str}
    """
    checkpoint_path = Path(checkpoint_path)
    parts = checkpoint_path.parts
    
    # exp 디렉토리 찾기
    try:
        exp_idx = parts.index('exp') if 'exp' in parts else -1
        if exp_idx >= 0 and len(parts) > exp_idx + 2:
            model_name = parts[exp_idx + 1]
            
            # Florence-2의 경우 mode 폴더 처리
            if model_name.lower() in ['florence2', 'florence-2'] and len(parts) > exp_idx + 3:
                if parts[exp_idx + 2] in ['zeroshot', 'finetuned']:
                    exp_name = parts[exp_idx + 3]
                else:
                    exp_name = parts[exp_idx + 2]
            else:
                exp_name = parts[exp_idx + 2]
            
            return {
                'model_name': model_name,
                'exp_name': exp_name
            }
    except:
        pass
    
    # 추출 실패 시 config에서 정보 가져오기
    model_name = config['model']['arch_name']
    base_save_dir = Path(config.get('experiment', {}).get('save_dir', 'exp'))
    model_dir = get_model_directory_florence2(base_save_dir, config)
    exp_name = get_experiment_name(model_dir, config)
    
    return {
        'model_name': model_name,
        'exp_name': exp_name
    }
