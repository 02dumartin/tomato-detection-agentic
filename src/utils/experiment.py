"""실험 관리 및 자동 로그 저장"""
import os
import json
import yaml
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


def save_config(config, filepath):
    """Config를 YAML 파일로 저장"""
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


class ExperimentManager:    
    def __init__(self, config):
        self.config = config
        
        # Experiment ID 생성 (날짜 + 번호)
        date_str = datetime.now().strftime("%y%m%d")
        model_name = config['model']['arch_name']
        dataset_name = config['data']['dataset_name']
        
        # 실험 디렉토리 구조: exp/{model}/{dataset}_{model}_{date}_{number}
        # 같은 날짜에 같은 데이터+모델 조합이 있으면 번호 증가
        exp_base_name = f"{dataset_name}_{model_name}_{date_str}"
        exp_name = self._get_experiment_name(Path(config['experiment']['save_dir']) / model_name, exp_base_name)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = exp_name
        
        self.exp_dir = Path(config['experiment']['save_dir']) / model_name / exp_name
        
        # 하위 디렉토리
        self.config_dir = self.exp_dir / "config"
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.tensorboard_dir = self.exp_dir / "tensorboard"
        self.results_dir = self.exp_dir / "results"
        self.log_dir = self.exp_dir / "log"
        
        # 디렉토리 생성
        self._create_directories()
        
        # Config와 metadata 저장
        self._save_experiment_info()
    
    def _get_experiment_name(self, base_dir: Path, exp_base_name: str) -> str:
        """실험 이름 생성: 같은 날짜에 같은 조합이 있으면 번호 증가"""
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # 기존 실험 디렉토리 확인
        existing_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith(exp_base_name)]
        
        if not existing_dirs:
            # 첫 번째 실험
            return f"{exp_base_name}_1"
        else:
            # 번호 추출 및 최대값 찾기
            numbers = []
            for dir_name in existing_dirs:
                parts = dir_name.name.split('_')
                if len(parts) > 0:
                    try:
                        num = int(parts[-1])
                        numbers.append(num)
                    except ValueError:
                        pass
            
            if numbers:
                next_number = max(numbers) + 1
            else:
                next_number = 1
            
            return f"{exp_base_name}_{next_number}"
    
    def _create_directories(self):
        """실험 폴더 생성"""
        # 기본 디렉토리들
        directories = [self.config_dir, self.tensorboard_dir, self.results_dir, self.log_dir]
        
        # YOLO 모델이 아닌 경우에만 checkpoint 디렉토리 생성
        model_name = self.config['model']['arch_name'].lower()
        if model_name not in ["yolov11", "yolov12", "yolo"]:
            directories.append(self.checkpoint_dir)
        else:
            print("🔧 YOLO model detected: Skipping checkpoint directory creation")
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _save_experiment_info(self):
        """Config 및 메타데이터 저장"""
        config_path = self.config_dir / "config.yaml"
        save_config(self.config, config_path)
        
        # Git 정보 저장
        git_info_path = self.exp_dir / "git_info.txt"
        try:
            git_info = []
            git_info.append(f"Commit: {self._get_git_commit()}")
            git_info.append(f"Branch: {self._get_git_branch()}")
            git_info.append(f"Status:\n{self._get_git_status()}")
            
            with open(git_info_path, 'w') as f:
                f.write('\n'.join(git_info))
        except:
            pass  # Git 정보 없어도 계속 진행
        
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
    
    def _get_git_commit(self):
        """현재 Git commit hash"""
        try:
            return subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], 
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
        except:
            return 'N/A'
    
    def _get_git_branch(self):
        """현재 Git branch"""
        try:
            return subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
        except:
            return 'N/A'
    
    def _get_git_status(self):
        """Git status"""
        try:
            return subprocess.check_output(
                ['git', 'status', '--short'],
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
        except:
            return 'N/A'
    
    def _save_summary(self, summary):
        """Summary JSON 저장"""
        summary_path = self.exp_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def save_final_results(self, results: Dict[str, Any]):
        """실험 최종 결과 저장"""
        # Summary JSON 업데이트
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
        print(f"     Log:          {self.log_dir}")
        print(f"{'='*70}\n")


def extract_exp_info_from_checkpoint(checkpoint_path: str, config: Dict) -> Dict[str, str]:
    """
    Checkpoint 경로에서 실험 정보 추출
    
    Args:
        checkpoint_path: 체크포인트 파일 경로
        config: 설정 딕셔너리
    
    Returns:
        {'model_name': str, 'exp_name': str}
    """
    checkpoint_path = Path(checkpoint_path)
    parts = checkpoint_path.parts
    
    # exp 디렉토리 찾기
    try:
        exp_idx = parts.index('exp') if 'exp' in parts else -1
        if exp_idx >= 0 and len(parts) > exp_idx + 2:
            model_name = parts[exp_idx + 1]  # yolov11
            exp_name = parts[exp_idx + 2]    # TomatOD_YOLO_3_20260106_105855 또는 TomatOD_YOLO_3_yolov11_20260106_1
            
            # exp_name이 새 형식인지 확인
            # 새 형식: dataset_model_date_number (예: TomatOD_YOLO_3_yolov11_20260106_1)
            exp_parts = exp_name.split('_')
            
            # 새 형식 체크: 마지막에서 3번째가 6자리 숫자(날짜)이고, 마지막이 숫자(번호)인 경우
            is_new_format = (len(exp_parts) >= 5 and 
                           exp_parts[-3].isdigit() and len(exp_parts[-3]) == 6 and
                           exp_parts[-1].isdigit() and
                           exp_parts[-2] == model_name.lower())
            
            if not is_new_format:
                # 기존 형식이면 그대로 사용 (하위 호환성)
                pass
            
            return {
                'model_name': model_name,
                'exp_name': exp_name
            }
    except:
        pass
    
    # 추출 실패 시 config에서 정보 가져오기
    model_name = config['model']['arch_name']
    dataset_name = config['data']['dataset_name']
    date_str = datetime.now().strftime("%y%m%d")
    exp_base_name = f"{dataset_name}_{model_name}_{date_str}"
    exp_dir = Path("exp") / model_name
    exp_name = get_experiment_name(exp_dir, exp_base_name)
    
    return {
        'model_name': model_name,
        'exp_name': exp_name
    }


def get_experiment_name(base_dir: Path, exp_base_name: str) -> str:
    """
    실험 이름 생성: 같은 날짜에 같은 조합이 있으면 번호 증가
    
    Args:
        base_dir: 실험 디렉토리 기본 경로
        exp_base_name: 실험 기본 이름
    
    Returns:
        실험 이름 (예: "TomatOD_YOLO_3_yolov11_260106_1")
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # 기존 실험 디렉토리 확인
    existing_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith(exp_base_name)]
    
    if not existing_dirs:
        # 첫 번째 실험
        return f"{exp_base_name}_1"
    else:
        # 번호 추출 및 최대값 찾기
        numbers = []
        for dir_name in existing_dirs:
            parts = dir_name.name.split('_')
            if len(parts) > 0:
                try:
                    num = int(parts[-1])
                    numbers.append(num)
                except ValueError:
                    pass
        
        if numbers:
            next_number = max(numbers) + 1
        else:
            next_number = 1
        
        return f"{exp_base_name}_{next_number}"