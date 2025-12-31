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
        
        # Experiment ID 생성 (타임스탬프)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = self.timestamp
        
        # 실험 디렉토리 구조: exp/{model}/{dataset}_{timestamp}
        model_name = config['model']['arch_name']
        dataset_name = config['data']['dataset_name']
        exp_name = f"{dataset_name}_{self.timestamp}"
        
        self.exp_dir = Path(config['experiment']['save_dir']) / model_name / exp_name
        
        # 하위 디렉토리
        self.config_dir = self.exp_dir / "config"
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.tensorboard_dir = self.exp_dir / "tensorboard"
        self.results_dir = self.exp_dir / "results"
        
        # 디렉토리 생성
        self._create_directories()
        
        # Config와 metadata 저장
        self._save_experiment_info()
    
    def _create_directories(self):
        """실험 폴더 생성"""
        for directory in [self.config_dir, self.checkpoint_dir, 
                         self.tensorboard_dir, self.results_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _save_experiment_info(self):
        """Config 및 메타데이터 저장"""
        # Config 저장
        config_path = self.exp_dir / "config.yaml"
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
        print(f"{'='*70}\n")