"""Config 관리 및 CLI merge"""
import yaml
from pathlib import Path
from typing import Dict, Any

class ConfigDict(dict):
    """점(.) 표기법으로 접근 가능한 dict"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = ConfigDict(value)
    
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"No attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value


def load_config(config_path: str) -> Dict:
    """YAML config 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def merge_config_with_args(config: Dict, args) -> Dict:
    """CLI 인자를 config에 merge"""
    
    # Model 파라미터
    if args.model:
        config['model']['arch_name'] = args.model
    if args.lr:
        config['model']['learning_rate'] = args.lr
    
    # Data 파라미터
    if args.data:
        config['data']['dataset_name'] = args.data
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    
    # Trainer 파라미터
    if args.epochs:
        config['trainer']['max_epochs'] = args.epochs
    if args.gpu is not None:
        config['trainer']['devices'] = [args.gpu]
    
    # Experiment 파라미터
    if args.tag:
        config.setdefault('experiment', {})
        config['experiment']['name'] = f"{config['experiment'].get('name', 'exp')}_{args.tag}"
    
    if args.debug:
        config['debug'] = True
    
    if args.resume:
        config['resume'] = args.resume
    
    return config


def get_default_config_path(model_name: str, dataset_name: str) -> Path:
    """기본 config 경로 찾기"""
    # configs/{MODEL}/{model}_{dataset}.yaml
    config_path = Path(f"configs/{model_name}/{model_name.lower()}_{dataset_name.lower()}.yaml")
    
    if config_path.exists():
        return config_path
    
    # configs/{MODEL}/{dataset}.yaml
    config_path = Path(f"configs/{model_name}/{dataset_name.lower()}.yaml")
    
    return config_path if config_path.exists() else None