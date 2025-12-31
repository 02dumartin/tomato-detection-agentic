"""Dataset Registry"""

class DatasetRegistry:
    """데이터셋 메타정보 레지스트리"""
    
    _registry = {}
    
    @classmethod
    def register(cls, name: str):
        """데이터셋을 등록하는 데코레이터"""
        def decorator(dataset_class):
            cls._registry[name] = dataset_class
            return dataset_class
        return decorator
    
    @classmethod
    def get(cls, name: str):
        """등록된 데이터셋 가져오기"""
        if name not in cls._registry:
            raise KeyError(f"Dataset '{name}' not found. "
                         f"Available: {list(cls._registry.keys())}")
        return cls._registry[name]
    
    @classmethod
    def list_datasets(cls):
        """등록된 모든 데이터셋 목록"""
        return list(cls._registry.keys())

