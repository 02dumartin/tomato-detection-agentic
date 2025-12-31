"""TomatOD 3-class 데이터셋 메타정보"""
import os

class TomatOD3Meta:
    """
    TomatOD 3-class 메타 정보
    모델과 독립적인 순수 데이터셋 정보만 포함
    """
    
    NAME = "TomatOD_3"
    FULL_NAME = "TomatOD_COCO_3"
    
    CLASS_MAPPING = {
        "fully-ripe": 0,
        "semi-ripe": 1,
        "unripe": 2
    }
    
    NUM_CLASSES = 3
    DATA_ROOT = "data/TomatOD_COCO_3"
    
    @classmethod
    def get_class_names(cls):
        """클래스 이름 리스트 반환"""
        return list(cls.CLASS_MAPPING.keys())
    
    @classmethod
    def get_id2label(cls):
        """ID -> Label 매핑"""
        return {v: k for k, v in cls.CLASS_MAPPING.items()}
    
    @classmethod
    def get_label2id(cls):
        """Label -> ID 매핑"""
        return cls.CLASS_MAPPING
    
    @classmethod
    def get_data_paths(cls, split: str, config=None):
        """
        split에 따른 경로 반환
        config가 주어지면 config의 경로 사용, 아니면 기본 경로 사용
        """
        if config:
            # Config에서 경로 가져오기
            split_dir = config['data'].get(f'{split}_dir', 
                                          os.path.join(config['data']['data_root'], split))
            ann_file = config['data'].get(f'{split}_ann_file', f'custom_{split}.json')
            
            return {
                'img_folder': os.path.join(split_dir, "images"),
                'ann_file': os.path.join(split_dir, ann_file)
            }
        else:
            # 기본 경로 (상대 경로)
            split_dir = os.path.join(cls.DATA_ROOT, split)
            return {
                'img_folder': os.path.join(split_dir, "images"),
                'ann_file': os.path.join(split_dir, f"custom_{split}.json")
            }

