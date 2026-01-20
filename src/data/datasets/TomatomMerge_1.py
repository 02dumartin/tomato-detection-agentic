"""Tomato_merge 1-class 데이터셋 메타정보"""
import os

class TomatomMerge1Meta:
    """
    Tomato_merge 1-class 메타 정보
    Merged dataset with single tomato class
    """
    
    NAME = "Tomato_merge_1"
    FULL_NAME = "Tomato_merge_COCO_1"
    
    CLASS_MAPPING = {"tomato": 0}
    NUM_CLASSES = 1
    DATA_ROOT = "data/Tomato_merge_1"
    
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
                'img_folder': os.path.join(split_dir, "img"),
                'ann_file': os.path.join(split_dir, ann_file)
            }
        else:
            # 기본 경로 (상대 경로)
            split_dir = os.path.join(cls.DATA_ROOT, split)
            return {
                'img_folder': os.path.join(split_dir, "img"),
                'ann_file': os.path.join(split_dir, f"custom_{split}.json")
            }

