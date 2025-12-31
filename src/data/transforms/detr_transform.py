"""DETR용 데이터 변환 및 Dataset"""
import torchvision
from torchvision import transforms as T
from transformers import DetrImageProcessor

class DetrCocoDataset(torchvision.datasets.CocoDetection):
    """
    DETR 학습을 위한 커스텀 COCO Detection 데이터셋 클래스
    """
    
    def __init__(
        self, 
        img_folder: str, 
        ann_file: str,
        imageprocessor: DetrImageProcessor, 
        train: bool = True
    ):
        super(DetrCocoDataset, self).__init__(img_folder, ann_file)
        
        self.imageprocessor = imageprocessor
        self.train = train
        
        # 학습 시 색상/블러 기반 증강 적용 (bbox 수정 불필요)
        self.augment = (
            T.Compose([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.2),
            ])
            if train
            else None
        )

    def __getitem__(self, idx: int):
        # PIL 이미지와 COCO 형식의 타겟 읽기
        img, target = super().__getitem__(idx)

        # 학습 시 이미지 증강
        if self.augment is not None:
            img = self.augment(img)

        # DETR 형식으로 이미지와 타겟 전처리
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.imageprocessor(images=img, annotations=target, return_tensors="pt")
        
        # 배치 차원 제거
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target

    @staticmethod
    def create_collate_fn(imageprocessor: DetrImageProcessor):
        """
        DETR용 collate function 생성
        Dataset과 함께 사용하는 collate_fn
        """
        def collate_fn(batch):
            pixel_values = [item[0] for item in batch]
            labels = [item[1] for item in batch]
            
            encoding = imageprocessor.pad(pixel_values, return_tensors="pt")
            
            return {
                'pixel_values': encoding['pixel_values'],
                'pixel_mask': encoding['pixel_mask'],
                'labels': labels
            }
        return collate_fn

def create_detr_dataset(dataset_meta, split: str, imageprocessor, config=None):
    """DETR Dataset 생성"""
    paths = dataset_meta.get_data_paths(split, config=config)
    train = (split == "train")
    
    return DetrCocoDataset(
        paths['img_folder'],
        paths['ann_file'],
        imageprocessor,
        train
    )