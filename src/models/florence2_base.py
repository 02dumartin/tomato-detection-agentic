"""
Florence-2 Base Model
Zero-shot과 Fine-tuned 공통 인터페이스
"""

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
from typing import List, Dict, Optional
from pathlib import Path
import os


class Florence2Base:
    """Florence-2 기본 클래스"""
    
    def __init__(
        self,
        model_name: str = "microsoft/Florence-2-base",
        device: str = "cuda",
        is_finetuned: bool = False,
        checkpoint_path: Optional[str] = None,
        config: Optional[Dict] = None 
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.is_finetuned = is_finetuned
        self.model_name = model_name
        self.config = config or {}  
        
        # Generation 옵션 읽기
        florence2_config = self.config.get('florence2', {})
        gen_config = florence2_config.get('generation', {})
        self.generation_kwargs = {
            'max_new_tokens': gen_config.get('max_new_tokens', 1024),
            'num_beams': gen_config.get('num_beams', 1),
            'do_sample': gen_config.get('do_sample', False),
            'use_cache': gen_config.get('use_cache', False),
        }
        
        print(f"Loading Florence-2 model on {self.device}...")
        
        # 프로세서 로드 (항상 원본 사용)
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # 모델 로드
        if is_finetuned and checkpoint_path:
            # Fine-tuned 체크포인트 로드
            self.model = self._load_finetuned_model(checkpoint_path)
        else:
            # 원본 모델 로드 (Zero-shot)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                attn_implementation="eager"
            )
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"  Florence-2 loaded successfully!")
        print(f"   Model dtype: {next(self.model.parameters()).dtype}")
    
    def _load_finetuned_model(self, checkpoint_path):
        """Fine-tuned 체크포인트 로드 (LoRA 지원)"""
        checkpoint_path = Path(checkpoint_path)
        
        print(f"Loading fine-tuned checkpoint from {checkpoint_path}...")
        
        # Base 모델 로드
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            attn_implementation="eager"
        )
        
        # LoRA 체크포인트인지 확인 (디렉토리 + adapter_config.json)
        if checkpoint_path.is_dir() and (checkpoint_path / "adapter_config.json").exists():
            print("   Detected LoRA checkpoint, loading with PEFT...")
            from peft import PeftModel
            
            model = PeftModel.from_pretrained(
                model,
                str(checkpoint_path),
                is_trainable=False
            )
            print("   LoRA adapter loaded!")
        
        # 일반 PyTorch 체크포인트 (.pt 파일)
        elif checkpoint_path.is_file():
            print("   Loading standard PyTorch checkpoint...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("   Checkpoint loaded!")
        
        else:
            raise ValueError(f"Invalid checkpoint path: {checkpoint_path}")
        
        print("Fine-tuned model loaded successfully!")
        
        return model
    
    def predict(
        self,
        image_path: str,
        task: str = "<OD>",  
        **generate_kwargs
    ) -> Dict:
        """
        기본 추론 메서드
        
        Args:
            image_path: 이미지 경로
            task: Florence-2 task prompt
            **generate_kwargs: generation 파라미터
        
        Returns:
            dict: 탐지 결과
        """
        # 이미지 로드
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path  # PIL Image 객체
        
        # 입력 준비
        inputs = self.processor(
            text=task,
            images=image,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in inputs.items()}
        
        # Config에서 읽은 generation 파라미터 사용
        default_kwargs = self.generation_kwargs.copy()
        # 함수 호출 시 전달된 kwargs로 덮어쓰기 (우선순위 높음)
        default_kwargs.update(generate_kwargs)
        
        # 추론
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **default_kwargs)
        
        # 디코딩
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        
        # 후처리
        result = self.processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(image.width, image.height)
        )
        
        return result
    
    def __repr__(self):
        mode = "Fine-tuned" if self.is_finetuned else "Zero-shot"
        return f"Florence2Base(mode={mode}, device={self.device})"

