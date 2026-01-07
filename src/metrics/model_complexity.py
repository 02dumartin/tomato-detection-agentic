"""모델 복잡도 및 성능 측정"""
import torch
from typing import Dict


def calculate_model_complexity(model, config: Dict) -> Dict:
    """
    모델 복잡도 계산 (Parameters, Size, GFLOPs)
    
    Args:
        model: 평가할 모델
        config: 설정 딕셔너리
    
    Returns:
        complexity_metrics: 복잡도 메트릭 딕셔너리
    """
    # 1. Parameters 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 2. Model Size 계산 (MB)
    model_size_mb = sum(p.numel() * 4 for p in model.parameters()) / (1024**2)  # float32 = 4 bytes
    
    # 3. GFLOPs 계산 (더미 입력 사용)
    model.eval()
    device = next(model.parameters()).device
    
    # 입력 크기 가져오기
    if 'image_size' in config.get('data', {}):
        img_size = config['data']['image_size']
    elif 'imgsz' in config.get('data', {}):
        img_size = config['data']['imgsz']
    else:
        img_size = 640  # 기본값
    
    # 더미 입력 생성 (배치 크기 1)
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
    
    gflops = None
    flops_formatted = None
    
    try:
        # thop을 사용한 GFLOPs 계산
        from thop import profile, clever_format
        
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        gflops = flops / 1e9  # GFLOPs로 변환
        
        # 또는 clever_format 사용
        flops_formatted, params_formatted = clever_format([flops, params], "%.3f")
    except ImportError:
        print("  thop not installed. GFLOPs calculation skipped.")
        print("   Install with: pip install thop")
    except Exception as e:
        print(f"  Could not calculate GFLOPs: {e}")
    
    return {
        'total_params': int(total_params),
        'trainable_params': int(trainable_params),
        'params_m': total_params / 1e6,
        'model_size_mb': model_size_mb,
        'gflops': gflops,
        'gflops_formatted': flops_formatted
    }

