import cv2
from PIL import Image
import numpy as np

def check_image_depth(image_path):
    """
    TIF 이미지의 depth를 여러 방법으로 확인합니다.
    
    Args:
        image_path (str): 이미지 파일 경로
        
    Returns:
        dict: 각 방법으로 확인한 이미지 정보
    """
    result = {}
    
    # PIL로 확인
    try:
        with Image.open(image_path) as img:
            result['PIL'] = {
                'mode': img.mode,  # 이미지 모드 (L: grayscale, RGB: color 등)
                'size': img.size,  # (width, height)
                'bands': len(img.getbands())  # 채널 수
            }
    except Exception as e:
        result['PIL'] = f"PIL Error: {str(e)}"
    
    # OpenCV로 확인
    try:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            result['OpenCV'] = {
                'shape': img.shape,  # (height, width, channels) 또는 (height, width)
                'dtype': str(img.dtype),
                'channels': 1 if len(img.shape) == 2 else img.shape[2]
            }
        else:
            result['OpenCV'] = "Failed to load image"
    except Exception as e:
        result['OpenCV'] = f"OpenCV Error: {str(e)}"
    
    return result

# 사용 예시
if __name__ == "__main__":
    # 이미지 파일 경로를 지정하세요
    image_path = "/data/2_data_server/cv-07/challenge/semantic_sementation/SegFormer/result/seg_former_iter_32000/ValArea_005.png"
    
    results = check_image_depth(image_path)
    
    print("\n=== Image Information ===")
    for method, info in results.items():
        print(f"\n{method} Results:")
        if isinstance(info, dict):
            for key, value in info.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {info}")