import numpy as np
from PIL import Image
import os

import numpy as np
from PIL import Image
import os
from tqdm import tqdm  # 진행상황 확인용

def calculate_mean_std(image_dir):
    # 중간 계산값들을 float64로 유지
    pixel_sum = 0.0
    pixel_sum_squared = 0.0
    num_pixels = 0
    
    # 모든 이미지 파일 순회
    for filename in tqdm(os.listdir(image_dir)):
        if filename.endswith(('.tif', '.png', '.jpg')):
            img_path = os.path.join(image_dir, filename)
            # float32로 변환하여 계산
            img = np.array(Image.open(img_path)).astype(np.float32)
            
            # 이미지별로 평균을 먼저 계산
            img_mean = np.mean(img)
            img_std = np.std(img)
            
            num_pixels += img.size
            pixel_sum += img_mean * img.size
            pixel_sum_squared += (img_std ** 2 + img_mean ** 2) * img.size
    
    # 전체 평균
    mean = pixel_sum / num_pixels
    
    # 전체 표준편차
    var = (pixel_sum_squared / num_pixels) - (mean ** 2)
    std = np.sqrt(var)
    
    return float(mean), float(std)

# 사용 예시
image_dir = '/data/2_data_server/cv-07/challenge/semantic_sementation/FeedFormer/FeedFormer-master/dataset/real_raw/train/sar_images'
mean, std = calculate_mean_std(image_dir)
print(f"Mean: {mean}")
print(f"Std: {std}")