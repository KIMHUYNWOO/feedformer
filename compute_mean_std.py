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

def calculate_channel_mean_std(image_dir):
    """
    이미지 디렉토리에서 채널별 평균과 표준편차를 계산합니다.
    
    Args:
        image_dir (str): 이미지가 저장된 디렉토리 경로
        
    Returns:
        tuple: (channel_means, channel_stds) - 각각 채널별 평균과 표준편차를 담은 numpy 배열
    """
    # 채널별 합계를 저장할 변수 초기화
    channel_sum = None
    channel_sum_squared = None
    num_pixels = 0
    
    # 모든 이미지 파일 순회
    for filename in tqdm(os.listdir(image_dir)):
        if filename.endswith(('.tif', '.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_dir, filename)
            # float32로 변환하여 계산
            img = np.array(Image.open(img_path)).astype(np.float32)
            
            # 첫 이미지에서 채널 수를 확인하고 변수 초기화
            if channel_sum is None:
                num_channels = img.shape[2] if len(img.shape) == 3 else 1
                channel_sum = np.zeros(num_channels, dtype=np.float64)
                channel_sum_squared = np.zeros(num_channels, dtype=np.float64)
            
            # 채널별 계산
            if len(img.shape) == 3:  # 다중 채널 이미지
                for c in range(img.shape[2]):
                    channel_data = img[:,:,c]
                    channel_mean = np.mean(channel_data)
                    channel_std = np.std(channel_data)
                    
                    num_pixels += channel_data.size
                    channel_sum[c] += channel_mean * channel_data.size
                    channel_sum_squared[c] += (channel_std**2 + channel_mean**2) * channel_data.size
            else:  # 단일 채널 이미지
                channel_mean = np.mean(img)
                channel_std = np.std(img)
                
                num_pixels += img.size
                channel_sum[0] += channel_mean * img.size
                channel_sum_squared[0] += (channel_std**2 + channel_mean**2) * img.size
    
    # 픽셀 수로 나누어 최종 평균 계산
    pixels_per_channel = num_pixels // len(channel_sum)
    channel_means = channel_sum / pixels_per_channel
    
    # 표준편차 계산
    channel_vars = (channel_sum_squared / pixels_per_channel) - (channel_means ** 2)
    channel_stds = np.sqrt(channel_vars)
    
    return channel_means, channel_stds

# 사용 예시
image_dir = '/data/2_data_server/cv-07/challenge/semantic_sementation/open_earth_map/data/train/images'
# mean, std = calculate_mean_std(image_dir)
# print(f"Mean: {mean}")
# print(f"Std: {std}")

image_dir = '/data/2_data_server/cv-07/challenge/semantic_sementation/open_earth_map/data/train/images'
means, stds = calculate_channel_mean_std(image_dir)

# 결과 출력
for i, (mean, std) in enumerate(zip(means, stds)):
    print(f"Channel {i}:")
    print(f"  Mean: {mean:.4f}")
    print(f"  Std:  {std:.4f}")