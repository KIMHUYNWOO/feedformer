import mmcv
import numpy as np
import os
from collections import Counter

# TIF 파일이 있는 디렉토리 경로
# tif_dir = '/data/2_data_server/cv-07/challenge/semantic_sementation/FeedFormer/FeedFormer-master/dataset/real_raw/train/feedformer_label_2'  # 실제 TIF 파일 경로로 변경해주세요
# tif_dir = '/data/2_data_server/cv-07/challenge/semantic_sementation/FeedFormer/FeedFormer-master/visualization/result_8_openearthmap'
# tif_dir = '/data/2_data_server/cv-07/challenge/semantic_sementation/open_earth_map/data/train/labels'
tif_dir = '/data/2_data_server/cv-07/challenge/semantic_sementation/SegFormer/dataset/threshold/80'

# 클래스 정보
class_names = {
   1: "Bareland", 
   2: "Grass",
   3: "Pavement",
   4: "Road", 
   5: "Tree",
   6: "Water",
   7: "Cropland",
   8: "buildings"
}

# 모든 파일의 클래스 통계를 저장할 딕셔너리
all_files_stats = {}

# TIF 파일 읽기
for filename in sorted(os.listdir(tif_dir)):
   if filename.endswith('.tif'):
       # TIF 파일 읽기
       tif_path = os.path.join(tif_dir, filename)
       img = mmcv.imread(tif_path, flag='unchanged')
       
       # 각 클래스별 픽셀 수 계산
       class_counts = Counter(img.flatten())
       total_pixels = img.size
       
       print(f"\nFile: {filename}")
       print(f"Shape: {img.shape}")
       print("Class distribution:")
       for class_id, count in sorted(class_counts.items()):
           percentage = (count / total_pixels) * 100
           class_name = class_names.get(class_id, f"Unknown class {class_id}")
           print(f"Class {class_id} ({class_name}): {count} pixels ({percentage:.2f}%)")
           
           # 전체 통계에 추가
           if class_id not in all_files_stats:
               all_files_stats[class_id] = 0
           all_files_stats[class_id] += count

# 전체 통계 출력
total_pixels_all = sum(all_files_stats.values())
print("\n\nOverall Statistics for all files:")
print("-" * 50)
for class_id, total_count in sorted(all_files_stats.items()):
   percentage = (total_count / total_pixels_all) * 100
   class_name = class_names.get(class_id, f"Unknown class {class_id}")
   print(f"Class {class_id} ({class_name}): {total_count} pixels ({percentage:.2f}%)")