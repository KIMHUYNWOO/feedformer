import pandas as pd
import shutil
import os

def move_high_confidence_images(csv_path, source_dir, target_dir, threshold=0.8):
    """
    mean_confidence가 threshold를 넘는 이미지들만 target_dir로 옮깁니다.
    
    Parameters:
    - csv_path: confidence scores가 저장된 CSV 파일 경로
    - source_dir: 원본 이미지가 있는 디렉토리
    - target_dir: 옮길 대상 디렉토리
    - threshold: mean_confidence의 기준값 (기본값: 0.8)
    """
    
    # CSV 파일 읽기
    df = pd.read_csv(csv_path)
    
    # threshold를 넘는 파일들만 필터링
    high_conf_files = df[df['mean_confidence'] > threshold]['image_name'].tolist()
    
    # target 디렉토리가 없으면 생성
    os.makedirs(target_dir, exist_ok=True)
    
    # 파일 이동
    moved_files = []
    for file_name in high_conf_files:
        source_path = os.path.join(source_dir, file_name)
        target_path = os.path.join(target_dir, file_name)
        
        try:
            shutil.copy2(source_path, target_path)  # copy2는 메타데이터도 보존
            moved_files.append(file_name)
            print(f"Moved: {file_name} (confidence: {df[df['image_name'] == file_name]['mean_confidence'].values[0]:.3f})")
        except Exception as e:
            print(f"Error moving {file_name}: {str(e)}")
    
    print(f"\nTotal files moved: {len(moved_files)}")
    print(f"Threshold: {threshold}")

# 사용 예시
csv_path = '/data/2_data_server/cv-07/challenge/semantic_sementation/SegFormer/confidence_scores/confidence_scores.csv'
source_dir = '/data/2_data_server/cv-07/challenge/semantic_sementation/FeedFormer/FeedFormer-master/dataset/real_raw/train/sar_images'  # 원본 이미지가 있는 디렉토리
target_dir = '/data/2_data_server/cv-07/challenge/semantic_sementation/SegFormer/dataset/threshold/train_80/images'  # 옮길 대상 디렉토리
threshold = 0.8  # mean_confidence threshold

move_high_confidence_images(csv_path, source_dir, target_dir, threshold)

# 90 275
# 80 1500
# 70 3387