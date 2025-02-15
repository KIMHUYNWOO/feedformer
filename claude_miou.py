import os
import numpy as np
import mmcv
from tqdm import tqdm

def calculate_metrics_for_files(folder1, folder2, num_classes=9, ignore_index=255):
    """두 폴더의 동일한 이름을 가진 tif 파일들의 mIoU를 계산합니다."""
    
    # tif 파일만 필터링
    files1 = set(f for f in os.listdir(folder1) if f.endswith('.tif'))
    files2 = set(f for f in os.listdir(folder2) if f.endswith('.tif'))
    
    # 공통된 파일 이름 찾기 (ValArea_001.tif ~ ValArea_210.tif 순서로 정렬)
    common_files = sorted(list(files1.intersection(files2)))
    
    if not common_files:
        raise ValueError("No matching .tif files found between the two folders.")
    
    results = []
    gt_seg_maps = []
    
    print("Loading and processing files...")
    for filename in tqdm(common_files):
        # 두 파일 읽기
        file1_path = os.path.join(folder1, filename)
        file2_path = os.path.join(folder2, filename)
        
        try:
            # tif 파일 읽기
            img1 = mmcv.imread(file1_path, flag='unchanged', backend='pillow')
            img2 = mmcv.imread(file2_path, flag='unchanged', backend='pillow')
            
            # 파일 쌍의 shape이 같은지 확인
            if img1.shape != img2.shape:
                print(f"Warning: Skipping {filename} due to shape mismatch: {img1.shape} vs {img2.shape}")
                continue
                
            results.append(img1)
            gt_seg_maps.append(img2)
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    print(f"\nProcessed {len(results)} file pairs")
    
    # mIoU 계산
    total_area_intersect = np.zeros((num_classes,), dtype=np.float64)
    total_area_union = np.zeros((num_classes,), dtype=np.float64)
    total_area_pred_label = np.zeros((num_classes,), dtype=np.float64)
    total_area_label = np.zeros((num_classes,), dtype=np.float64)
    
    print("\nCalculating metrics...")
    for pred, gt in tqdm(zip(results, gt_seg_maps)):
        # 각 클래스에 대한 영역 계산
        for i in range(num_classes):
            pred_i = pred == i
            gt_i = gt == i
            
            intersect = (pred_i & gt_i).sum()
            union = (pred_i | gt_i).sum()
            pred_label = pred_i.sum()
            label = gt_i.sum()
            
            total_area_intersect[i] += intersect
            total_area_union[i] += union
            total_area_pred_label[i] += pred_label
            total_area_label[i] += label
    
    # IoU 계산
    iou = total_area_intersect / total_area_union
    mean_iou = np.nanmean(iou)
    
    # 결과 출력
    print("\nResults:")
    print(f"Mean IoU: {mean_iou:.4f}")
    print("\nPer-class IoU:")
    for i in range(num_classes):
        print(f"Class {i}: {iou[i]:.4f}")
    
    return mean_iou, iou, common_files

# 사용 예시
if __name__ == "__main__":
    folder1 = "/data/2_data_server/cv-07/challenge/semantic_sementation/SegFormer/z_miou/aa"   # 첫 번째 폴더 경로
    folder2 = "/data/2_data_server/cv-07/challenge/semantic_sementation/SegFormer/z_miou/bb"  # 두 번째 폴더 경로
    
    mean_iou, class_ious, processed_files = calculate_metrics_for_files(folder1, folder2)
    
    # 결과를 CSV로 저장
    import pandas as pd
    
    # 클래스별 IoU 결과
    results_df = pd.DataFrame({
        'class': range(len(class_ious)),
        'iou': class_ious
    })
    results_df.loc[len(results_df)] = ['mean', mean_iou]
    results_df.to_csv('miou_results.csv', index=False)
    
    # 처리된 파일 목록도 저장
    files_df = pd.DataFrame({
        'processed_files': processed_files
    })
    files_df.to_csv('processed_files.csv', index=False)
    
    print("\nResults saved to miou_results.csv")
    print("Processed files list saved to processed_files.csv")

# # 사용 예시
# if __name__ == "__main__":
#     folder1 = "dataset/only_sar/val/labels/TrainArea_3901.tif dataset/only_sar/val/labels/TrainArea_3902.tif dataset/only_sar/val/labels/TrainArea_3903.tif dataset/only_sar/val/labels/TrainArea_3904.tif"   # 첫 번째 폴더 경로
#     folder2 = "dataset/only_sar/val/labels/TrainArea_3901.tif dataset/only_sar/val/labels/TrainArea_3902.tif dataset/only_sar/val/labels/TrainArea_3903.tif dataset/only_sar/val/labels/TrainArea_3904.tif"  # 두 번째 폴더 경로
    
#     mean_iou, class_ious = calculate_metrics_for_files(folder1, folder2)
    
#     # 결과를 CSV로 저장
#     import pandas as pd
#     results_df = pd.DataFrame({
#         'class': range(len(class_ious)),
#         'iou': class_ious
#     })
#     results_df.loc[len(results_df)] = ['mean', mean_iou]
#     results_df.to_csv('miou_results.csv', index=False)
#     print("\nResults saved to miou_results.csv")