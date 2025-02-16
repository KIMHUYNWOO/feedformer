import mmcv
import numpy as np
import os

# Load results
results = mmcv.load('/home/cv-05/FeedFormer/FeedFormer-master/pseudo_label_2.pkl')

# Create save directory
save_dir = '/home/cv-05/FeedFormer/FeedFormer-master/submit5' 
os.makedirs(save_dir, exist_ok=True)
# Process each result
for i, npy_path in enumerate(results):
    j=i+3901
    try:
        # Load numpy array from the temporary file
        result = np.load(npy_path)
        
        # Verify shape and values
        assert result.shape == (1024, 1024), f"Unexpected shape: {result.shape}"
        
        # Save as PNG
        save_path = os.path.join(save_dir, f'TrainArea_{j:03d}.tif')
        mmcv.imwrite(result.astype(np.uint8), save_path)
        
        print(f"Processed {j:03d}.tif successfully")
        
    except Exception as e:
        print(f"Error processing result {j}: {str(e)}")

print(f"Finished processing {len(results)} results")