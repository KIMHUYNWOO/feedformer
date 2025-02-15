import numpy as np
import cv2
import os
from tqdm import tqdm  # for progress bar

# Create output directory if it doesn't exist
output_dir = 'random_pixel_images'
os.makedirs(output_dir, exist_ok=True)

# Image parameters
height, width = 1024, 1024

# Generate 210 images with progress bar
for i in tqdm(range(210), desc="Generating images"):
    # Create random image with values 1-9 for each pixel
    random_image = np.random.randint(1, 10, size=(height, width), dtype=np.uint8)
    
    # Save the image
    j = i + 1
    filename = os.path.join(output_dir, f'ValArea_{j:03d}.png')
    cv2.imwrite(filename, random_image)

print("\nComplete! All images have been saved to the 'random_pixel_images' directory.")

# # Optional: Verify one random image
# random_idx = np.random.randint(0, 210)
# verify_image = cv2.imread(os.path.join(output_dir, f'ValArea_{random_idx:03d}.png'), 
#                          cv2.IMREAD_UNCHANGED)
# print(f"\nVerifying image_{random_idx:03d}.png:")
# print("Shape:", verify_image.shape)
# print("Dtype:", verify_image.dtype)
# print("Value range:", verify_image.min(), "-", verify_image.max())
# print("Unique values:", np.unique(verify_image))