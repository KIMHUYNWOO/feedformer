import numpy as np
import cv2
import os

# Create output directory if it doesn't exist
output_dir = '/data/2_data_server/cv-07/challenge/semantic_sementation/SegFormer/random_value_5'
os.makedirs(output_dir, exist_ok=True)

# Generate 210 images
for i in range(210):
    # For each image, select one random value between 1 and 9
    # value = np.random.randint(1, 10)
    #2 6 8
    value = 5
    
    # Create image filled with the selected value
    image = np.full((1024, 1024), value, dtype=np.uint8)
    
    # Save the image with a numbered filename\
    j = i + 1
    filename = os.path.join(output_dir, f'ValArea_{j:03d}.png')
    cv2.imwrite(filename, image)
    
    # Print progress
    print(f"Created image {i+1}/210 with value {value}")

# Print summary
print("\nComplete! All images have been saved to the 'random_images' directory.")