"""
Visualize training samples to check if masks are correct
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# Paths
PREPROCESSED_DIR = r'c:\Users\iremc\Desktop\Fetal-Brain-Anomaly-Detection\segmentation_unet\preprocessed_data'
TRAIN_IMAGES_DIR = os.path.join(PREPROCESSED_DIR, 'train', 'images')
TRAIN_MASKS_DIR = os.path.join(PREPROCESSED_DIR, 'train', 'masks')

# Get random samples
image_files = [f for f in os.listdir(TRAIN_IMAGES_DIR) if f.endswith('.png')]
random_samples = random.sample(image_files, min(12, len(image_files)))

# Create visualization
fig, axes = plt.subplots(4, 6, figsize=(18, 12))
fig.suptitle('Training Samples - Images and Masks', fontsize=16)

for idx, img_name in enumerate(random_samples):
    row = idx // 3
    col_img = (idx % 3) * 2
    col_mask = col_img + 1
    
    # Load image and mask
    img_path = os.path.join(TRAIN_IMAGES_DIR, img_name)
    mask_path = os.path.join(TRAIN_MASKS_DIR, img_name)
    
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Calculate mask statistics
    mask_ratio = (mask > 0).sum() / mask.size
    unique_vals = np.unique(mask)
    
    # Display image
    axes[row, col_img].imshow(image)
    axes[row, col_img].set_title(f'{img_name[:15]}...', fontsize=8)
    axes[row, col_img].axis('off')
    
    # Display mask
    axes[row, col_mask].imshow(mask, cmap='gray')
    axes[row, col_mask].set_title(f'Mask: {mask_ratio*100:.2f}%\nVals: {unique_vals}', fontsize=8)
    axes[row, col_mask].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(PREPROCESSED_DIR, 'training_samples_visualization.png'), dpi=150, bbox_inches='tight')
print(f"\n✅ Saved visualization: {os.path.join(PREPROCESSED_DIR, 'training_samples_visualization.png')}")

# Calculate overall statistics
print("\n" + "="*60)
print("MASK STATISTICS")
print("="*60)

all_mask_ratios = []
for img_name in image_files[:100]:  # Check first 100
    mask_path = os.path.join(TRAIN_MASKS_DIR, img_name)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_ratio = (mask > 0).sum() / mask.size
    all_mask_ratios.append(mask_ratio)

print(f"Average mask ratio: {np.mean(all_mask_ratios)*100:.3f}%")
print(f"Min mask ratio: {np.min(all_mask_ratios)*100:.3f}%")
print(f"Max mask ratio: {np.max(all_mask_ratios)*100:.3f}%")
print(f"Std mask ratio: {np.std(all_mask_ratios)*100:.3f}%")

plt.show()
