"""
Check original HC18 masks vs preprocessed masks
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Paths
ORIGINAL_DIR = r'C:\Users\iremc\Desktop\Fetal-Brain-Anomaly-Detection\1327317\training_set\training_set'
PREPROCESSED_DIR = r'c:\Users\iremc\Desktop\Fetal-Brain-Anomaly-Detection\segmentation_unet\preprocessed_data\train'

# Check first image
original_img = os.path.join(ORIGINAL_DIR, '001_HC.png')
original_mask = os.path.join(ORIGINAL_DIR, '001_HC_Annotation.png')

# Check if they exist
if os.path.exists(original_img):
    img = cv2.imread(original_img, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(original_mask, cv2.IMREAD_GRAYSCALE)
    
    print(f"Original Image shape: {img.shape}")
    print(f"Original Mask shape: {mask.shape}")
    print(f"Mask unique values: {np.unique(mask)}")
    print(f"Mask ratio: {(mask > 0).sum() / mask.size * 100:.2f}%")
    
    # Find preprocessed version
    preprocessed_files = os.listdir(os.path.join(PREPROCESSED_DIR, 'images'))
    matching = [f for f in preprocessed_files if '001_HC' in f]
    
    if matching:
        prep_img = cv2.imread(os.path.join(PREPROCESSED_DIR, 'images', matching[0]))
        prep_mask = cv2.imread(os.path.join(PREPROCESSED_DIR, 'masks', matching[0]), cv2.IMREAD_GRAYSCALE)
        
        print(f"\nPreprocessed Image shape: {prep_img.shape}")
        print(f"Preprocessed Mask shape: {prep_mask.shape}")
        print(f"Preprocessed Mask unique values: {np.unique(prep_mask)}")
        print(f"Preprocessed Mask ratio: {(prep_mask > 0).sum() / prep_mask.size * 100:.2f}%")
        
        # Visualize
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes[0, 0].imshow(img, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 1].imshow(mask, cmap='gray')
        axes[0, 1].set_title(f'Original Mask ({(mask > 0).sum() / mask.size * 100:.2f}%)')
        
        axes[1, 0].imshow(prep_img)
        axes[1, 0].set_title('Preprocessed Image (256x256)')
        axes[1, 1].imshow(prep_mask, cmap='gray')
        axes[1, 1].set_title(f'Preprocessed Mask ({(prep_mask > 0).sum() / prep_mask.size * 100:.2f}%)')
        
        plt.tight_layout()
        plt.savefig('original_vs_preprocessed.png', dpi=150)
        print("\n✅ Saved comparison: original_vs_preprocessed.png")
        plt.show()
else:
    print("❌ Original files not found!")
