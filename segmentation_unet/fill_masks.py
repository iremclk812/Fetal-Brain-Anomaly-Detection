"""
Fill HC18 boundary masks to create proper segmentation masks
HC18 annotations are boundaries (thin lines), but we need filled regions
"""
import os
import cv2
import numpy as np
from tqdm import tqdm

# Paths
PREPROCESSED_DIR = r'c:\Users\iremc\Desktop\Fetal-Brain-Anomaly-Detection\segmentation_unet\preprocessed_data'

def fill_mask(mask):
    """Fill boundary mask to create filled region"""
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return mask
    
    # Create filled mask
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    
    return filled

# Process all splits
for split in ['train', 'val', 'test']:
    masks_dir = os.path.join(PREPROCESSED_DIR, split, 'masks')
    
    if not os.path.exists(masks_dir):
        continue
    
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.png')]
    
    print(f"\nProcessing {split} set ({len(mask_files)} masks)...")
    
    before_ratios = []
    after_ratios = []
    
    for mask_file in tqdm(mask_files):
        mask_path = os.path.join(masks_dir, mask_file)
        
        # Read mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Calculate before ratio
        before_ratio = (mask > 0).sum() / mask.size
        before_ratios.append(before_ratio)
        
        # Fill mask
        filled = fill_mask(mask)
        
        # Calculate after ratio
        after_ratio = (filled > 0).sum() / filled.size
        after_ratios.append(after_ratio)
        
        # Save filled mask
        cv2.imwrite(mask_path, filled)
    
    print(f"\n{split.upper()} SET STATISTICS:")
    print(f"  Before filling: {np.mean(before_ratios)*100:.3f}% ± {np.std(before_ratios)*100:.3f}%")
    print(f"  After filling:  {np.mean(after_ratios)*100:.3f}% ± {np.std(after_ratios)*100:.3f}%")
    print(f"  Improvement: {(np.mean(after_ratios) - np.mean(before_ratios))*100:.2f}%")

print("\n✅ All masks filled successfully!")
print("🔄 Masks are now proper filled regions instead of boundaries")
