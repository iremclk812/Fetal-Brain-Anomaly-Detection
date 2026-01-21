"""
Visualize HC18 preprocessing pipeline for U-Net segmentation
"""
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

# Load config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'config.json')

with open(CONFIG_PATH, 'r') as file:
    config = json.load(file)

PREPROCESSED_DIR = os.path.join(SCRIPT_DIR, config['paths']['dir_preprocessed'])
IMAGE_SIZE = config['preprocessing']['image_size']

def visualize_preprocessing_pipeline():
    """Show preprocessing steps for HC18 segmentation"""
    
    print("=" * 70)
    print("HC18 SEGMENTATION PREPROCESSING VISUALIZATION")
    print("=" * 70)
    
    train_img_dir = os.path.join(PREPROCESSED_DIR, 'train', 'images')
    train_mask_dir = os.path.join(PREPROCESSED_DIR, 'train', 'masks')
    
    if not os.path.exists(train_img_dir):
        print(f"❌ Preprocessed data not found at: {train_img_dir}")
        print("   Please run preprocess_hc18.py first!")
        return
    
    # Select 3 random samples
    image_files = [f for f in os.listdir(train_img_dir) if f.endswith('.png')]
    selected_files = random.sample(image_files, min(3, len(image_files)))
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    for row_idx, filename in enumerate(selected_files):
        img_path = os.path.join(train_img_dir, filename)
        mask_path = os.path.join(train_mask_dir, filename)
        
        # Load image and mask
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 1. Original preprocessed image
        axes[row_idx, 0].imshow(img, cmap='gray')
        axes[row_idx, 0].set_title(f'Image {row_idx+1}\n({IMAGE_SIZE}×{IMAGE_SIZE})', fontsize=10)
        axes[row_idx, 0].axis('off')
        
        # 2. Ground truth mask
        axes[row_idx, 1].imshow(mask, cmap='gray')
        axes[row_idx, 1].set_title('Ground Truth Mask', fontsize=10)
        axes[row_idx, 1].axis('off')
        
        # 3. Overlay (image + mask)
        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        mask_colored = np.zeros_like(overlay)
        mask_colored[:, :, 0] = mask  # Red channel
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        
        axes[row_idx, 2].imshow(overlay)
        axes[row_idx, 2].set_title('Overlay (Image + Mask)', fontsize=10)
        axes[row_idx, 2].axis('off')
        
        # 4. Normalized histogram
        axes[row_idx, 3].hist(img.ravel(), bins=50, color='blue', alpha=0.7)
        axes[row_idx, 3].set_title('Pixel Distribution', fontsize=10)
        axes[row_idx, 3].set_xlabel('Pixel Value')
        axes[row_idx, 3].set_ylabel('Frequency')
        axes[row_idx, 3].grid(alpha=0.3)
    
    plt.suptitle('HC18 Segmentation Preprocessing Pipeline', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    save_path = os.path.join(SCRIPT_DIR, 'preprocessing_visualization.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {save_path}")
    plt.show()

def visualize_dataset_statistics():
    """Show dataset statistics"""
    
    splits = ['train', 'val', 'test']
    counts = {}
    
    for split in splits:
        img_dir = os.path.join(PREPROCESSED_DIR, split, 'images')
        if os.path.exists(img_dir):
            counts[split] = len([f for f in os.listdir(img_dir) if f.endswith('.png')])
        else:
            counts[split] = 0
    
    # Bar chart
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    splits_names = list(counts.keys())
    values = list(counts.values())
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    bars = ax.bar(splits_names, values, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on top of bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Split', fontsize=14)
    ax.set_ylabel('Number of Images', fontsize=14)
    ax.set_title('HC18 Dataset Distribution', fontsize=16, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(SCRIPT_DIR, 'dataset_statistics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Statistics saved to: {save_path}")
    plt.show()
    
    # Print statistics
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    total = sum(counts.values())
    for split, count in counts.items():
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{split.upper()}: {count} images ({percentage:.1f}%)")
    print(f"TOTAL: {total} images")
    print("=" * 70)

def compare_augmentations():
    """Show original vs augmented samples"""
    
    train_img_dir = os.path.join(PREPROCESSED_DIR, 'train', 'images')
    train_mask_dir = os.path.join(PREPROCESSED_DIR, 'train', 'masks')
    
    if not os.path.exists(train_img_dir):
        print(f"❌ Preprocessed data not found!")
        return
    
    # Find base images (without augmentation suffix)
    all_files = [f for f in os.listdir(train_img_dir) if f.endswith('.png')]
    
    # Group by base name (before _0, _1, _2)
    base_groups = {}
    for f in all_files:
        # Extract base name (everything before last underscore if it's a number)
        parts = f.replace('.png', '').split('_')
        if parts[-1].isdigit():
            base = '_'.join(parts[:-1])
        else:
            base = '_'.join(parts)
        
        if base not in base_groups:
            base_groups[base] = []
        base_groups[base].append(f)
    
    # Pick a group with multiple augmentations
    selected_group = None
    for base, files in base_groups.items():
        if len(files) > 1:
            selected_group = sorted(files)[:4]  # Take first 4
            break
    
    if not selected_group:
        print("⚠️ No augmented samples found")
        return
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for idx, filename in enumerate(selected_group):
        img_path = os.path.join(train_img_dir, filename)
        mask_path = os.path.join(train_mask_dir, filename)
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Image
        axes[0, idx].imshow(img, cmap='gray')
        axes[0, idx].set_title(f'Aug {idx}', fontsize=10)
        axes[0, idx].axis('off')
        
        # Mask
        axes[1, idx].imshow(mask, cmap='gray')
        axes[1, idx].set_title(f'Mask {idx}', fontsize=10)
        axes[1, idx].axis('off')
    
    axes[0, 0].set_ylabel('Image', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Mask', fontsize=12, fontweight='bold')
    
    plt.suptitle('Data Augmentation Examples (Same Original, Different Transforms)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(SCRIPT_DIR, 'augmentation_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Augmentation comparison saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    print("\n🎨 Generating visualizations...\n")
    
    visualize_preprocessing_pipeline()
    print()
    visualize_dataset_statistics()
    print()
    compare_augmentations()
    
    print("\n" + "=" * 70)
    print("✅ ALL VISUALIZATIONS COMPLETED!")
    print("=" * 70)
