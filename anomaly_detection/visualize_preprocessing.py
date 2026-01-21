"""
Visualize preprocessing results for Binary Plane Classification
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

# Binary classes
CLASS_NAMES = ['Trans-thalamic', 'Other']

def visualize_preprocessing_pipeline():
    """Show preprocessing steps: Original → Resized → Normalized"""
    
    print("=" * 70)
    print("PREPROCESSING PIPELINE VISUALIZATION")
    print("=" * 70)
    
    # Select random samples from each class
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for row_idx, class_name in enumerate(CLASS_NAMES):
        train_dir = os.path.join(PREPROCESSED_DIR, 'train')
        class_files = [f for f in os.listdir(train_dir) if class_name in f]
        
        if len(class_files) == 0:
            print(f"⚠️ No files found for class: {class_name}")
            continue
        
        # Pick random sample
        sample_file = random.choice(class_files)
        img_path = os.path.join(train_dir, sample_file)
        
        # Load image
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Original (preprocessed)
        axes[row_idx, 0].imshow(img_rgb)
        axes[row_idx, 0].set_title(f'{class_name}\nPreprocessed Image', fontsize=12, fontweight='bold')
        axes[row_idx, 0].axis('off')
        
        # Normalized (0-1 range visualization)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        axes[row_idx, 1].imshow(img_normalized)
        axes[row_idx, 1].set_title('Normalized (0-1)', fontsize=12)
        axes[row_idx, 1].axis('off')
        
        # Histogram
        axes[row_idx, 2].hist(img_rgb.ravel(), bins=50, color='gray', alpha=0.7)
        axes[row_idx, 2].set_title('Pixel Distribution', fontsize=12)
        axes[row_idx, 2].set_xlabel('Pixel Value')
        axes[row_idx, 2].set_ylabel('Frequency')
        axes[row_idx, 2].grid(alpha=0.3)
    
    plt.suptitle('Preprocessing Pipeline: Binary Plane Classification', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    save_path = os.path.join(SCRIPT_DIR, 'preprocessing_visualization.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {save_path}")
    plt.show()

def visualize_class_distribution():
    """Show class distribution across splits"""
    
    splits = ['train', 'val', 'test']
    class_counts = {split: {cls: 0 for cls in CLASS_NAMES} for split in splits}
    
    for split in splits:
        split_dir = os.path.join(PREPROCESSED_DIR, split)
        if not os.path.exists(split_dir):
            continue
        
        for filename in os.listdir(split_dir):
            for class_name in CLASS_NAMES:
                if class_name in filename:
                    class_counts[split][class_name] += 1
                    break
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart per split
    x = np.arange(len(CLASS_NAMES))
    width = 0.25
    
    for i, split in enumerate(splits):
        counts = [class_counts[split][cls] for cls in CLASS_NAMES]
        axes[0].bar(x + i*width, counts, width, label=split.capitalize())
    
    axes[0].set_xlabel('Class', fontsize=12)
    axes[0].set_ylabel('Number of Images', fontsize=12)
    axes[0].set_title('Class Distribution per Split', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(CLASS_NAMES, rotation=0)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Pie chart for train set
    train_counts = [class_counts['train'][cls] for cls in CLASS_NAMES]
    colors = ['#2ecc71', '#e74c3c']
    
    axes[1].pie(train_counts, labels=CLASS_NAMES, autopct='%1.1f%%',
                colors=colors, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    axes[1].set_title('Training Set Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    save_path = os.path.join(SCRIPT_DIR, 'class_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Class distribution saved to: {save_path}")
    plt.show()
    
    # Print statistics
    print("\n" + "=" * 70)
    print("CLASS DISTRIBUTION STATISTICS")
    print("=" * 70)
    for split in splits:
        total = sum(class_counts[split].values())
        print(f"\n{split.upper()}: {total} images")
        for class_name in CLASS_NAMES:
            count = class_counts[split][class_name]
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  - {class_name}: {count} ({percentage:.1f}%)")

def visualize_sample_grid():
    """Show grid of sample images from each class"""
    
    samples_per_class = 6
    fig, axes = plt.subplots(len(CLASS_NAMES), samples_per_class, figsize=(18, 6))
    
    train_dir = os.path.join(PREPROCESSED_DIR, 'train')
    
    for row_idx, class_name in enumerate(CLASS_NAMES):
        class_files = [f for f in os.listdir(train_dir) if class_name in f]
        selected = random.sample(class_files, min(samples_per_class, len(class_files)))
        
        for col_idx, filename in enumerate(selected):
            img_path = os.path.join(train_dir, filename)
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            axes[row_idx, col_idx].imshow(img_rgb)
            axes[row_idx, col_idx].axis('off')
            
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel(class_name, fontsize=12, fontweight='bold', rotation=0, 
                                                    ha='right', va='center')
    
    plt.suptitle('Sample Images: Binary Plane Classification', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(SCRIPT_DIR, 'sample_grid.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Sample grid saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    print("\n🎨 Generating visualizations...\n")
    
    # Check if preprocessed data exists
    if not os.path.exists(PREPROCESSED_DIR):
        print(f"❌ Preprocessed data not found at: {PREPROCESSED_DIR}")
        print("   Please run preprocess_data.py first!")
        exit(1)
    
    # Generate all visualizations
    visualize_preprocessing_pipeline()
    print()
    visualize_class_distribution()
    print()
    visualize_sample_grid()
    
    print("\n" + "=" * 70)
    print("✅ ALL VISUALIZATIONS COMPLETED!")
    print("=" * 70)
