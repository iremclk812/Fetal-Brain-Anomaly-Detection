"""
Preprocess HC18 dataset for U-Net segmentation
Loads images and annotation masks, applies augmentation
"""
import os
import json
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import albumentations as A
from sklearn.model_selection import train_test_split

# Load config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'config.json')

with open(CONFIG_PATH, 'r') as file:
    config = json.load(file)

# Paths
HC18_DIR = config['paths']['hc18_dir']
HC18_TRAIN_DIR = config['paths']['hc18_train_dir']
SAVE_DIR = os.path.join(SCRIPT_DIR, config['paths']['dir_preprocessed'])

IMAGE_SIZE = config['preprocessing']['image_size']
N_AUGMENTATIONS = config['preprocessing']['augmentation']['n_augmentations_train']
TRAIN_VAL_SPLIT = config['training']['train_val_split']

print("=" * 70)
print("HC18 DATASET PREPROCESSING FOR U-NET SEGMENTATION")
print("=" * 70)
print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Augmentations per train image: {N_AUGMENTATIONS}")
print("=" * 70)

# Create directories
os.makedirs(os.path.join(SAVE_DIR, 'train', 'images'), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, 'train', 'masks'), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, 'val', 'images'), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, 'val', 'masks'), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, 'test', 'images'), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, 'test', 'masks'), exist_ok=True)

# Augmentation pipeline
aug_config = config['preprocessing']['augmentation']
AUGMENTATION = A.Compose([
    A.ShiftScaleRotate(
        shift_limit=aug_config['shift_limit'],
        scale_limit=aug_config['scale_limit'],
        rotate_limit=aug_config['rotation_range'],
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        mask_value=0,
        p=1.0
    ),
    A.ElasticTransform(
        alpha=1, sigma=50, alpha_affine=50,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        value=0, mask_value=0,
        p=0.5 if aug_config['elastic_transform'] else 0.0
    ),
    A.GridDistortion(
        num_steps=5, distort_limit=0.3,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        value=0, mask_value=0,
        p=0.5 if aug_config['grid_distortion'] else 0.0
    ),
    A.OpticalDistortion(
        distort_limit=0.3, shift_limit=0.3,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        value=0, mask_value=0,
        p=0.5 if aug_config['optical_distortion'] else 0.0
    ),
])

RESIZE = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE, interpolation=cv2.INTER_CUBIC, always_apply=True)
])

def find_hc18_data():
    """Find HC18 images and annotations"""
    # Try standard structure first
    if os.path.exists(HC18_TRAIN_DIR):
        train_images = list(Path(HC18_TRAIN_DIR).glob('*.png'))
        train_annotations = list(Path(HC18_TRAIN_DIR).glob('*_Annotation.png'))
        
        # Filter out annotation files from images
        train_images = [img for img in train_images if '_Annotation' not in str(img)]
        
        print(f"Found {len(train_images)} training images")
        print(f"Found {len(train_annotations)} training annotations")
        
        return train_images, train_annotations
    else:
        # Search recursively
        print(f"Standard structure not found. Searching in: {HC18_DIR}")
        all_images = list(Path(HC18_DIR).rglob('*.png'))
        
        images = [img for img in all_images if '_Annotation' not in str(img)]
        annotations = [img for img in all_images if '_Annotation' in str(img)]
        
        print(f"Found {len(images)} images")
        print(f"Found {len(annotations)} annotations")
        
        return images, annotations

def create_annotation_mapping(images, annotations):
    """Map images to their annotations"""
    mapping = {}
    
    for img_path in images:
        img_name = img_path.stem
        
        # Look for corresponding annotation
        for ann_path in annotations:
            ann_name = ann_path.stem
            
            # Check if annotation matches image (remove _Annotation suffix)
            if ann_name.replace('_Annotation', '') == img_name:
                mapping[str(img_path)] = str(ann_path)
                break
    
    print(f"Matched {len(mapping)} image-annotation pairs")
    return mapping

def process_and_save(img_path, mask_path, split, idx=0, augment=False):
    """Process image-mask pair and save"""
    try:
        # Load image and mask
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            print(f"Failed to load: {img_path}")
            return 0
        
        # Convert to RGB for image (3 channels expected)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Binarize mask
        mask = (mask > 127).astype(np.uint8) * 255
        
        count = 0
        
        if augment and split == 'train':
            # Apply augmentation multiple times
            for i in range(N_AUGMENTATIONS):
                augmented = AUGMENTATION(image=image, mask=mask)
                aug_image = augmented['image']
                aug_mask = augmented['mask']
                
                # Resize
                resized = RESIZE(image=aug_image, mask=aug_mask)
                final_image = resized['image']
                final_mask = resized['mask']
                
                # Save
                img_name = f"{Path(img_path).stem}_{i}.png"
                cv2.imwrite(
                    os.path.join(SAVE_DIR, split, 'images', img_name),
                    final_image
                )
                cv2.imwrite(
                    os.path.join(SAVE_DIR, split, 'masks', img_name),
                    final_mask
                )
                count += 1
        else:
            # No augmentation for val/test
            resized = RESIZE(image=image, mask=mask)
            final_image = resized['image']
            final_mask = resized['mask']
            
            img_name = f"{Path(img_path).stem}.png"
            cv2.imwrite(
                os.path.join(SAVE_DIR, split, 'images', img_name),
                final_image
            )
            cv2.imwrite(
                os.path.join(SAVE_DIR, split, 'masks', img_name),
                final_mask
            )
            count += 1
        
        return count
    
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return 0

# Find data
images, annotations = find_hc18_data()
img_ann_mapping = create_annotation_mapping(images, annotations)

if len(img_ann_mapping) == 0:
    print("\nERROR: No image-annotation pairs found!")
    print("Please check the HC18 dataset structure.")
    exit(1)

# Split data
img_paths = list(img_ann_mapping.keys())
# First split: 85% (train+val) / 15% (test)
train_val_paths, test_paths = train_test_split(img_paths, train_size=0.85, random_state=42)
# Second split: 75% train / 25% val (from train+val)
train_paths, val_paths = train_test_split(train_val_paths, train_size=TRAIN_VAL_SPLIT, random_state=42)

print(f"\nData split:")
print(f"  Train: {len(train_paths)}")
print(f"  Val:   {len(val_paths)}")
print(f"  Test:  {len(test_paths)}")

# Process train
print("\nProcessing TRAIN set...")
train_count = 0
for img_path in tqdm(train_paths):
    mask_path = img_ann_mapping[img_path]
    train_count += process_and_save(img_path, mask_path, 'train', augment=True)

# Process val
print("Processing VAL set...")
val_count = 0
for img_path in tqdm(val_paths):
    mask_path = img_ann_mapping[img_path]
    val_count += process_and_save(img_path, mask_path, 'val', augment=False)

# Process test
print("Processing TEST set...")
test_count = 0
for img_path in tqdm(test_paths):
    mask_path = img_ann_mapping[img_path]
    test_count += process_and_save(img_path, mask_path, 'test', augment=False)

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE!")
print("=" * 70)
print(f"Train: {train_count} images")
print(f"Val:   {val_count} images")
print(f"Test:  {test_count} images")
print(f"TOTAL: {train_count + val_count + test_count}")
print(f"\nData saved to: {SAVE_DIR}")
print("=" * 70)
