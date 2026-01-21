"""
Preprocess FETAL_PLANES_DB for BINARY classification:
- Class 0: Trans-thalamic (valid for HC measurement)
- Class 1: Other (Trans-cerebellum, Trans-ventricular, other planes)
"""
import os
import json
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split

# Load config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'config.json')

with open(CONFIG_PATH, 'r') as file:
    config = json.load(file)

# Paths
FETAL_PLANES_DIR = config['paths']['fetal_planes_dir']
FETAL_PLANES_CSV = config['paths']['fetal_planes_csv']
SAVE_PREPROCESSED_DIR = os.path.join(SCRIPT_DIR, config['paths']['dir_preprocessed'])

# Preprocessing params
IMAGE_SIZE = config['preprocessing']['image_size']
N_AUGMENTATIONS = config['preprocessing']['augmentation']['n_augmentations_train']
TRAIN_VAL_SPLIT = config['training']['train_val_split']

# BINARY CLASS MAPPING
DICT_CLSNAME_TO_CLSINDEX = {
    'Trans-thalamic': 0,  # Valid for HC measurement
    'Other': 1,           # All other planes (Trans-cerebellum, Trans-ventricular, etc.)
}

print("=" * 70)
print("FETAL BRAIN BINARY PLANE CLASSIFICATION - PREPROCESSING")
print("=" * 70)
print(f"Classes: {list(DICT_CLSNAME_TO_CLSINDEX.keys())}")
print(f"  Class 0 (Trans-thalamic): Valid for HC measurement")
print(f"  Class 1 (Other): Trans-cerebellum, Trans-ventricular, Other planes")
print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Augmentations per train image: {N_AUGMENTATIONS}")
print(f"Train/Val split: {TRAIN_VAL_SPLIT}/{1-TRAIN_VAL_SPLIT}")
print("=" * 70)

# Create directories
os.makedirs(os.path.join(SAVE_PREPROCESSED_DIR, 'train'), exist_ok=True)
os.makedirs(os.path.join(SAVE_PREPROCESSED_DIR, 'val'), exist_ok=True)
os.makedirs(os.path.join(SAVE_PREPROCESSED_DIR, 'test'), exist_ok=True)

# Augmentation pipeline (for training only)
aug_config = config['preprocessing']['augmentation']
AUGMENTATION = A.Compose([
    A.ColorJitter(
        brightness=aug_config['color_jitter_brightness'],
        contrast=aug_config['color_jitter_contrast'],
        saturation=aug_config['color_jitter_saturation'],
        hue=aug_config['color_jitter_hue'],
        p=0.5
    ),
    A.CLAHE(clip_limit=aug_config['clahe_clip_limit'], p=0.5),
    A.ShiftScaleRotate(
        shift_limit=aug_config['width_shift_range'],
        scale_limit=0.0,
        rotate_limit=aug_config['rotation_range'],
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        p=1.0
    ),
    A.HorizontalFlip(p=0.5 if aug_config['horizontal_flip'] else 0.0),
])

# Resize preprocessing
preprocessing = A.Compose([
    A.Resize(
        IMAGE_SIZE, IMAGE_SIZE,
        interpolation=cv2.INTER_CUBIC,
        always_apply=True
    ),
])

def make_image_square_with_zero_padding(image):
    """Add zero padding to make image square"""
    width, height = image.size
    max_side = max(width, height)

    if image.mode == 'RGBA':
        image = image.convert('RGB')

    if image.mode == "RGB":
        padding_color = (0, 0, 0)
    elif image.mode == "L":
        padding_color = 0

    new_image = Image.new(image.mode, (max_side, max_side), padding_color)
    padding_left = (max_side - width) // 2
    padding_top = (max_side - height) // 2
    new_image.paste(image, (padding_left, padding_top))
    return new_image

def process_and_save_image(image_path, class_name, patient_id, split, augment=False):
    """Process single image and save"""
    if not os.path.exists(image_path):
        return 0
    
    try:
        image = Image.open(image_path)
        image = make_image_square_with_zero_padding(image)
        image = np.array(image)
        
        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        count = 0
        if augment and split == 'train':
            # Apply augmentation multiple times for training
            for i in range(N_AUGMENTATIONS):
                augmented = AUGMENTATION(image=image)
                aug_image = augmented['image']
                processed = preprocessing(image=aug_image)
                final_image = processed['image']
                
                save_name = f"{patient_id}_{class_name}_{i}.png"
                save_path = os.path.join(SAVE_PREPROCESSED_DIR, split, save_name)
                cv2.imwrite(save_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
                count += 1
        else:
            # No augmentation for val/test
            processed = preprocessing(image=image)
            final_image = processed['image']
            
            save_name = f"{patient_id}_{class_name}.png"
            save_path = os.path.join(SAVE_PREPROCESSED_DIR, split, save_name)
            cv2.imwrite(save_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
            count += 1
        
        return count
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return 0

def map_to_binary_class(brain_plane):
    """Map original plane names to binary classes"""
    if brain_plane == 'Trans-thalamic':
        return 'Trans-thalamic'
    else:
        return 'Other'  # Trans-cerebellum, Trans-ventricular, etc.

# ============================================================================
# Process FETAL_PLANES_DB (Binary Classification)
# ============================================================================
print("\nProcessing FETAL_PLANES_DB dataset for binary classification...")

df_planes = pd.read_csv(FETAL_PLANES_CSV, sep=';')

# FILTER OUT "Not A Brain" - We only want brain images
print(f"\nTotal images in CSV: {len(df_planes)}")
df_planes = df_planes[df_planes['Brain_plane'] != 'Not A Brain']
print(f"After removing 'Not A Brain': {len(df_planes)}")

# Map to binary classes
df_planes['Binary_Class'] = df_planes['Brain_plane'].apply(map_to_binary_class)

# Show original class distribution
print("\nOriginal class distribution (Brain images only):")
print(df_planes['Brain_plane'].value_counts())
print("\nBinary class distribution:")
print(df_planes['Binary_Class'].value_counts())

# Split by train/test flag in CSV
df_planes_train = df_planes[df_planes['Train '] == 1]
df_planes_test = df_planes[df_planes['Train '] == 0]

# Further split train into train/val
patients_train = df_planes_train['Patient_num'].unique()
patients_train_split, patients_val_split = train_test_split(
    patients_train, 
    train_size=TRAIN_VAL_SPLIT, 
    random_state=42
)

print(f"\nSplit statistics:")
print(f"  Train patients: {len(patients_train_split)}")
print(f"  Val patients: {len(patients_val_split)}")
print(f"  Test patients: {len(df_planes_test['Patient_num'].unique())}")

# Process train split
print("\nProcessing TRAIN split...")
train_count_thalamic = 0
train_count_other = 0
for _, row in tqdm(df_planes_train.iterrows(), total=len(df_planes_train), desc="  Train"):
    if row['Patient_num'] not in patients_train_split:
        continue
    
    image_name = row['Image_name'] if row['Image_name'].endswith('.png') else row['Image_name'] + '.png'
    image_path = os.path.join(FETAL_PLANES_DIR, image_name)
    
    binary_class = row['Binary_Class']
    count = process_and_save_image(
        image_path, 
        binary_class, 
        row['Patient_num'], 
        'train', 
        augment=True
    )
    
    if binary_class == 'Trans-thalamic':
        train_count_thalamic += count
    else:
        train_count_other += count

# Process val split
print("Processing VAL split...")
val_count_thalamic = 0
val_count_other = 0
for _, row in tqdm(df_planes_train.iterrows(), total=len(df_planes_train), desc="  Val"):
    if row['Patient_num'] not in patients_val_split:
        continue
    
    image_name = row['Image_name'] if row['Image_name'].endswith('.png') else row['Image_name'] + '.png'
    image_path = os.path.join(FETAL_PLANES_DIR, image_name)
    
    binary_class = row['Binary_Class']
    count = process_and_save_image(
        image_path, 
        binary_class, 
        row['Patient_num'], 
        'val', 
        augment=False
    )
    
    if binary_class == 'Trans-thalamic':
        val_count_thalamic += count
    else:
        val_count_other += count

# Process test split
print("Processing TEST split...")
test_count_thalamic = 0
test_count_other = 0
for _, row in tqdm(df_planes_test.iterrows(), total=len(df_planes_test), desc="  Test"):
    image_name = row['Image_name'] if row['Image_name'].endswith('.png') else row['Image_name'] + '.png'
    image_path = os.path.join(FETAL_PLANES_DIR, image_name)
    
    binary_class = row['Binary_Class']
    count = process_and_save_image(
        image_path, 
        binary_class, 
        row['Patient_num'], 
        'test', 
        augment=False
    )
    
    if binary_class == 'Trans-thalamic':
        test_count_thalamic += count
    else:
        test_count_other += count

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE!")
print("=" * 70)

total_train = len(os.listdir(os.path.join(SAVE_PREPROCESSED_DIR, 'train')))
total_val = len(os.listdir(os.path.join(SAVE_PREPROCESSED_DIR, 'val')))
total_test = len(os.listdir(os.path.join(SAVE_PREPROCESSED_DIR, 'test')))

print(f"\nTotal preprocessed images:")
print(f"  Train: {total_train}")
print(f"    - Trans-thalamic: {train_count_thalamic}")
print(f"    - Other: {train_count_other}")
print(f"  Val:   {total_val}")
print(f"    - Trans-thalamic: {val_count_thalamic}")
print(f"    - Other: {val_count_other}")
print(f"  Test:  {total_test}")
print(f"    - Trans-thalamic: {test_count_thalamic}")
print(f"    - Other: {test_count_other}")
print(f"  TOTAL: {total_train + total_val + total_test}")

print(f"\nPreprocessed data saved to: {SAVE_PREPROCESSED_DIR}")
print("=" * 70)
