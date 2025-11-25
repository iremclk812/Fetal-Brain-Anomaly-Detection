import os
import json
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from tqdm import tqdm
from PIL import Image

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'config.json')

with open(CONFIG_PATH, 'r') as file:
    config = json.load(file)

DIR_DATA = config['paths']['dir_data']
PATH_DATA_CSV = config['paths']['path_data_csv']
PATH_TRAIN_VAL_SPLIT = os.path.join(SCRIPT_DIR, config['paths']['path_train_val_split'])
PATH_TEST_SPLIT = os.path.join(SCRIPT_DIR, config['paths']['path_test_split'])
SAVE_PREPROCESSED_DIR = os.path.join(SCRIPT_DIR, config['paths']['dir_preprocessed'])

SAVE_IMAGE_SIZE = 224
N_AUGMENTATIONS = 10
DICT_CLSNAME_TO_CLSINDEX = {
    'Trans-thalamic': 0,
    'Trans-cerebellum': 1,
    'Trans-ventricular': 2,
}

print("Loading data...")
df_data = pd.read_csv(PATH_DATA_CSV, sep=';')

with open(PATH_TRAIN_VAL_SPLIT, 'r') as file:
    dict_list_pid = json.load(file)

with open(PATH_TEST_SPLIT, 'r') as file:
    list_test_pid = json.load(file)

AUGMENTATION = A.Compose([
    A.ColorJitter(0.2, 0.2, 0.2, 0.2, p=0.5),
    A.CLAHE(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.2,
        scale_limit=0.0,
        rotate_limit=20,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT, value=0, p=1.
    ),
])

preprocessing = A.Compose([
    A.Resize(
        SAVE_IMAGE_SIZE, SAVE_IMAGE_SIZE, interpolation=cv2.INTER_CUBIC,
        mask_interpolation=0, always_apply=True
    ),
])

print("Creating directories...")
os.makedirs(os.path.join(SAVE_PREPROCESSED_DIR, 'train'), exist_ok=True)
os.makedirs(os.path.join(SAVE_PREPROCESSED_DIR, 'val'), exist_ok=True)
os.makedirs(os.path.join(SAVE_PREPROCESSED_DIR, 'test'), exist_ok=True)

def make_image_square_with_zero_padding(image):
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

# Get lists
list_pid_train = []
list_pid_val = []
for _, val in dict_list_pid.items():
    for v in val:
        list_pid_train.extend(v[0])
        list_pid_val.extend(v[1])
list_pid_train = list(set(list_pid_train))
list_pid_val = list(set(list_pid_val))

# Convert string PIDs to integers (CSV has integer Patient_num)
list_pid_train = [int(p.replace('Patient', '')) if isinstance(p, str) else p for p in list_pid_train]
list_pid_val = [int(p.replace('Patient', '')) if isinstance(p, str) else p for p in list_pid_val]
list_test_pid = [int(p.replace('Patient', '')) if isinstance(p, str) else p for p in list_test_pid]

print(f"Train: {len(list_pid_train)}, Val: {len(list_pid_val)}, Test: {len(list_test_pid)}")

# Process train data
print("\nProcessing TRAIN data...")
count = 0
for idx, row in tqdm(df_data.iterrows(), total=len(df_data), desc="Train"):
    if row['Patient_num'] not in list_pid_train:
        continue
    if row['Brain_plane'] not in DICT_CLSNAME_TO_CLSINDEX:
        continue
    
    image_name = row['Image_name'] if row['Image_name'].endswith('.png') else row['Image_name'] + '.png'
    image_path = os.path.join(DIR_DATA, image_name)
    if not os.path.exists(image_path):
        continue
    
    image = Image.open(image_path)
    image = make_image_square_with_zero_padding(image)
    image = np.array(image)
    
    for i in range(N_AUGMENTATIONS):
        augmented = AUGMENTATION(image=image)
        aug_image = augmented['image']
        processed = preprocessing(image=aug_image)
        final_image = processed['image']
        
        save_name = f"{row['Patient_num']}_{row['Brain_plane']}_{i}.png"
        save_path = os.path.join(SAVE_PREPROCESSED_DIR, 'train', save_name)
        cv2.imwrite(save_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
        count += 1
print(f"Saved {count} train images")

# Process val data
print("\nProcessing VAL data...")
count = 0
for idx, row in tqdm(df_data.iterrows(), total=len(df_data), desc="Val"):
    if row['Patient_num'] not in list_pid_val:
        continue
    if row['Brain_plane'] not in DICT_CLSNAME_TO_CLSINDEX:
        continue
    
    image_name = row['Image_name'] if row['Image_name'].endswith('.png') else row['Image_name'] + '.png'
    image_path = os.path.join(DIR_DATA, image_name)
    if not os.path.exists(image_path):
        continue
    
    image = Image.open(image_path)
    image = make_image_square_with_zero_padding(image)
    image = np.array(image)
    processed = preprocessing(image=image)
    final_image = processed['image']
    
    save_name = f"{row['Patient_num']}_{row['Brain_plane']}.png"
    save_path = os.path.join(SAVE_PREPROCESSED_DIR, 'val', save_name)
    cv2.imwrite(save_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
    count += 1
print(f"Saved {count} val images")

# Process test data
print("\nProcessing TEST data...")
count = 0
for idx, row in tqdm(df_data.iterrows(), total=len(df_data), desc="Test"):
    if row['Patient_num'] not in list_test_pid:
        continue
    if row['Brain_plane'] not in DICT_CLSNAME_TO_CLSINDEX:
        continue
    
    image_name = row['Image_name'] if row['Image_name'].endswith('.png') else row['Image_name'] + '.png'
    image_path = os.path.join(DIR_DATA, image_name)
    if not os.path.exists(image_path):
        continue
    
    image = Image.open(image_path)
    image = make_image_square_with_zero_padding(image)
    image = np.array(image)
    processed = preprocessing(image=image)
    final_image = processed['image']
    
    save_name = f"{row['Patient_num']}_{row['Brain_plane']}.png"
    save_path = os.path.join(SAVE_PREPROCESSED_DIR, 'test', save_name)
    cv2.imwrite(save_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
    count += 1
print(f"Saved {count} test images")

print("\n✓ Preprocessing complete!")
print(f"Train images: {len(os.listdir(os.path.join(SAVE_PREPROCESSED_DIR, 'train')))}")
print(f"Val images: {len(os.listdir(os.path.join(SAVE_PREPROCESSED_DIR, 'val')))}")
print(f"Test images: {len(os.listdir(os.path.join(SAVE_PREPROCESSED_DIR, 'test')))}")
