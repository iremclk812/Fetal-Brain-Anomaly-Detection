"""
Train U-Net with K-Fold Cross Validation for robust evaluation
Uses 5-fold CV to get average performance across different train/val splits
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from tqdm import tqdm
import open_clip
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold

from unet_model import FetalCLIPUNet, DiceBCELoss, dice_coefficient, iou_score

# Load config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'config.json')

with open(CONFIG_PATH, 'r') as file:
    config = json.load(file)

# Paths
PREPROCESSED_DIR = os.path.join(SCRIPT_DIR, config['paths']['dir_preprocessed'])
EXPERIMENT_LOGS_DIR = os.path.join(SCRIPT_DIR, config['paths']['dir_experiment_logs'])
PATH_FETALCLIP_CONFIG = os.path.join(SCRIPT_DIR, config['paths']['path_fetalclip_config'])
PATH_FETALCLIP_WEIGHT = os.path.join(SCRIPT_DIR, config['paths']['path_fetalclip_weight'])

# Training params
BATCH_SIZE = config['training']['batch_size']
EPOCHS = config['training']['epochs']
LEARNING_RATE = config['training']['learning_rate']
DICE_WEIGHT = config['training']['dice_weight']
BCE_WEIGHT = config['training']['bce_weight']
K_FOLDS = 5  # Number of folds for cross-validation

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create experiment directory
os.makedirs(EXPERIMENT_LOGS_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_dir = os.path.join(EXPERIMENT_LOGS_DIR, f"kfold_{timestamp}")
os.makedirs(experiment_dir, exist_ok=True)

print("=" * 70)
print("U-NET K-FOLD CROSS VALIDATION TRAINING")
print("=" * 70)
print(f"Model: FetalCLIP (frozen) + U-Net Decoder")
print(f"K-Folds: {K_FOLDS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs per fold: {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Loss: {DICE_WEIGHT} Dice + {BCE_WEIGHT} BCE")
print(f"Experiment dir: {experiment_dir}")
print("=" * 70)

# Dataset
class HC18SegmentationDataset(Dataset):
    def __init__(self, preprocessed_dir, split='train', transform=None):
        self.images_dir = os.path.join(preprocessed_dir, split, 'images')
        self.masks_dir = os.path.join(preprocessed_dir, split, 'masks')
        self.transform = transform
        
        self.image_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.png')])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # Load image and mask
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        # Convert mask to tensor and normalize
        mask = np.array(mask).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]
        
        return image, mask

# Load FetalCLIP
print("\nLoading FetalCLIP model...")
with open(PATH_FETALCLIP_CONFIG, "r") as file:
    config_fetalclip = json.load(file)
open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = config_fetalclip

model_fetalclip, _, preprocess = open_clip.create_model_and_transforms(
    "FetalCLIP",
    pretrained=PATH_FETALCLIP_WEIGHT
)
print("✅ FetalCLIP loaded successfully")

# Load full training dataset (we'll split it for K-fold)
print("\nPreparing dataset for K-Fold CV...")
full_dataset = HC18SegmentationDataset(PREPROCESSED_DIR, split='train', transform=preprocess)
print(f"Total training samples: {len(full_dataset)}")

# Load test set (separate from K-fold)
test_dataset = HC18SegmentationDataset(PREPROCESSED_DIR, split='test', transform=preprocess)
print(f"Test samples: {len(test_dataset)}")

# K-Fold Cross Validation
kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

# Store results for each fold
fold_results = []

for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
    print("\n" + "=" * 70)
    print(f"FOLD {fold + 1}/{K_FOLDS}")
    print("=" * 70)
    print(f"Train samples: {len(train_ids)}")
    print(f"Val samples: {len(val_ids)}")
    
    # Create data subsets for this fold
    train_subsampler = Subset(full_dataset, train_ids)
    val_subsampler = Subset(full_dataset, val_ids)
    
    # Create data loaders
    train_loader = DataLoader(train_subsampler, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subsampler, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Initialize model for this fold
    model = FetalCLIPUNet(model_fetalclip, freeze_encoder=True).to(device)
    
    # Loss and optimizer
    criterion = DiceBCELoss(dice_weight=DICE_WEIGHT, bce_weight=BCE_WEIGHT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training history for this fold
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_dice': [],
        'val_iou': []
    }
    
    best_val_dice = 0.0
    patience_counter = 0
    early_stopping_patience = 20
    
    # Training loop for this fold
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        
        with tqdm(train_loader, desc=f'Fold {fold+1} Epoch {epoch+1}/{EPOCHS} [Train]') as pbar:
            for images, masks in pbar:
                images, masks = images.to(device), masks.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                val_dice += dice_coefficient(outputs, masks)
                val_iou += iou_score(outputs, masks)
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        
        history['val_loss'].append(avg_val_loss)
        history['val_dice'].append(avg_val_dice)
        history['val_iou'].append(avg_val_iou)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS} - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Dice: {avg_val_dice:.4f}, "
              f"Val IoU: {avg_val_iou:.4f}")
        
        # Save best model for this fold
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            patience_counter = 0
            torch.save(model.state_dict(), 
                      os.path.join(experiment_dir, f'best_model_fold{fold+1}.pth'))
            print(f"✅ Saved best model (Dice: {best_val_dice:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"⚠️ Early stopping triggered at epoch {epoch+1}")
            break
    
    # Store fold results
    fold_results.append({
        'fold': fold + 1,
        'best_val_dice': best_val_dice,
        'final_val_dice': history['val_dice'][-1],
        'final_val_iou': history['val_iou'][-1],
        'history': history
    })
    
    # Save fold history
    pd.DataFrame(history).to_csv(
        os.path.join(experiment_dir, f'fold{fold+1}_history.csv'), index=False
    )

# Calculate average metrics across all folds
print("\n" + "=" * 70)
print("K-FOLD CROSS VALIDATION RESULTS")
print("=" * 70)

best_dice_scores = [r['best_val_dice'] for r in fold_results]
final_dice_scores = [r['final_val_dice'] for r in fold_results]
final_iou_scores = [r['final_val_iou'] for r in fold_results]

print(f"\nBest Validation Dice (per fold):")
for i, score in enumerate(best_dice_scores):
    print(f"  Fold {i+1}: {score:.4f}")
print(f"\nAverage Best Dice: {np.mean(best_dice_scores):.4f} ± {np.std(best_dice_scores):.4f}")
print(f"Average Final Dice: {np.mean(final_dice_scores):.4f} ± {np.std(final_dice_scores):.4f}")
print(f"Average Final IoU: {np.mean(final_iou_scores):.4f} ± {np.std(final_iou_scores):.4f}")

# Save summary
summary = {
    'avg_best_dice': np.mean(best_dice_scores),
    'std_best_dice': np.std(best_dice_scores),
    'avg_final_dice': np.mean(final_dice_scores),
    'std_final_dice': np.std(final_dice_scores),
    'avg_final_iou': np.mean(final_iou_scores),
    'std_final_iou': np.std(final_iou_scores),
    'fold_results': fold_results
}

with open(os.path.join(experiment_dir, 'kfold_summary.json'), 'w') as f:
    json.dump(summary, f, indent=4, default=str)

print(f"\n✅ K-Fold Cross Validation complete!")
print(f"📊 Results saved to: {experiment_dir}")
print("=" * 70)
