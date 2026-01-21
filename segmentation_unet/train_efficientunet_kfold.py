"""
Train EfficientUNet with K-Fold Cross Validation
Fast training with EfficientNet-B0 encoder
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
import cv2

from attention_unet_model import AttentionUNet, DiceTverskyLoss, dice_coefficient, iou_score, pixel_accuracy

# Load config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'config.json')

with open(CONFIG_PATH, 'r') as file:
    config = json.load(file)

# Paths
PREPROCESSED_DIR = os.path.join(SCRIPT_DIR, config['paths']['dir_preprocessed'])
EXPERIMENT_LOGS_DIR = os.path.join(SCRIPT_DIR, config['paths']['dir_experiment_logs'])

# Training params
BATCH_SIZE = 16  # Increased for better gradient stability
EPOCHS = 50  # Tversky needs more epochs to converge
LEARNING_RATE = config['training']['learning_rate']
DICE_WEIGHT = 0.7  # Dice weight
FOCAL_WEIGHT = 0.3  # Tversky weight
K_FOLDS = 3 # Keep 3 for KFold to work
TEST_SINGLE_FOLD = True  # Set False to run all 3 folds

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Mixed precision training
use_amp = device.type == 'cuda'
scaler = GradScaler() if use_amp else None

# Create experiment directory
os.makedirs(EXPERIMENT_LOGS_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_dir = os.path.join(EXPERIMENT_LOGS_DIR, f"efficientunet_kfold_{timestamp}")
os.makedirs(experiment_dir, exist_ok=True)

print("=" * 70)
print("ATTENTION U-NET K-FOLD CROSS VALIDATION")
print("=" * 70)
print(f"Model: EfficientNet-B0 + Attention U-Net")
print(f"Loss: DiceTverskyLoss (optimized for tiny targets)")
print(f"K-Folds: {K_FOLDS}")
print(f"Batch size: {BATCH_SIZE} (increased for stability)")
print(f"Epochs per fold: {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Loss weights: Dice={DICE_WEIGHT}, Tversky={FOCAL_WEIGHT}")
print(f"Mixed Precision: {use_amp}")
print(f"Experiment dir: {experiment_dir}")
print("=" * 70)

# Dataset
class HC18SegmentationDataset(Dataset):
    def __init__(self, preprocessed_dir, split='train'):
        self.images_dir = os.path.join(preprocessed_dir, split, 'images')
        self.masks_dir = os.path.join(preprocessed_dir, split, 'masks')
        
        self.image_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.png')])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # Load image and mask
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize to 256x256
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)
        
        # To tensor
        image = torch.from_numpy(image).permute(2, 0, 1)  # [3, H, W]
        mask = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]
        
        return image, mask

# Load full training dataset
print("\nPreparing dataset for K-Fold CV...")
full_dataset = HC18SegmentationDataset(PREPROCESSED_DIR, split='train')
print(f"Total training samples: {len(full_dataset)}")

# Load test set
test_dataset = HC18SegmentationDataset(PREPROCESSED_DIR, split='test')
print(f"Test samples: {len(test_dataset)}")

if __name__ == '__main__':
    # K-Fold Cross Validation
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
        print("\n" + "=" * 70)
        print(f"FOLD {fold + 1}/{K_FOLDS}")
        print("=" * 70)
        print(f"Train samples: {len(train_ids)}")
        print(f"Val samples: {len(val_ids)}")
        
        # Create data subsets
        train_subsampler = Subset(full_dataset, train_ids)
        val_subsampler = Subset(full_dataset, val_ids)
        
        # Data loaders
        train_loader = DataLoader(train_subsampler, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
        val_loader = DataLoader(val_subsampler, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
        
        # Initialize Attention U-Net
        model = AttentionUNet().to(device)
        
        # Loss and optimizer (Tversky for tiny targets)
        criterion = DiceTverskyLoss(
            dice_weight=DICE_WEIGHT,
            tversky_weight=FOCAL_WEIGHT,
            tversky_alpha=0.3,  # Low FP penalty
            tversky_beta=0.7    # High FN penalty - prioritize recall over precision
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_dice': [],
            'val_iou': [],
            'val_accuracy': []
        }
        
        best_val_dice = 0.0
        patience_counter = 0
        early_stopping_patience = 15
        
        # Training loop
        first_batch_checked = False
        for epoch in range(EPOCHS):
            # Training phase
            model.train()
            train_loss = 0.0
            
            with tqdm(train_loader, desc=f'Fold {fold+1} Epoch {epoch+1}/{EPOCHS} [Train]') as pbar:
                for batch_idx, (images, masks) in enumerate(pbar):
                    # Debug first batch of first epoch
                    if epoch == 0 and batch_idx == 0 and not first_batch_checked:
                        print(f"\n[DEBUG] First batch check:")
                        print(f"  Image shape: {images.shape}, range: [{images.min():.3f}, {images.max():.3f}]")
                        print(f"  Mask shape: {masks.shape}, unique values: {torch.unique(masks).cpu().numpy()}")
                        print(f"  Mask mean: {masks.mean():.4f} (should be ~0.05-0.15 for HC)")
                        first_batch_checked = True
                    
                    images, masks = images.to(device), masks.to(device)
                    
                    optimizer.zero_grad()
                    
                    # Mixed precision forward pass
                    if use_amp:
                        with autocast():
                            outputs = model(images)
                            loss = criterion(outputs, masks)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        outputs = model(images)
                        loss = criterion(outputs, masks)
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
            val_accuracy = 0.0
            
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    
                    if use_amp:
                        with autocast():
                            outputs = model(images)
                            loss = criterion(outputs, masks)
                    else:
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                    
                    val_loss += loss.item()
                    val_dice += dice_coefficient(outputs, masks)
                    val_iou += iou_score(outputs, masks)
                    val_accuracy += pixel_accuracy(outputs, masks)
            
            avg_val_loss = val_loss / len(val_loader)
            avg_val_dice = val_dice / len(val_loader)
            avg_val_iou = val_iou / len(val_loader)
            avg_val_accuracy = val_accuracy / len(val_loader)
            
            history['val_loss'].append(avg_val_loss)
            history['val_dice'].append(avg_val_dice)
            history['val_iou'].append(avg_val_iou)
            history['val_accuracy'].append(avg_val_accuracy)
            
            scheduler.step(avg_val_loss)
            
            print(f"Epoch {epoch+1}/{EPOCHS} - "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, "
                  f"Val Dice: {avg_val_dice:.4f}, "
                  f"Val IoU: {avg_val_iou:.4f}, "
                  f"Val Acc: {avg_val_accuracy:.4f}")
            
            # Save best model
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
        
        # Break after first fold if testing
        if TEST_SINGLE_FOLD:
            print(f"\n⚠️ Testing mode: Only ran 1 fold. Set TEST_SINGLE_FOLD=False to run all {K_FOLDS} folds.")
            break

    # Calculate average metrics
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
        'avg_best_dice': float(np.mean(best_dice_scores)),
        'std_best_dice': float(np.std(best_dice_scores)),
        'avg_final_dice': float(np.mean(final_dice_scores)),
        'std_final_dice': float(np.std(final_dice_scores)),
        'avg_final_iou': float(np.mean(final_iou_scores)),
        'std_final_iou': float(np.std(final_iou_scores)),
    }

    with open(os.path.join(experiment_dir, 'kfold_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"\n✅ K-Fold Cross Validation complete!")
    print(f"📊 Results saved to: {experiment_dir}")
    print("=" * 70)
