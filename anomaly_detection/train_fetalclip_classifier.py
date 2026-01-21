"""
Binary Plane Classification using FetalCLIP Embeddings + Custom Classifier
Gatekeeper model to validate plane before HC measurement

Architecture:
- FetalCLIP (frozen) → Extract 768-dim embeddings
- Global Average Pooling
- Dense(512, ReLU) + Dropout(0.5)
- Dense(2, Softmax)

Classes:
0: Trans-thalamic (Valid for HC measurement)
1: Other (Trans-cerebellum, Trans-ventricular, etc.)
"""
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from tqdm import tqdm
import open_clip
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

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
DENSE_UNITS = config['model']['dense_units']
DROPOUT_RATE = config['model']['dropout_rate']

# Class mapping
CLASS_NAMES = config['classes']
NUM_CLASSES = len(CLASS_NAMES)
DICT_CLSNAME_TO_CLSINDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create experiment logs dir
os.makedirs(EXPERIMENT_LOGS_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_dir = os.path.join(EXPERIMENT_LOGS_DIR, f"exp_{timestamp}")
os.makedirs(experiment_dir, exist_ok=True)

print("=" * 70)
print("BINARY PLANE CLASSIFICATION TRAINING (GATEKEEPER MODEL)")
print("=" * 70)
print(f"Model: FetalCLIP (frozen) + Custom Classifier")
print(f"Classes: {CLASS_NAMES}")
print(f"  - Class 0 (Trans-thalamic): Valid for HC measurement")
print(f"  - Class 1 (Other): Invalid planes")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Experiment dir: {experiment_dir}")
print("=" * 70)

# ============================================================================
# Dataset
# ============================================================================
class FetalBrainDataset(Dataset):
    def __init__(self, preprocessed_dir, split='train', transform=None):
        self.preprocessed_dir = os.path.join(preprocessed_dir, split)
        self.transform = transform
        self.data = []
        
        for filename in os.listdir(self.preprocessed_dir):
            if not filename.endswith('.png'):
                continue
            
            # Parse filename: {patient_id}_{class_name}_{aug_idx}.png or {patient_id}_{class_name}.png
            parts = filename.replace('.png', '').split('_')
            
            # Find class name in filename
            class_name = None
            for cn in CLASS_NAMES:
                if cn in filename:
                    class_name = cn
                    break
            
            if class_name is None:
                # Try with hyphens replaced
                for cn in CLASS_NAMES:
                    if cn.replace('-', ' ') in filename or cn.replace('-', '') in filename:
                        class_name = cn
                        break
            
            if class_name is not None:
                self.data.append({
                    'path': os.path.join(self.preprocessed_dir, filename),
                    'label': DICT_CLSNAME_TO_CLSINDEX[class_name],
                    'class_name': class_name
                })
        
        print(f"  {split.upper()} set: {len(self.data)} images")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, item['label']

# ============================================================================
# Load FetalCLIP
# ============================================================================
print("\nLoading FetalCLIP model...")
with open(PATH_FETALCLIP_CONFIG, "r") as file:
    config_fetalclip = json.load(file)
open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = config_fetalclip

model_fetalclip, _, preprocess = open_clip.create_model_and_transforms(
    "FetalCLIP", 
    pretrained=PATH_FETALCLIP_WEIGHT
)
model_fetalclip = model_fetalclip.to(device)
model_fetalclip.eval()

# Freeze FetalCLIP
for param in model_fetalclip.parameters():
    param.requires_grad = False

print("✓ FetalCLIP loaded and frozen")

# ============================================================================
# Custom Classifier
# ============================================================================
class FetalBrainClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, num_classes=4, dropout=0.5):
        super(FetalBrainClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

classifier = FetalBrainClassifier(
    input_dim=768,  # FetalCLIP embedding dimension
    hidden_dim=DENSE_UNITS,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT_RATE
).to(device)

print(f"✓ Classifier initialized: 768 → {DENSE_UNITS} → {NUM_CLASSES}")

# ============================================================================
# Data Loaders
# ============================================================================
print("\nPreparing datasets...")
train_dataset = FetalBrainDataset(PREPROCESSED_DIR, 'train', transform=preprocess)
val_dataset = FetalBrainDataset(PREPROCESSED_DIR, 'val', transform=preprocess)
test_dataset = FetalBrainDataset(PREPROCESSED_DIR, 'test', transform=preprocess)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ============================================================================
# Training Setup
# ============================================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, verbose=True
)

# Training history
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'val_f1': []
}

best_val_acc = 0.0
best_val_f1 = 0.0
early_stopping_counter = 0
early_stopping_patience = 15

# ============================================================================
# Training Loop
# ============================================================================
print("\nStarting training...")
print("=" * 70)

for epoch in range(EPOCHS):
    # ========== TRAIN ==========
    classifier.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [TRAIN]")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Extract FetalCLIP embeddings (frozen)
        with torch.no_grad():
            embeddings = model_fetalclip.encode_image(images)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        # Forward pass through classifier
        optimizer.zero_grad()
        outputs = classifier(embeddings)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{100.*train_correct/train_total:.2f}%"
        })
    
    train_loss /= len(train_loader)
    train_acc = 100. * train_correct / train_total
    
    # ========== VALIDATION ==========
    classifier.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [VAL]")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Extract embeddings
            embeddings = model_fetalclip.encode_image(images)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            
            # Predict
            outputs = classifier(embeddings)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100.*val_correct/val_total:.2f}%"
            })
    
    val_loss /= len(val_loader)
    val_acc = 100. * val_correct / val_total
    val_f1 = f1_score(all_labels, all_preds, average='weighted') * 100
    
    # Update history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_f1'].append(val_f1)
    
    # Learning rate scheduler
    scheduler.step(val_loss)
    
    # Print epoch summary
    print(f"\nEpoch {epoch+1}/{EPOCHS} Summary:")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}% | Val F1: {val_f1:.2f}%")
    
    # Save best model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'classifier_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_f1': val_f1,
        }, os.path.join(experiment_dir, 'best_model.pth'))
        print(f"  ✓ Best model saved (F1: {val_f1:.2f}%)")
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
    
    # Early stopping
    if early_stopping_counter >= early_stopping_patience:
        print(f"\nEarly stopping triggered (patience={early_stopping_patience})")
        break
    
    print("-" * 70)

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print(f"Best Val Accuracy: {best_val_acc:.2f}%")
print(f"Best Val F1-Score: {best_val_f1:.2f}%")
print("=" * 70)

# ============================================================================
# Test Evaluation
# ============================================================================
print("\nEvaluating on TEST set...")

# Load best model
checkpoint = torch.load(os.path.join(experiment_dir, 'best_model.pth'))
classifier.load_state_dict(checkpoint['classifier_state_dict'])
classifier.eval()

test_preds = []
test_labels = []
test_probs = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        labels = labels.to(device)
        
        # Extract embeddings
        embeddings = model_fetalclip.encode_image(images)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        # Predict
        outputs = classifier(embeddings)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        
        test_preds.extend(predicted.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())
        test_probs.extend(probs.cpu().numpy())

# Metrics
test_acc = accuracy_score(test_labels, test_preds) * 100
test_f1 = f1_score(test_labels, test_preds, average='weighted') * 100

print(f"\nTEST RESULTS:")
print(f"  Accuracy: {test_acc:.2f}%")
print(f"  F1-Score: {test_f1:.2f}%")

# Classification report
print("\nClassification Report:")
print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES, digits=4))

# Confusion matrix
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix - Test Set')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(experiment_dir, 'confusion_matrix_test.png'), dpi=300)
print(f"✓ Confusion matrix saved")

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Loss curve
axes[0].plot(history['train_loss'], label='Train Loss')
axes[0].plot(history['val_loss'], label='Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True)

# Accuracy curve
axes[1].plot(history['train_acc'], label='Train Accuracy')
axes[1].plot(history['val_acc'], label='Val Accuracy')
axes[1].plot(history['val_f1'], label='Val F1-Score')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy / F1 (%)')
axes[1].set_title('Training and Validation Metrics')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(experiment_dir, 'training_curves.png'), dpi=300)
print(f"✓ Training curves saved")

# Save results to CSV
results_df = pd.DataFrame({
    'metric': ['test_accuracy', 'test_f1', 'best_val_accuracy', 'best_val_f1'],
    'value': [test_acc, test_f1, best_val_acc, best_val_f1]
})
results_df.to_csv(os.path.join(experiment_dir, 'results.csv'), index=False)

# Save detailed predictions
predictions_df = pd.DataFrame({
    'true_label': [CLASS_NAMES[l] for l in test_labels],
    'predicted_label': [CLASS_NAMES[p] for p in test_preds],
    'true_idx': test_labels,
    'predicted_idx': test_preds,
    **{f'prob_{CLASS_NAMES[i]}': [prob[i] for prob in test_probs] for i in range(NUM_CLASSES)}
})
predictions_df.to_csv(os.path.join(experiment_dir, 'predictions_test.csv'), index=False)

print(f"\n✓ All results saved to: {experiment_dir}")
print("=" * 70)
