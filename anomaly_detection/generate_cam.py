"""
Generate Class Activation Maps (CAM) for 4-class brain classification
Shows which regions the model focuses on for each prediction
"""
import os
import json
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import open_clip
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch.nn as nn

# Load config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'config.json')

with open(CONFIG_PATH, 'r') as file:
    config = json.load(file)

# Paths
PREPROCESSED_DIR = os.path.join(SCRIPT_DIR, config['paths']['dir_preprocessed'])
PATH_FETALCLIP_CONFIG = os.path.join(SCRIPT_DIR, config['paths']['path_fetalclip_config'])
PATH_FETALCLIP_WEIGHT = os.path.join(SCRIPT_DIR, config['paths']['path_fetalclip_weight'])
EXPERIMENT_DIR = input("Enter experiment directory (e.g., experiment_logs/exp_20231201_120000): ")
MODEL_PATH = os.path.join(SCRIPT_DIR, EXPERIMENT_DIR, 'best_model.pth')

# CAM output
CAM_OUTPUT_DIR = os.path.join(SCRIPT_DIR, EXPERIMENT_DIR, 'cam_visualizations')
os.makedirs(CAM_OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(CAM_OUTPUT_DIR, 'images'), exist_ok=True)

# Parameters
N_SAMPLES_PER_CLASS = 10
CLASS_NAMES = config['classes']
DICT_CLSNAME_TO_CLSINDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# Load Models
# ============================================================================
print("Loading FetalCLIP...")
with open(PATH_FETALCLIP_CONFIG, "r") as file:
    config_fetalclip = json.load(file)
open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = config_fetalclip

model_fetalclip, _, preprocess = open_clip.create_model_and_transforms(
    "FetalCLIP", 
    pretrained=PATH_FETALCLIP_WEIGHT
)
model_fetalclip = model_fetalclip.to(device)
model_fetalclip.eval()

# Load classifier
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
    input_dim=768,
    hidden_dim=config['model']['dense_units'],
    num_classes=len(CLASS_NAMES),
    dropout=config['model']['dropout_rate']
).to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device)
classifier.load_state_dict(checkpoint['classifier_state_dict'])
classifier.eval()
print("✓ Models loaded")

# ============================================================================
# Combined Model Wrapper for CAM
# ============================================================================
class CombinedModelWrapper(nn.Module):
    def __init__(self, fetalclip, classifier):
        super().__init__()
        self.fetalclip = fetalclip
        self.classifier = classifier
    
    def forward(self, images):
        # Extract embeddings
        embeddings = self.fetalclip.encode_image(images)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        # Classify
        outputs = self.classifier(embeddings)
        return outputs

combined_model = CombinedModelWrapper(model_fetalclip, classifier).to(device)
combined_model.eval()

# Target layer: Last transformer layer of FetalCLIP
target_layers = [combined_model.fetalclip.visual.transformer.resblocks[-1].ln_1]

def reshape_transform(tensor, height=16, width=16):
    """Reshape transformer output for spatial visualization"""
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

cam = GradCAM(
    model=combined_model,
    target_layers=target_layers,
    reshape_transform=reshape_transform
)

# ============================================================================
# Generate CAMs
# ============================================================================
test_dir = os.path.join(PREPROCESSED_DIR, 'test')
test_images = os.listdir(test_dir)

# Group by class
class_images = {cls: [] for cls in CLASS_NAMES}
for img_name in test_images:
    for cls in CLASS_NAMES:
        if cls in img_name:
            class_images[cls].append(img_name)
            break

print("\n" + "=" * 70)
print("GENERATING CLASS ACTIVATION MAPS")
print("=" * 70)

cam_summary = []

for target_class in CLASS_NAMES:
    print(f"\nProcessing class: {target_class}")
    target_idx = DICT_CLSNAME_TO_CLSINDEX[target_class]
    
    images_for_class = class_images[target_class][:N_SAMPLES_PER_CLASS]
    
    for img_name in tqdm(images_for_class, desc=f"  {target_class}"):
        img_path = os.path.join(test_dir, img_name)
        
        # Load and preprocess image
        img_pil = Image.open(img_path).convert('RGB')
        img_tensor = preprocess(img_pil).unsqueeze(0).to(device)
        
        # For visualization
        img_np = np.array(img_pil.resize((224, 224)))
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
        
        # Get prediction
        with torch.no_grad():
            output = combined_model(img_tensor)
            probs = torch.softmax(output, dim=1)
            pred_class = output.argmax(1).item()
            pred_prob = probs[0][pred_class].item()
        
        # Generate CAM
        targets = [ClassifierOutputTarget(target_idx)]
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Overlay CAM on image
        cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True, image_weight=0.5)
        
        # Save
        pred_class_name = CLASS_NAMES[pred_class]
        save_name = f"{target_class}_TRUE_{img_name.split('.')[0]}_PRED_{pred_class_name}_{int(pred_prob*100)}.png"
        save_path = os.path.join(CAM_OUTPUT_DIR, save_name)
        cv2.imwrite(save_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
        
        # Save original
        orig_save_path = os.path.join(CAM_OUTPUT_DIR, 'images', f"orig_{img_name}")
        cv2.imwrite(orig_save_path, cv2.cvtColor(img_np.astype(np.float32), cv2.COLOR_RGB2BGR) * 255)
        
        # Log
        cam_summary.append({
            'true_class': target_class,
            'predicted_class': pred_class_name,
            'confidence': pred_prob,
            'correct': target_class == pred_class_name,
            'image': img_name,
            'cam_file': save_name
        })

# Save summary
import pandas as pd
summary_df = pd.DataFrame(cam_summary)
summary_df.to_csv(os.path.join(CAM_OUTPUT_DIR, 'cam_summary.csv'), index=False)

# Print statistics
print("\n" + "=" * 70)
print("CAM GENERATION COMPLETE")
print("=" * 70)
print(f"Total CAMs generated: {len(cam_summary)}")
print(f"Accuracy on sampled images: {summary_df['correct'].mean()*100:.2f}%")
print(f"\nPer-class results:")
for cls in CLASS_NAMES:
    cls_df = summary_df[summary_df['true_class'] == cls]
    if len(cls_df) > 0:
        acc = cls_df['correct'].mean() * 100
        print(f"  {cls}: {acc:.2f}% ({cls_df['correct'].sum()}/{len(cls_df)})")
print(f"\nCAM visualizations saved to: {CAM_OUTPUT_DIR}")
print("=" * 70)
