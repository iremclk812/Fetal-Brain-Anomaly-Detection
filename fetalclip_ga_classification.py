"""
FINAL SOLUTION: Two-Stage GA-Based Classification for HC18

Stage 1: FetalCLIP Zero-Shot GA Estimation (Visual Features)
    - Input: Ultrasound image
    - Method: Text prompt similarity with GA-specific descriptions
    - Output: Estimated GA (weeks + days)
    - Source: test_ga_dot_prods_map.py approach

Stage 2: GA-Specific Percentile Classification
    - Input: True HC measurement + Estimated GA
    - Method: Compare HC with GA-specific percentiles (3rd-97th)
    - Output: Normal (0) or Abnormal (1)
    - Rationale: Avoids circular dependency of pure HC→GA conversion

Clinical Justification:
    - FetalCLIP trained on fetal ultrasound images learns visual patterns
      beyond just head size (brain structure, anatomical landmarks, etc.)
    - Visual GA estimation reduces bias from abnormal HC measurements
    - Example: Macrocephalic fetus at 20w may have HC of large GA,
      but visual features (brain structure, body proportion) reveal true GA
"""

import os
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import open_clip
import matplotlib.pyplot as plt

# ============================
# CONFIGURATION
# ============================
DIR_IMAGES = r'1327317\training_set'
PATH_CSV = r'1327317\training_set_pixel_size_and_HC.csv'
PATH_FETALCLIP_WEIGHT = "FetalCLIP_weights.pt"
PATH_FETALCLIP_CONFIG = "FetalCLIP_config.json"
OUTPUT_CSV = "HC18_fetalclip_ga_labels.csv"
OUTPUT_PLOT = "HC18_fetalclip_validation.png"

INPUT_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# GA-specific percentile reference (from Fetal Medicine Foundation)
GA_PERCENTILE_TABLE = {
    16: {'p3': 103, 'p50': 120, 'p97': 137},
    18: {'p3': 121, 'p50': 141, 'p97': 161},
    20: {'p3': 141, 'p50': 164, 'p97': 187},
    22: {'p3': 162, 'p50': 189, 'p97': 216},
    24: {'p3': 184, 'p50': 214, 'p97': 244},
    26: {'p3': 206, 'p50': 240, 'p97': 274},
    28: {'p3': 228, 'p50': 265, 'p97': 302},
    30: {'p3': 249, 'p50': 289, 'p97': 329},
    32: {'p3': 269, 'p50': 312, 'p97': 355},
    34: {'p3': 288, 'p50': 333, 'p97': 378},
    36: {'p3': 305, 'p50': 353, 'p97': 401},
    38: {'p3': 321, 'p50': 371, 'p97': 421},
    40: {'p3': 335, 'p50': 387, 'p97': 439},
}

# ============================
# HELPER FUNCTIONS
# ============================
def make_image_square_with_zero_padding(image):
    """Add zero padding to make image square"""
    width, height = image.size
    max_side = max(width, height)
    
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    padding_color = (0, 0, 0) if image.mode == "RGB" else 0
    new_image = Image.new(image.mode, (max_side, max_side), padding_color)
    
    padding_left = (max_side - width) // 2
    padding_top = (max_side - height) // 2
    new_image.paste(image, (padding_left, padding_top))
    
    return new_image

def interpolate_percentile(ga, percentile_key):
    """Interpolate percentile for given GA"""
    ga_keys = sorted(GA_PERCENTILE_TABLE.keys())
    
    if ga <= ga_keys[0]:
        return GA_PERCENTILE_TABLE[ga_keys[0]][percentile_key]
    if ga >= ga_keys[-1]:
        return GA_PERCENTILE_TABLE[ga_keys[-1]][percentile_key]
    
    for i in range(len(ga_keys) - 1):
        if ga_keys[i] <= ga < ga_keys[i+1]:
            ga1, ga2 = ga_keys[i], ga_keys[i+1]
            val1 = GA_PERCENTILE_TABLE[ga1][percentile_key]
            val2 = GA_PERCENTILE_TABLE[ga2][percentile_key]
            return val1 + (val2 - val1) * (ga - ga1) / (ga2 - ga1)
    
    return None

def classify_with_visual_ga(hc_mm, visual_ga_weeks):
    """
    Classify using visually estimated GA (FetalCLIP)
    This reduces circular dependency from HC-based GA estimation
    """
    if visual_ga_weeks is None or visual_ga_weeks < 16 or visual_ga_weeks > 40:
        return None
    
    p3 = interpolate_percentile(visual_ga_weeks, 'p3')
    p97 = interpolate_percentile(visual_ga_weeks, 'p97')
    
    if p3 is None or p97 is None:
        return None
    
    return 1 if (hc_mm < p3 or hc_mm > p97) else 0

# ============================
# LOAD FETALCLIP MODEL
# ============================
print("\nLoading FetalCLIP model...")
with open(PATH_FETALCLIP_CONFIG, "r") as file:
    config_fetalclip = json.load(file)
open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = config_fetalclip

model, _, preprocess = open_clip.create_model_and_transforms(
    'FetalCLIP', 
    pretrained=PATH_FETALCLIP_WEIGHT,
    device=DEVICE
)
tokenizer = open_clip.get_tokenizer('FetalCLIP')
model.eval()
print("✅ FetalCLIP loaded successfully")

# ============================
# PREPARE TEXT PROMPTS
# ============================
print("\nPreparing GA-specific text prompts...")
TEMPLATE = "Ultrasound image at {weeks} weeks and {days} days gestation focusing on the fetal brain"

list_ga_in_days = [weeks * 7 + days for weeks in range(16, 41) for days in range(0, 7)]
text_prompts = []

for ga_days in list_ga_in_days:
    weeks = ga_days // 7
    days = ga_days % 7
    prompt = TEMPLATE.format(weeks=weeks, days=days)
    text_prompts.append(prompt)

# Encode all text prompts
with torch.no_grad():
    text_tokens = tokenizer(text_prompts).to(DEVICE)
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

print(f"✅ Prepared {len(text_prompts)} GA-specific prompts")

# ============================
# PROCESS IMAGES
# ============================
print("\nProcessing HC18 images...")
df = pd.read_csv(PATH_CSV)

results = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Estimating GA"):
    filename = row['filename']
    true_hc = row['head circumference (mm)']
    
    # Load and preprocess image
    image_path = os.path.join(DIR_IMAGES, filename)
    if not os.path.exists(image_path):
        print(f"Warning: {image_path} not found")
        continue
    
    image = Image.open(image_path)
    image = make_image_square_with_zero_padding(image)
    image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    
    # Get image features
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # Calculate similarity with all GA prompts
    similarity = (image_features @ text_features.T).squeeze(0)
    
    # Get top N most similar GAs and take median (reduces noise)
    top_n = 15
    top_indices = torch.topk(similarity, top_n).indices.cpu().numpy()
    top_indices_sorted = sorted(top_indices)
    median_idx = top_indices_sorted[top_n // 2]
    
    estimated_ga_days = list_ga_in_days[median_idx]
    estimated_ga_weeks = estimated_ga_days / 7
    
    # Classify based on visual GA estimation
    label = classify_with_visual_ga(true_hc, estimated_ga_weeks)
    
    results.append({
        'filename': filename,
        'true_hc_mm': true_hc,
        'visual_ga_weeks': round(estimated_ga_weeks, 1),
        'visual_ga_days': estimated_ga_days,
        'label': label,
        'max_similarity': similarity[median_idx].item()
    })

# ============================
# SAVE RESULTS
# ============================
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Results saved to: {OUTPUT_CSV}")

# ============================
# STATISTICS
# ============================
valid_labels = results_df['label'].dropna()
normal_count = (valid_labels == 0).sum()
abnormal_count = (valid_labels == 1).sum()

print("\n" + "=" * 50)
print("FETALCLIP GA-BASED LABELING RESULTS")
print("=" * 50)
print(f"Total samples processed: {len(results_df)}")
print(f"Valid labels: {len(valid_labels)}")
print(f"Normal: {normal_count} ({normal_count/len(valid_labels)*100:.1f}%)")
print(f"Abnormal: {abnormal_count} ({abnormal_count/len(valid_labels)*100:.1f}%)")
print(f"\nMean visual GA: {results_df['visual_ga_weeks'].mean():.1f} weeks")
print(f"Std visual GA: {results_df['visual_ga_weeks'].std():.1f} weeks")

# ============================
# VISUALIZATION
# ============================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Visual GA distribution
axes[0, 0].hist(results_df['visual_ga_weeks'], bins=30, edgecolor='black', alpha=0.7, color='blue')
axes[0, 0].set_xlabel('Visual GA Estimation (weeks)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('FetalCLIP Visual GA Distribution')
axes[0, 0].grid(alpha=0.3)

# HC vs Visual GA with percentiles
normal_data = results_df[results_df['label'] == 0]
abnormal_data = results_df[results_df['label'] == 1]

axes[0, 1].scatter(normal_data['visual_ga_weeks'], normal_data['true_hc_mm'], 
                   color='green', s=20, alpha=0.6, label='Normal')
axes[0, 1].scatter(abnormal_data['visual_ga_weeks'], abnormal_data['true_hc_mm'], 
                   color='red', s=20, alpha=0.6, label='Abnormal')

# Plot percentile curves
ga_range = np.linspace(16, 40, 100)
p3_curve = [interpolate_percentile(ga, 'p3') for ga in ga_range]
p50_curve = [interpolate_percentile(ga, 'p50') for ga in ga_range]
p97_curve = [interpolate_percentile(ga, 'p97') for ga in ga_range]
axes[0, 1].plot(ga_range, p3_curve, 'k--', label='3rd %ile', linewidth=2)
axes[0, 1].plot(ga_range, p50_curve, 'k-', label='50th %ile', linewidth=2)
axes[0, 1].plot(ga_range, p97_curve, 'k--', label='97th %ile', linewidth=2)
axes[0, 1].set_xlabel('Visual GA (weeks)')
axes[0, 1].set_ylabel('True HC (mm)')
axes[0, 1].set_title('HC vs FetalCLIP Visual GA')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Label distribution
axes[1, 0].bar(['Normal', 'Abnormal'], [normal_count, abnormal_count], 
               color=['green', 'red'], alpha=0.7, edgecolor='black')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title('FetalCLIP-Based Classification')
axes[1, 0].grid(alpha=0.3, axis='y')
for i, v in enumerate([normal_count, abnormal_count]):
    axes[1, 0].text(i, v + 5, f'{v}\n({v/len(valid_labels)*100:.1f}%)', 
                    ha='center', fontweight='bold')

# Similarity score distribution
axes[1, 1].hist(results_df['max_similarity'], bins=30, edgecolor='black', alpha=0.7, color='purple')
axes[1, 1].set_xlabel('Max Similarity Score')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('FetalCLIP Confidence Distribution')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
print(f"✅ Visualization saved to: {OUTPUT_PLOT}")

print("\n" + "=" * 50)
print("ANALYSIS COMPLETE")
print("=" * 50)
