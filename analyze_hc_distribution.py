import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load HC18 data
df = pd.read_csv(r'1327317\training_set_pixel_size_and_HC.csv')

# Calculate percentiles
print("=" * 50)
print("HC18 DATASET - HEAD CIRCUMFERENCE DISTRIBUTION")
print("=" * 50)
print(f"\nTotal samples: {len(df)}")
print(f"Mean HC: {df['head circumference (mm)'].mean():.2f} mm")
print(f"Std HC: {df['head circumference (mm)'].std():.2f} mm")
print(f"Min HC: {df['head circumference (mm)'].min():.2f} mm")
print(f"Max HC: {df['head circumference (mm)'].max():.2f} mm")

print("\n" + "=" * 50)
print("PERCENTILE DISTRIBUTION")
print("=" * 50)
percentiles = [1, 3, 5, 10, 25, 50, 75, 90, 95, 97, 99]
for p in percentiles:
    val = np.percentile(df['head circumference (mm)'], p)
    print(f"{p:2d}th percentile: {val:6.1f} mm")

# Hadlock formula for GA estimation
def hadlock_hc_to_ga(hc_mm):
    """Convert HC to GA using Hadlock formula"""
    if hc_mm <= 0:
        return None
    log_hc = np.log10(hc_mm)
    a, b, c = 0.0004, 0.0079, 1.6988 - log_hc
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return None
    ga = (-b + np.sqrt(discriminant)) / (2*a)
    return ga if 14 <= ga <= 42 else None

# Estimate GA for each sample
df['estimated_ga_weeks'] = df['head circumference (mm)'].apply(hadlock_hc_to_ga)
df['estimated_ga_weeks'] = df['estimated_ga_weeks'].round(1)

print("\n" + "=" * 50)
print("ESTIMATED GESTATIONAL AGE DISTRIBUTION")
print("=" * 50)
valid_ga = df['estimated_ga_weeks'].dropna()
print(f"Valid GA estimations: {len(valid_ga)} / {len(df)}")
print(f"Mean GA: {valid_ga.mean():.1f} weeks")
print(f"Std GA: {valid_ga.std():.1f} weeks")
print(f"Min GA: {valid_ga.min():.1f} weeks")
print(f"Max GA: {valid_ga.max():.1f} weeks")

print("\n" + "=" * 50)
print("GA DISTRIBUTION BY TRIMESTER")
print("=" * 50)
first_tri = len(valid_ga[valid_ga < 14])
second_tri = len(valid_ga[(valid_ga >= 14) & (valid_ga < 28)])
third_tri = len(valid_ga[valid_ga >= 28])
print(f"First trimester (<14w): {first_tri} samples ({first_tri/len(valid_ga)*100:.1f}%)")
print(f"Second trimester (14-28w): {second_tri} samples ({second_tri/len(valid_ga)*100:.1f}%)")
print(f"Third trimester (≥28w): {third_tri} samples ({third_tri/len(valid_ga)*100:.1f}%)")

# GA-specific percentile reference (from literature)
ga_percentile_table = {
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

def interpolate_percentile(ga, percentile_key):
    """Interpolate percentile for given GA"""
    ga_keys = sorted(ga_percentile_table.keys())
    
    if ga <= ga_keys[0]:
        return ga_percentile_table[ga_keys[0]][percentile_key]
    if ga >= ga_keys[-1]:
        return ga_percentile_table[ga_keys[-1]][percentile_key]
    
    for i in range(len(ga_keys) - 1):
        if ga_keys[i] <= ga < ga_keys[i+1]:
            ga1, ga2 = ga_keys[i], ga_keys[i+1]
            val1 = ga_percentile_table[ga1][percentile_key]
            val2 = ga_percentile_table[ga2][percentile_key]
            return val1 + (val2 - val1) * (ga - ga1) / (ga2 - ga1)
    
    return None

def classify_with_ga_percentile(hc_mm, ga_weeks):
    """Classify as normal/abnormal based on GA-specific percentiles"""
    if ga_weeks is None or np.isnan(ga_weeks) or ga_weeks < 16 or ga_weeks > 40:
        return None
    
    p3 = interpolate_percentile(ga_weeks, 'p3')
    p97 = interpolate_percentile(ga_weeks, 'p97')
    
    if p3 is None or p97 is None:
        return None
    
    if hc_mm < p3 or hc_mm > p97:
        return 1  # Abnormal
    else:
        return 0  # Normal

# Apply GA-based classification
df['ga_based_label'] = df.apply(
    lambda row: classify_with_ga_percentile(row['head circumference (mm)'], row['estimated_ga_weeks']),
    axis=1
)

print("\n" + "=" * 50)
print("GA-BASED AUTOMATED LABELING RESULTS")
print("=" * 50)
valid_labels = df['ga_based_label'].dropna()
normal_count = (valid_labels == 0).sum()
abnormal_count = (valid_labels == 1).sum()
print(f"Valid labels: {len(valid_labels)} / {len(df)}")
print(f"Normal: {normal_count} samples ({normal_count/len(valid_labels)*100:.1f}%)")
print(f"Abnormal: {abnormal_count} samples ({abnormal_count/len(valid_labels)*100:.1f}%)")

# Check extreme cases
extreme_low = df[df['head circumference (mm)'] < 100]
extreme_high = df[df['head circumference (mm)'] > 300]
print("\n" + "=" * 50)
print("EXTREME CASES (Potential Annotation Issues)")
print("=" * 50)
print(f"HC < 100 mm: {len(extreme_low)} samples")
if len(extreme_low) > 0:
    print(f"  Min: {extreme_low['head circumference (mm)'].min():.1f} mm")
    print(f"  Estimated GA: {extreme_low['estimated_ga_weeks'].min():.1f} weeks (pre-viable)")
print(f"HC > 300 mm: {len(extreme_high)} samples")
if len(extreme_high) > 0:
    print(f"  Max: {extreme_high['head circumference (mm)'].max():.1f} mm")
    print(f"  Estimated GA: {extreme_high['estimated_ga_weeks'].max():.1f} weeks")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# HC distribution
axes[0, 0].hist(df['head circumference (mm)'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(df['head circumference (mm)'].mean(), color='red', linestyle='--', label='Mean')
axes[0, 0].set_xlabel('Head Circumference (mm)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('HC Distribution in HC18 Dataset')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# GA distribution
axes[0, 1].hist(valid_ga, bins=30, edgecolor='black', alpha=0.7, color='green')
axes[0, 1].axvline(valid_ga.mean(), color='red', linestyle='--', label='Mean')
axes[0, 1].set_xlabel('Estimated Gestational Age (weeks)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Estimated GA Distribution (Hadlock Formula)')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# HC vs GA scatter
axes[1, 0].scatter(df['estimated_ga_weeks'], df['head circumference (mm)'], alpha=0.5, s=10)
# Plot percentile curves
ga_range = np.linspace(16, 40, 100)
p3_curve = [interpolate_percentile(ga, 'p3') for ga in ga_range]
p50_curve = [interpolate_percentile(ga, 'p50') for ga in ga_range]
p97_curve = [interpolate_percentile(ga, 'p97') for ga in ga_range]
axes[1, 0].plot(ga_range, p3_curve, 'r--', label='3rd percentile', linewidth=2)
axes[1, 0].plot(ga_range, p50_curve, 'g-', label='50th percentile', linewidth=2)
axes[1, 0].plot(ga_range, p97_curve, 'r--', label='97th percentile', linewidth=2)
axes[1, 0].set_xlabel('Estimated Gestational Age (weeks)')
axes[1, 0].set_ylabel('Head Circumference (mm)')
axes[1, 0].set_title('HC vs GA with Clinical Percentiles')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Label distribution
label_counts = valid_labels.value_counts()
axes[1, 1].bar(['Normal', 'Abnormal'], [normal_count, abnormal_count], 
               color=['green', 'red'], alpha=0.7, edgecolor='black')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('GA-Based Automated Labeling')
axes[1, 1].grid(alpha=0.3, axis='y')
for i, v in enumerate([normal_count, abnormal_count]):
    axes[1, 1].text(i, v + 5, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('HC18_distribution_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ Visualization saved: HC18_distribution_analysis.png")

# Save labeled dataset
output_df = df[['filename', 'pixel size(mm)', 'head circumference (mm)', 
                'estimated_ga_weeks', 'ga_based_label']].copy()
output_df.to_csv('HC18_with_GA_labels.csv', index=False)
print("✅ Labeled dataset saved: HC18_with_GA_labels.csv")

print("\n" + "=" * 50)
print("ANALYSIS COMPLETE")
print("=" * 50)
