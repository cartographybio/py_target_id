
tid.utils.h5ls(manifest["Local_h5map"].values[0])

h5_file = manifest["Local_h5map"].values[0]

with h5py.File(h5_file, 'r') as f:
    # Read coldata (cell metadata)
    coldata_group = f['coldata']
    coldata_dict = {}
    
    for key in coldata_group.keys():
        data = coldata_group[key][:]
        # Decode byte strings if needed
        if data.dtype.kind == 'S':
            data = data.astype(str)
        coldata_dict[key] = data
    
    coldata_df = pd.DataFrame(coldata_dict)

coldata_df["malig"].value_counts()
coldata_df[coldata_df["malig"]=="malig"]["nCount_RNA"]
coldata_df[coldata_df["malig"]=="malig"]["nFeature_RNA"]


import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize containers for results
malig_counts_list = []
ncount_malig_list = []
nfeature_malig_list = []
file_labels = []

# Loop through each file in manifest
for idx, h5_file in enumerate(manifest["Local_h5map"].values):
    print(f"Processing file {idx + 1}/{len(manifest)}: {h5_file}")
    
    try:
        with h5py.File(h5_file, 'r') as f:
            # Read coldata (cell metadata)
            coldata_group = f['coldata']
            coldata_dict = {}
            
            for key in coldata_group.keys():
                data = coldata_group[key][:]
                # Decode byte strings if needed
                if data.dtype.kind == 'S':
                    data = data.astype(str)
                coldata_dict[key] = data
            
            coldata_df = pd.DataFrame(coldata_dict)
            
            # Summary 1: Malignancy value counts
            malig_counts = coldata_df["malig"].value_counts()
            malig_counts_list.append(malig_counts)
            
            # Summary 2: nCount_RNA for malignant cells
            ncount_malig = coldata_df[coldata_df["malig"] == "malig"]["nCount_RNA"]
            ncount_malig_list.append(ncount_malig)
            
            # Summary 3: nFeature_RNA for malignant cells
            nfeature_malig = coldata_df[coldata_df["malig"] == "malig"]["nFeature_RNA"]
            nfeature_malig_list.append(nfeature_malig)
            
            # Store file label (extract from path)
            file_label = h5_file.split('/')[-1].replace('.h5', '')
            file_labels.append(file_label)
            
    except Exception as e:
        print(f"  Error processing {h5_file}: {e}")
        continue

import re

# Function to extract CBP ID
def extract_cbp(filename):
    match = re.search(r'(CBP\d+)', filename)
    return match.group(1) if match else None

# Combine malig counts into a single dataframe
malig_counts_df = pd.concat(malig_counts_list, axis=1)
malig_counts_df.columns = file_labels
malig_counts_df = malig_counts_df.fillna(0).T
malig_counts_df['id'] =  list(pd.Series(malig_counts_df.index).apply(extract_cbp))

print("\nMalignancy counts by file:")
print(malig_counts_df)

# Combine nCount and nFeature into dataframes for easier plotting
ncount_df = pd.DataFrame([
    {
        'CBP': extract_cbp(file_labels[i]),
        'nCount_RNA': val
    }
    for i, counts in enumerate(ncount_malig_list)
    for val in counts
])

nfeature_df = pd.DataFrame([
    {
        'CBP': extract_cbp(file_labels[i]),
        'nFeature_RNA': val
    }
    for i, features in enumerate(nfeature_malig_list)
    for val in features
])

nfeature_df['id']



import matplotlib.pyplot as plt
import numpy as np

# Calculate total cells and sort (descending)
malig_counts_df['total'] = malig_counts_df['nonmalig'] + malig_counts_df['malig']
malig_counts_sorted = malig_counts_df.sort_values('total', ascending=False)

# Define color palette
color_palette = ["lightgrey", "#e0f3db", "#6BC291", "#18B5CB", "#2E95D2", "#28154C"]

# Identify samples with malig < 100
low_malig_mask = malig_counts_sorted['malig'] < 100

# Create stacked bar plot with subplots below for ncount and nfeature
fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), height_ratios=[1.5, 1, 1], sharex=True)

x = np.arange(len(malig_counts_sorted))
width = 0.6

# Plot nonmalig and malig stacked
ax.bar(x, malig_counts_sorted['nonmalig'], width, label='Nonmalignant', color='lightgrey', edgecolor='black', linewidth=0.5)
ax.bar(x, malig_counts_sorted['malig'], width, bottom=malig_counts_sorted['nonmalig'], 
       color='#6BC291', label='Malignant', edgecolor='black', linewidth=0.5)

ax.set_xlabel('Sample ID')
ax.set_ylabel('Cell Count')
ax.set_title('Malignancy Distribution Across Samples (sorted by total cells)\n(Red text indicates malig < 100)')
ax.set_xticks(x)

# Color x-axis labels red if malig < 100
labels = ax.set_xticklabels(malig_counts_sorted['id'], rotation=45, ha='right')
for i, label in enumerate(labels):
    if low_malig_mask.iloc[i]:
        label.set_color('red')

ax.legend()

# Plot ncount_log10 below (for malignant cells only)
# Calculate median nCount_RNA for each CBP (from malignant cells)
ncount_by_cbp = ncount_df[ncount_df['CBP'].isin(malig_counts_sorted['id'])].groupby('CBP')['nCount_RNA'].median()

# Reorder to match sorted order
ncount_sorted = ncount_by_cbp.reindex(malig_counts_sorted['id'])

# Plot ncount (not log10)
ax2.bar(x, ncount_sorted.values, width, color='#18B5CB', alpha=0.7, edgecolor='black', linewidth=0.5)
ax2.set_ylabel('Median nCount_RNA\n(malignant cells)')
ax2.set_xlabel('Sample ID')
ax2.set_title('Median Transcripts per Cell')

# Rotate x-axis labels on bottom plot
labels2 = ax2.set_xticklabels(malig_counts_sorted['id'], rotation=45, ha='right')
for i, label in enumerate(labels2):
    if low_malig_mask.iloc[i]:
        label.set_color('red')

# Plot nfeature (for malignant cells only)
nfeature_by_cbp = nfeature_df[nfeature_df['CBP'].isin(malig_counts_sorted['id'])].groupby('CBP')['nFeature_RNA'].median()
nfeature_sorted = nfeature_by_cbp.reindex(malig_counts_sorted['id'])

ax3.bar(x, nfeature_sorted.values, width, color='#2E95D2', alpha=0.7, edgecolor='black', linewidth=0.5)
ax3.set_ylabel('Median nFeature_RNA\n(malignant cells)')
ax3.set_xlabel('Sample ID')
ax3.set_title('Median Genes per Cell')

# Rotate x-axis labels on third plot
labels3 = ax3.set_xticklabels(malig_counts_sorted['id'], rotation=45, ha='right')
for i, label in enumerate(labels3):
    if low_malig_mask.iloc[i]:
        label.set_color('red')

plt.tight_layout()
plt.savefig('malig_stacked_barplot.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Samples with malig < 100:")
print(malig_counts_sorted[low_malig_mask][['id', 'nonmalig', 'malig', 'total']])





