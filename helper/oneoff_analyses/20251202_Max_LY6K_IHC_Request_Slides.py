import py_target_id as tid
from rpy2.robjects import r
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import os
from importlib.resources import files
from py_target_id import utils

#Load Cohort
IND = os.path.basename(os.getcwd())

#Load
manifest, malig_adata, malig_med_adata, ref_adata, ref_med_adata = utils.load_cohort(IND)

#Info
m = malig_med_adata[:, "LY6K"]
ref_ff = ref_med_adata[:, "LY6K"]
ref_sc = tid.utils.get_ref_lv4_sc_med_adata()[:, "LY6K"]

#Create Data Frames
df_malig = pd.DataFrame({"CBP" : m.obs["Patient"].values, "Log2_CP10K" : np.log2(m.X[:,0] + 1), "Pos_Exp" : np.where(m.layers["positivity"][:, 0].astype(bool), "Positive", "Negative")})
df_malig["CBP"]=df_malig["CBP"].str.replace("Breast_","").str.replace("_FFPE","")
df_IHC = pd.read_csv("TNBC_IHC_Scoring.csv")
df_IHC.columns = ["CBP", "H_Score"]
df_IHC["CBP"] = df_IHC["CBP"].str.replace(" ", "")
df_malig = pd.merge(df_malig, df_IHC, on = "CBP", how = "outer")
df_malig = df_malig.dropna()

import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_curve, auc
import numpy as np

# Calculate correlations
pearson_r, pearson_p = pearsonr(df_malig['Log2_CP10K'], df_malig['H_Score'])
spearman_r, spearman_p = spearmanr(df_malig['Log2_CP10K'], df_malig['H_Score'])

# Calculate R² for linear fit
z = np.polyfit(df_malig['Log2_CP10K'], df_malig['H_Score'], 1)
p = np.poly1d(z)
y_pred = p(df_malig['Log2_CP10K'])
ss_res = np.sum((df_malig['H_Score'] - y_pred) ** 2)
ss_tot = np.sum((df_malig['H_Score'] - df_malig['H_Score'].mean()) ** 2)
r_squared = 1 - (ss_res / ss_tot)

# Convert Pos_Exp to binary (1 for Positive, 0 for Negative)
y_true = (df_malig['Pos_Exp'] == 'Positive').astype(int)

# Use H_Score as predictor for ROC
fpr, tpr, thresholds = roc_curve(y_true, df_malig['H_Score'])
roc_auc = auc(fpr, tpr)

# Set publication-quality style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

# Define colors
color_pos = '#28154C'
color_neg = '#D3D3D3'
color_pos_edge = '#1a0d2e'
color_neg_edge = '#808080'

# Create side-by-side plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=300)

# Left plot: Scatter with line of best fit
positive = df_malig[df_malig['Pos_Exp'] == 'Positive']
negative = df_malig[df_malig['Pos_Exp'] == 'Negative']

ax1.scatter(negative['Log2_CP10K'], negative['H_Score'], 
           color=color_neg, label='Negative', s=90, alpha=0.8, edgecolors='black', linewidth=1.2)
ax1.scatter(positive['Log2_CP10K'], positive['H_Score'], 
           color=color_pos, label='Positive', s=90, alpha=0.85, edgecolors='black', linewidth=1.2)

# Add line of best fit
x_line = np.linspace(df_malig['Log2_CP10K'].min(), df_malig['Log2_CP10K'].max(), 100)
y_line = p(x_line)
ax1.plot(x_line, y_line, color='#2c3e50', lw=2.5, linestyle='--', alpha=0.8, label='Linear fit')

ax1.set_xlabel('Log₂(CP10K + 1)', fontsize=11, fontweight='bold')
ax1.set_ylabel('H-Score', fontsize=11, fontweight='bold')
ax1.set_title('(A) Expression vs Immunohistochemistry', fontsize=12, fontweight='bold', pad=15)
ax1.legend(fontsize=10, loc='upper left', framealpha=0.95, edgecolor='black', fancybox=False)
ax1.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
ax1.set_axisbelow(True)
stats_text = f'Pearson: r = {pearson_r:.3f}\n(p = {pearson_p:.2e})\n\nSpearman: ρ = {spearman_r:.3f}\n(p = {spearman_p:.2e})\n\nn = {len(df_malig)}'
ax1.text(0.98, 0.02, stats_text, transform=ax1.transAxes, fontsize=9.5,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#ecf0f1', alpha=0.95, edgecolor='#34495e', linewidth=1.2),
        family='monospace')

# Identify and label outliers
# Define outliers as points that are >1.5 std from the line
residuals = np.abs(df_malig['H_Score'] - p(df_malig['Log2_CP10K']))
outlier_threshold = residuals.mean() + 1.5 * residuals.std()
outliers = df_malig[residuals > outlier_threshold]

for idx, row in outliers.iterrows():
    ax1.annotate(row['CBP'], 
                xy=(row['Log2_CP10K'], row['H_Score']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, alpha=0.7)

# Right plot: ROC curve
ax2.plot(fpr, tpr, color=color_pos, lw=3, label=f'H-Score (AUC = {roc_auc:.3f})', zorder=3)
ax2.plot([0, 1], [0, 1], color='#95a5a6', lw=2, linestyle='--', label='Random classifier', zorder=1)
ax2.fill_between(fpr, tpr, alpha=0.15, color=color_pos)

ax2.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
ax2.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
ax2.set_title('(B) ROC: H-Score Predicting Pos_Exp', fontsize=12, fontweight='bold', pad=15)
ax2.legend(fontsize=10, loc='lower right', framealpha=0.95, edgecolor='black', fancybox=False)
ax2.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
ax2.set_axisbelow(True)
ax2.set_xlim([-0.02, 1.02])
ax2.set_ylim([-0.02, 1.05])

# Make plots square in size (not aspect ratio)
fig.set_size_inches(12, 6)

# Improve spine visibility
for spine in ax1.spines.values():
    spine.set_linewidth(1.2)
    spine.set_color('#2c3e50')
for spine in ax2.spines.values():
    spine.set_linewidth(1.2)
    spine.set_color('#2c3e50')

plt.tight_layout()
plt.savefig('cbp_malig_scatter.pdf', format='pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()