import subprocess; import os; subprocess.run(["bash", os.path.expanduser("~/update_tid.sh")]) #Update Quick

import py_target_id as tid
from rpy2.robjects import r
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import os
from importlib.resources import files
from py_target_id import utils
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import numpy as np
from joblib import dump
import csv
from tqdm import tqdm
from functools import reduce
pd.set_option('display.max_columns', None)

#Surface
surface = tid.utils.surface_genes()
tabs = set(tid.utils.tabs_genes())

#Risk
haz_sgl = tid.utils.get_single_risk_scores()
haz_dbl = tid.utils.get_multi_risk_scores()

######################################################################
# Create Single Target Training Data Sets
######################################################################

inds = {
    "AML_Single" : "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/AML/AML.Single.Results.20251029.parquet",
    "CRC_Single" : "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/CRC/CRC.Single.Results.20251029.parquet",
    "KIRC_Single" : "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/KIRC/KIRC.Single.Results.20251029.parquet",
    "LUAD_Single" : "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/LUAD.Magellan/LUAD.Magellan.Single.Results.20251029.parquet",
    "TNBC_Single" : "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/TNBC.Magellan/TNBC.Magellan.Single.Results.20251029.parquet",
    "PDAC_Single" : "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/PDAC_FFPE/PDAC_FFPE.Single.Results.20251029.parquet"    
}

df_ind = pd.read_parquet(inds["TNBC_Single"])
df_ind = df_ind.loc[df_ind["gene_name"].isin(surface)].copy()
df_ind = pd.merge(df_ind, haz_sgl, on = "gene_name", how="left")
df_ind = tid.run.target_quality_v2_01(df_ind)


dfs = []
for indication, path in inds.items():
    print(path)
    df = pd.read_parquet(path)
    df = df.loc[df["gene_name"].isin(surface)].copy()
    df = pd.merge(df, haz_sgl, on="gene_name", how="left")
    df = tid.run.target_quality_v2_01(df)
    df["indication"] = indication
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

inds = {
    "CRC_Multi" : "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/CRC/CRC.Multi.Results.20251029.parquet",
    "LUAD_Multi" : "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/LUAD.Magellan/LUAD.Magellan.Multi.Results.20251029.parquet",
    "TNBC_Multi" : "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/TNBC.Magellan/TNBC.Magellan.Multi.Results.20251029.parquet"
}

dfs_m = []
for indication, path in inds.items():
    print(path)
    df = pd.read_parquet(path)
    df = pd.merge(df, haz_dbl, on="gene_name", how="left")
    df = tid.run.target_quality_v2_01(df)
    df["indication"] = indication
    dfs_m.append(df)

dfs_m_all = pd.concat(dfs_m, ignore_index=True)

df_combined = pd.concat([dfs_m_all, df_all], ignore_index=True)


df_combined.to_feather("All_IND_TQ.feather")

#Engineer These Features
df_combined["LFC_On_50_vs_Tox"] = np.log2(df_combined["On_Val_50"].values + 1) - np.log2(
    np.maximum(np.maximum(df_combined["Tox_Brain"].values, df_combined["Tox_Heart"].values), df_combined["Tox_Lung"].values) + 1
)
df_combined["TI_Tox"] = np.minimum(np.minimum(df_combined["TI_Brain"].values, df_combined["TI_Heart"].values), df_combined["TI_Lung"].values)

feature_cols = [
    'N_Off_Targets', 'N_Off_Targets_0.5',
    'Specificity', 'Corrected_Specificity',
    'P_Pos_Per',
    'P_Pos', 
    'SC_2nd_Target_Val',
    'On_Val_75', 'On_Val_50', 
    'LFC_On_50_vs_Tox', #'Tox_Brain', 'Tox_Heart', 'Tox_Lung',
    'TI_Tox', #'TI_Brain', 'TI_Heart', 'TI_Lung', 
    'TI_NonImmune', 'TI',
    'N_Off_Targets_1.0', 'N_Off_Targets_0.25', 'N_Off_Targets_0.1', 'N_Off_Targets_0.05',
    'Hazard_SC_v1', 'Hazard_FFPE_v1', 'Hazard_GTEX_v1',
    'GTEX_Tox_Tier1', 'GTEX_Tox_Tier2'
]





















































print(f"Combined dataset: {len(df_combined)} rows")
print(f"Score range: {df_combined['TargetQ_Final_v2'].min():.2f} - {df_combined['TargetQ_Final_v2'].max():.2f}")

feature_cols = [
    'N_Off_Targets', 'N_Off_Targets_0.5',
    'Specificity', 'Corrected_Specificity',
    'P_Pos_Per',
    'P_Pos', 
    'SC_2nd_Target_Val',
    'On_Val_75', 'On_Val_50', 
    'Tox_Brain', 'Tox_Heart', 'Tox_Lung',
    'TI_Brain', 'TI_Heart', 'TI_Lung', 'TI_NonImmune', 'TI',
    'N_Off_Targets_1.0', 'N_Off_Targets_0.25', 'N_Off_Targets_0.1', 'N_Off_Targets_0.05',
    'Hazard_SC_v1', 'Hazard_FFPE_v1', 'Hazard_GTEX_v1',
    'GTEX_Tox_Tier1', 'GTEX_Tox_Tier2'
]


from lightgbm import LGBMRegressor
import lightgbm

# Filter to TargetQ > 50
mask = df_combined['TargetQ_Final_v2'] > 50
df_filtered = df_combined[mask].copy()

print(f"Filtered dataset: {len(df_filtered)} rows")
print(f"Score range: {df_filtered['TargetQ_Final_v2'].min():.2f} - {df_filtered['TargetQ_Final_v2'].max():.2f}")

# Your curated features
feature_cols = [
    'N_Off_Targets', 'N_Off_Targets_0.5',
    'Specificity', 'Corrected_Specificity',
    'P_Pos_Per',
    'P_Pos', 
    'SC_2nd_Target_Val',
    'On_Val_75', 'On_Val_50', 
    'Tox_Brain', 'Tox_Heart', 'Tox_Lung',
    'TI_Brain', 'TI_Heart', 'TI_Lung', 'TI_NonImmune', 'TI',
    'N_Off_Targets_1.0', 'N_Off_Targets_0.25', 'N_Off_Targets_0.1', 'N_Off_Targets_0.05',
    'Hazard_SC_v1', 'Hazard_FFPE_v1', 'Hazard_GTEX_v1',
    'GTEX_Tox_Tier1', 'GTEX_Tox_Tier2'
]

X = df_filtered[feature_cols].fillna(0)
y = df_filtered['TargetQ_Final_v2']

# Stratified split with top in test
y_bins = pd.qcut(y, q=5, labels=False, duplicates='drop')
top_10_pct_idx = y.nlargest(int(len(y) * 0.10)).index
remaining_idx = y.index.difference(top_10_pct_idx)

X_top, y_top = X.loc[top_10_pct_idx], y.loc[top_10_pct_idx]
X_rem, y_rem = X.loc[remaining_idx], y.loc[remaining_idx]

X_top_train, X_top_test, y_top_train, y_top_test = train_test_split(
    X_top, y_top, test_size=0.2, random_state=42
)
X_rem_train, X_rem_test, y_rem_train, y_rem_test = train_test_split(
    X_rem, y_rem, test_size=0.2, random_state=42
)

X_train = pd.concat([X_top_train, X_rem_train])
X_test = pd.concat([X_top_test, X_rem_test])
y_train = pd.concat([y_top_train, y_rem_train])
y_test = pd.concat([y_top_test, y_rem_test])

print(f"Train: {len(X_train):,}, Test: {len(X_test):,}\n")

# Train
model = LGBMRegressor(
    num_leaves=127,
    learning_rate=0.01,
    n_estimators=1000,
    min_child_samples=2,
    lambda_l1=0.5,
    lambda_l2=0.5,
    random_state=42,
    force_col_wise=True,
    n_jobs=-1,
    device='gpu',  # Add this
    gpu_device_id=0,
    verbose=50
)

model.fit(X_train, y_train, 
          eval_set=[(X_test, y_test)],
          callbacks=[lightgbm.early_stopping(200)])

train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)

print(f"\nTrain R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")

# Top performer accuracy
predictions = model.predict(X_test)
top_mask = y_test >= 90
if top_mask.sum() > 0:
    print(f"\nTop performers (>90): {top_mask.sum()}")
    print(f"  MAE: {abs(predictions[top_mask] - y_test[top_mask]).mean():.4f}")
    print(f"  Max error: {abs(predictions[top_mask] - y_test[top_mask]).max():.4f}")


X2 = df_combined[feature_cols].fillna(0)
predictions = model.predict(X2, pred_leaf=False)






                  feature  importance
5                   P_Pos       13025
0           N_Off_Targets       10726
6       SC_2nd_Target_Val        9910
8               On_Val_50        8744
14                TI_Lung        8728
1       N_Off_Targets_0.5        7959
12               TI_Brain        7441
2             Specificity        7190
11               Tox_Lung        6416
16                     TI        4913
23         Hazard_GTEX_v1        4626
9               Tox_Brain        4444
3   Corrected_Specificity        4271
15           TI_NonImmune        4078
13               TI_Heart        3831
7               On_Val_75        3340
24         GTEX_Tox_Tier1        2854
10              Tox_Heart        2491
17      N_Off_Targets_1.0        1961
21           Hazard_SC_v1        1956
22         Hazard_FFPE_v1        1920
4               P_Pos_Per        1748
19      N_Off_Targets_0.1        1033
25         GTEX_Tox_Tier2         877
18     N_Off_Targets_0.25         817
20     N_Off_Targets_0.05         701
















from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import numpy as np

# Sample randomly
sample_size = 10000
sample_idx = np.random.choice(len(df_ind), size=sample_size, replace=False)
df_sample = df_all

# Prepare data
feature_cols = [
    'N_Off_Targets', 'N_Off_Targets_0.5',
    'Corrected_Specificity',
    'P_Pos_Per',
    'P_Pos', 
    'SC_2nd_Target_Val',
    'On_Val_50', 'Tox_Brain', 'Tox_Heart', 'Tox_Lung',
    'TI_Brain', 'TI_Heart', 'TI_Lung', 'TI',
    'N_Off_Targets_1.0', 'N_Off_Targets_0.25', 'N_Off_Targets_0.1', 'N_Off_Targets_0.05',
    'TI_NonImmune',
    'On_Val_75',
    'Specificity',
    'Hazard_SC_v1', 'Hazard_FFPE_v1', 'Hazard_GTEX_v1',
    'GTEX_Tox_Tier1', 'GTEX_Tox_Tier2'
]

X = df_sample[feature_cols]
y = df_sample['TargetQ_Final_v2']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = XGBRegressor(max_depth=5, learning_rate=0.05, n_estimators=200)
model.fit(X_train, y_train)

# Evaluate
print(f"Train R²: {model.score(X_train, y_train):.4f}")
print(f"Test R²: {model.score(X_test, y_test):.4f}")

# Feature importance
importance = model.get_booster().get_score(importance_type='gain')
for feat, score in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]:
    print(f"{feat}: {score}")

# Predict on new data
new_predictions = model.predict(X_test)

# Add to dataframe
results_df = X_test.copy()
results_df['TargetQ_predicted'] = new_predictions
results_df['TargetQ_actual'] = y_test.values
results_df['error'] = abs(results_df['TargetQ_predicted'] - results_df['TargetQ_actual'])

results_df.sort_values("TargetQ_actual", ascending=False)

# View results
print(results_df[['TargetQ_predicted', 'TargetQ_actual', 'error']].head(20))

# Check error distribution
print(f"\nMean absolute error: {results_df['error'].mean():.4f}")
print(f"Max error: {results_df['error'].max():.4f}")
print(f"Median error: {results_df['error'].median():.4f}")










df_ind

df = df_ind[df_ind["gene_name"].isin(["ENPP3", "CA9", "UGT1A9", "SLC17A3"])]

df_ind[df_ind["gene_name"]=="CEACAM5"].iloc[0]

ca9 = df_ind[df_ind["gene_name"]=="XKRX"].iloc[0]
muc21 = df_ind[df_ind["gene_name"]=="MUC21"].iloc[0]

pd.concat([
    df_ind[df_ind["gene_name"]=="XKRX"].iloc[0],
    df_ind[df_ind["gene_name"]=="MUC21"].iloc[0]
], axis=1)

inds = {
    "LUAD_Multi" : "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/LUAD.Magellan/LUAD.Magellan.Multi.Results.20251029.parquet",
    "TNBC_Multi" : "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/TNBC.Magellan/TNBC.Magellan.Multi.Results.20251029.parquet",
    "CRC_Multi" : "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/CRC/CRC.Multi.Results.20251029.parquet",
    "PDAC_Multi" : "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/PDAC_FFPE/PDAC_FFPE.Multi.Results.20251029.parquet"
}

df_m_ind = pd.read_parquet(inds["CRC_Multi"])
df_m_ind = pd.merge(df_m_ind, haz_dbl, on = "gene_name", how="left")
df_m_ind = target_quality_v2(df_m_ind)
df_m_ind = df_m_ind[df_m_ind["Positive_Final_v2"] > 15]

df_m_ind.head(20)
df_m_ind[~df_m_ind["gene_name"].str.contains("SMPDL3B|PRAME|CALML5|VTCN1|ABCC11|MAGEA4|COL2A1|LY6K|RASD2|CT83|LY6G6D|IGF2")].head(20)

pd.concat([
    df_m_ind[df_m_ind["gene_name"]=="CEACAM5_DPEP1"].iloc[0],
    df_m_ind[df_m_ind["gene_name"]=="CDH17_CEACAM5"].iloc[0]
], axis=1)

df_m_ind_tabs = df_m_ind[df_m_ind["gene_name"].str.split("_").apply(lambda x: any(g in tabs for g in x))].copy()
df_m_ind_tabs2 = df_m_ind_tabs[df_m_ind_tabs["GTEX_Tox_Tier1"] <= 1]
df_m_ind_tabs2 = df_m_ind_tabs2[~df_m_ind_tabs2["gene_name"].str.contains("SMPDL3B|PRAME|CALML5|VTCN1|ABCC11|MAGEA4|COL2A1|LY6K|RASD2|CT83|LY6G6D|IGF2")]
df_m_ind_tabs2 = df_m_ind_tabs2[df_m_ind_tabs2["Positive_Final_v2"] > 35]
df_m_ind_tabs2.head(20)


df_m_ind[df_m_ind["gene_name"].str.contains("CDH17")].head(30)

pd.concat([
    df_m_ind_tabs[df_m_ind_tabs["gene_name"]=="CEACAM6_HPN"].iloc[0],
    df_m_ind_tabs[df_m_ind_tabs["gene_name"]=="CEACAM5_HPN"].iloc[0]
], axis=1)

pd.concat([
    df_m_ind[df_m_ind["gene_name"]=="DPP4_MUC21"].iloc[0],
    df_m_ind[df_m_ind["gene_name"]=="EPCAM_MUC21"].iloc[0]
], axis=1)

pd.concat([
    df_m_ind_tabs[df_m_ind_tabs["gene_name"]=="CEACAM6_DPEP1"].iloc[0],
    df_m_ind_tabs[df_m_ind_tabs["gene_name"]=="CEACAM5_DPEP1"].iloc[0]
], axis=1)

df = df_m_ind[df_m_ind["gene_name"].isin(["DPP4_MUC21", "EPCAM_MUC21"])]


df_m_ind_tabs[df_m_ind_tabs["gene_name"]=="LRP8_VTCN1"].iloc[0]
df_m_ind_tabs[df_m_ind_tabs["gene_name"]=="SEZ6L2_SLC34A2"].iloc[0]



tabs = set(tid.utils.tabs_genes())
df_ind2 = df_ind[df_ind["gene_name"].str.split("_").apply(lambda x: any(g in tabs for g in x))].copy()
df_ind2.head(50)
df_ind2[df_ind2["gene_name"]=="ADAM8_MSLN"].iloc[0]



np.max(abs(df_ind[["Positive_Final_0.5"]].values - df_ind[["P_Pos_Per"]].values * 100))






def compute_target_quality_score_v2(  # NO SURFACE ASSUME ALL ARE SURFACE
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute target quality scores with surface protein evidence.
    
    Final score normalization:
    - 0.25: penalty weight per failed specificity criterion (Score_1, Score_2, Score_3 == 10)
    - 1.75: normalization constant; scales penalized scores to 0-100 range where:
      * penalized_scores range: 0 (best) to 1.75 (worst)
      * TargetQ_Final_vNS = (100 / 1.75) * (1.75 - penalized_scores)
    """
    
    import numpy as np
    
    # Work with copies to avoid modifying intermediate columns
    Score_1 = pd.Series(10, index=df.index) # Too many off Targets if 10
    Score_1.loc[df['N_Off_Targets'] <= 3] = df.loc[df['N_Off_Targets'] <= 3, 'N_Off_Targets']
    Score_1.loc[df['N_Off_Targets'] > 3] = 3 + (df.loc[df['N_Off_Targets'] > 3, 'N_Off_Targets'] - 3) * 2
    Score_1 = np.minimum(Score_1, 10)
    
    Score_2 = pd.Series(10, index=df.index) # Too many off Targets if 10
    Score_2.loc[df['N_Off_Targets_0.5'] <= 3] = df.loc[df['N_Off_Targets_0.5'] <= 3, 'N_Off_Targets_0.5']
    Score_2.loc[df['N_Off_Targets_0.5'] > 3] = 3 + (df.loc[df['N_Off_Targets_0.5'] > 3, 'N_Off_Targets_0.5'] - 3) * 2
    Score_2 = np.minimum(Score_2, 10)
    
    Score_3 = pd.Series(10, index=df.index) # Did Not Converge if 10
    Score_3.loc[df['Corrected_Specificity'] >= 0.75] = 0
    Score_3.loc[(df['Corrected_Specificity'] >= 0.5) & (df['Corrected_Specificity'] < 0.75)] = 1
    Score_3.loc[(df['Corrected_Specificity'] >= 0.35) & (df['Corrected_Specificity'] < 0.5)] = 3
    
    Score_4 = pd.Series(10, index=df.index)
    Score_4.loc[df['P_Pos_Per'] > 0.25] = 0
    Score_4.loc[(df['P_Pos_Per'] > 0.15) & (df['P_Pos_Per'] <= 0.25)] = 1
    Score_4.loc[(df['P_Pos_Per'] > 0.025) & (df['P_Pos_Per'] <= 0.15)] = 3
    Score_4.loc[df['N_Pos_Val'] == 1] = 10
    
    Score_5 = pd.Series(10, index=df.index)
    Score_5.loc[df['P_Pos'] > 0.25] = 0
    Score_5.loc[(df['P_Pos'] > 0.15) & (df['P_Pos'] <= 0.25)] = 1
    Score_5.loc[(df['P_Pos'] > 0.025) & (df['P_Pos'] <= 0.15)] = 3
    Score_5.loc[df['N_Pos'] == 1] = 10
    
    Score_6 = pd.Series(10, index=df.index)
    Score_6.loc[df['SC_2nd_Target_Val'] > 2] = 0
    Score_6.loc[(df['SC_2nd_Target_Val'] > 1) & (df['SC_2nd_Target_Val'] <= 2)] = 1
    Score_6.loc[(df['SC_2nd_Target_Val'] > 0.5) & (df['SC_2nd_Target_Val'] <= 1)] = 3
    Score_6.loc[(df['SC_2nd_Target_Val'] > 0.1) & (df['SC_2nd_Target_Val'] <= 0.5)] = 5
    
    Score_7 = pd.Series(10, index=df.index)  # Legacy
    
    # Compute final TargetQ score
    score_columns = [Score_1, Score_2, Score_3, Score_5, Score_6, Score_7]
    penalty_columns = [Score_1, Score_2, Score_3]  # Specificity criteria that get penalized
    
    # Step 1: Sum all score components (each 0-10, so max raw sum = 60)
    raw_scores = pd.concat(score_columns, axis=1).sum(axis=1)
    
    # Step 2: Count how many specificity criteria failed (Score == 10 means criterion not met)
    penalty_count = pd.concat(penalty_columns, axis=1).apply(lambda x: (x == 10).sum(), axis=1)
    
    # Step 3: Define scaling constants
    # penalty_weight: penalty of 0.25 per failed specificity criterion
    penalty_weight = 0.25
    # max_score: theoretical maximum if all scores hit 10 (6 scores × 10)
    max_score = len(score_columns) * 10
    # max_penalized_score: worst case = 1.0 (raw normalized) + 0.75 (3 penalties × 0.25)
    max_penalized_score = 1 + len(penalty_columns) * penalty_weight
    
    # Step 4: Calculate penalized score (range 0 to max_penalized_score)
    # Normalize raw scores to 0-1, then add penalty for each failed criterion
    penalized_scores = raw_scores / max_score + penalty_weight * penalty_count
    
    # Step 5: Invert and scale to 0-100 range
    # Lower penalized_scores → higher TargetQ (better target)
    # 0 penalized_score → 100 TargetQ; max_penalized_score → 0 TargetQ
    df['TargetQ_Final_vNS'] = (100 / max_penalized_score) * (max_penalized_score - penalized_scores)
    df.loc[df['Target_Val'] == 0, 'TargetQ_Final_vNS'] = 0
    
    return df


def piecewise_linear_vectorized(value, thresholds, scores):
    """
    Vectorized piecewise linear interpolation.
    thresholds and scores must match the linear_piecewise_score logic.
    """
    result = np.zeros_like(value, dtype=float)
    
    # Top threshold
    result = np.where(value > thresholds[0], scores[0], result)
    
    # Bottom threshold
    result = np.where(value <= thresholds[-1], scores[-1], result)
    
    # Interpolate between thresholds
    for i in range(len(thresholds) - 1):
        mask = (value > thresholds[i + 1]) & (value <= thresholds[i])
        if mask.any():
            t_high = thresholds[i]
            t_low = thresholds[i + 1]
            s_high = scores[i]
            s_low = scores[i + 1]
            interp = s_high + (t_high - value) / (t_high - t_low) * (s_low - s_high)
            result = np.where(mask, interp, result)
    
    return result

def add_training_metrics_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """Training Metrics - Vectorized"""
    
    df = df.copy()
    
    # Handle NaNs
    for col in ['Positive_Final_v2', 'SC_2nd_Target_Val', 'On_Val_75', 'On_Val_50']:
        df[col] = df[col].fillna(0)
    for col in ['sc_hazard_risk', 'ff_hazard_risk']:
        df[col] = df[col].fillna(100000)
    
    # Custom Scores
    df["Training.Specificity_Combo"] = df["N_Off_Targets"] + (1 - df["Corrected_Specificity"])
    df['Training.Score_N_Off_Spec'] = np.where(df['N_Off_Targets'] <= 3, df['N_Off_Targets'], 10)
    
    # Conditional N_Off scoring based on On_Val_75
    df['Training.Score_N_Off_Val'] = 10  # Default

    df.loc[df['On_Val_75'] > 0.25, 'Training.Score_N_Off_Val'] = np.where(
        df.loc[df['On_Val_75'] > 0.25, 'N_Off_Targets_0.25'] <= 3,
        df.loc[df['On_Val_75'] > 0.25, 'N_Off_Targets_0.25'],
        10
    )
    df.loc[df['On_Val_75'] > 0.5, 'Training.Score_N_Off_Val'] = np.where(
        df.loc[df['On_Val_75'] > 0.5, 'N_Off_Targets_0.5'] <= 3,
        df.loc[df['On_Val_75'] > 0.5, 'N_Off_Targets_0.5'],
        10
    )
    df.loc[df['On_Val_75'] > 1, 'Training.Score_N_Off_Val'] = np.where(
        df.loc[df['On_Val_75'] > 1, 'N_Off_Targets_1.0'] <= 3,
        df.loc[df['On_Val_75'] > 1, 'N_Off_Targets_1.0'],
        10
    )
        
    # Off-target hazard with exponential weighting (strict thresholds weighted heavily)
    weights = np.array([10**2, 5**2, 2.5**2, 1**2, 0.5**2])
    off_target_cols = ['N_Off_Targets_1.0', 'N_Off_Targets_0.5', 'N_Off_Targets_0.25', 'N_Off_Targets_0.1', 'N_Off_Targets_0.05']
    
    df['Training.Off_Target_Hazard'] = 0
    for col, weight in zip(off_target_cols, weights):
        df['Training.Off_Target_Hazard'] += df[col] * weight

    df['Training.Off_Target_Hazard'] = np.sqrt(df['Training.Off_Target_Hazard'])

    # Piecewise linear using vectorized function
    df['Training.LFC'] = piecewise_linear_vectorized(
        df['Corrected_Log2_Fold_Change'].values,
        thresholds=[5, 2.5, 1, 0.5],
        scores=[0, 3, 5, 7, 10]
    )

    df['Training.Positivity'] = piecewise_linear_vectorized(
        df['Positive_Final_v2'].values,
        thresholds=[50, 35, 25, 15],
        scores=[0, 1, 3, 5, 10]
    )

    df['Training.2nd_Target_Val'] = piecewise_linear_vectorized(
        df['SC_2nd_Target_Val'].values,
        thresholds=[2, 1, 0.5, 0.1],
        scores=[0, 1, 3, 7, 10]
    )
    
    df['Training.On_Val_75'] = piecewise_linear_vectorized(
        df['On_Val_75'].values,
        thresholds=[2, 1, 0.5, 0.1],
        scores=[0, 1, 3, 7, 10]
    )
    
    df['Training.On_Val_50'] = piecewise_linear_vectorized(
        df['On_Val_50'].values,
        thresholds=[2, 1, 0.5, 0.1],
        scores=[0, 1, 3, 7, 10]
    )

    df['Training.SC_Hazard'] = piecewise_linear_vectorized(
        df['sc_hazard_risk'].values,
        thresholds=[100000, 50, 25, 10, 5],
        scores=[10, 10, 7, 3, 1, 0]
    )

    df['Training.FF_Hazard'] = piecewise_linear_vectorized(
        df['ff_hazard_risk'].values,
        thresholds=[100000, 50, 25, 10, 5],
        scores=[10, 10, 7, 3, 1, 0]
    )
    
    return df

#Risk Scores
sc_haz = pd.read_parquet("input/resources/SC_Single_Risk_Scores.20251017.parquet")
ff_haz = pd.read_parquet("input/resources/FFPE_Single_Risk_Scores.20251017.parquet")
sc_haz.columns = ["gene_name", "sc_hazard_risk"]
ff_haz.columns = ["gene_name", "ff_hazard_risk"]

sc_m_haz = pd.read_parquet("input/resources/SC_Multi_Risk_Scores.20251017.parquet")
ff_m_haz = pd.read_parquet("input/resources/FFPE_Multi_Risk_Scores.20251017.parquet")
sc_m_haz.columns = ["gene_name", "sc_hazard_risk"]
ff_m_haz.columns = ["gene_name", "ff_hazard_risk"]

# Create lookup dictionaries
sc_haz_dict = dict(zip(sc_m_haz["gene_name"], sc_m_haz["sc_hazard_risk"]))
ff_haz_dict = dict(zip(ff_m_haz["gene_name"], ff_m_haz["ff_hazard_risk"]))

surface = tid.utils.surface_genes()

######################################################################
# Create Single Target Training Data Sets
######################################################################

inds = {
    "AML_Single" : "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/AML/AML.Single.Results.20251029.parquet",
    "CRC_Single" : "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/CRC/CRC.Single.Results.20251029.parquet",
    "KIRC_Single" : "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/KIRC/KIRC.Single.Results.20251029.parquet",
    "LUAD_Single" : "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/LUAD.Magellan/LUAD.Magellan.Single.Results.20251029.parquet",
    "TNBC_Single" : "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/TNBC.Magellan/TNBC.Magellan.Single.Results.20251029.parquet",
    "PDAC_Single" : "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/PDAC_FFPE/PDAC_FFPE.Single.Results.20251029.parquet"    
}

df_ind = pd.read_parquet(inds["TNBC_Single"])
df_ind = df_ind.loc[df_ind["gene_name"].isin(surface)].copy()
df_ind = reduce(lambda left, right: pd.merge(left, right, how="left"), [df_ind, sc_haz, ff_haz])
df_ind = compute_target_quality_score_v2(df_ind)
df_ind.head(20)

inds = {
    "LUAD_Multi" : "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/LUAD.Magellan/LUAD.Magellan.Multi.Results.20251029.parquet",
    "TNBC_Multi" : "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/TNBC.Magellan/TNBC.Magellan.Multi.Results.20251029.parquet",
    "CRC_Multi" : "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/CRC/CRC.Multi.Results.20251029.parquet",
    "PDAC_Multi" : "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/PDAC_FFPE/PDAC_FFPE.Multi.Results.20251029.parquet"
}

df_ind = pd.read_parquet(inds["PDAC_Multi"])
hazard = pd.read_parquet("input/resources/Multi_Risk_Scores.20251017.sorted.parquet")
df_ind = pd.merge(df_ind, hazard, on = "gene_name", how = "left")
df_ind = compute_target_quality_score_v2(df_ind)
df_ind.head(20)

tabs = set(tid.utils.tabs_genes())
df_ind2 = df_ind[df_ind["gene_name"].str.split("_").apply(lambda x: any(g in tabs for g in x))].copy()
df_ind2.head(50)
df_ind2[df_ind2["gene_name"]=="ADAM8_MSLN"].iloc[0]


# df_ind = compute_target_quality_score_ns(df_ind)
# df_ind = reduce(lambda left, right: pd.merge(left, right, how="left"), [df_ind, sc_haz, ff_haz])

# df_ind["LFC_On_50_vs_Top"] = np.log2(df_ind["On_Val_50"] + 1) - np.log2(df_ind["Top_Off_Target_Val"] + 1)
# df_ind["Tox"] = np.max(df_ind[["Tox_Brain", "Tox_Heart", "Tox_Lung"]], axis = 1)
# df_ind["LFC_On_50_vs_Tox"] = np.log2(df_ind["On_Val_50"] + 1) - np.log2(df_ind["Tox"] + 1)
# df_ind["TI_Tox"] = np.min(df_ind[["TI_Brain", "TI_Heart", "TI_Lung"]], axis = 1)
# df_ind = df_ind[[col for col in df_ind.columns if col != 'gene_name'] + ['gene_name']]

# base_score = df_ind["TargetQ_Final_vNS"].copy()
# base_score.loc[(df_ind["Positive_Final_v2"] <= 1) | (df_ind["Target_Val"] <= 0.1)] = 0
# base_score += df_ind["Positive_Final_v2"] / 5
# base_score += 10 * np.minimum(df_ind["LFC_On_50_vs_Tox"], 1)
# base_score += 10 * df_ind["TI"]
# base_score += 10 * df_ind["TI_Tox"]
# base_score.loc[df_ind["TI_Tox"] < 0.1] -= 20
# base_score += 5 * np.minimum(df_ind["On_Val_75"], 2)
# base_score.loc[df_ind["sc_hazard_risk"] > 25] -= 5
# base_score.loc[df_ind["ff_hazard_risk"] > 25] -= 5
# base_score.loc[df_ind["TI_Tox"] == 0] -= 10
# base_score.loc[df_ind["N_Off_Targets_1.0"] > 5] -= 10
# df_ind["Training_Score"] = base_score
# df_ind = df_ind.sort_values("Training_Score", ascending=False)
# df_ind.head(20)

inds = {
    "LUAD_Multi" : "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/LUAD.Magellan/LUAD.Magellan.Multi.Results.20251029.parquet",
    "TNBC_Multi" : "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/TNBC.Magellan/TNBC.Magellan.Multi.Results.20251029.parquet",
    "CRC_Multi" : "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/CRC/CRC.Multi.Results.20251029.parquet",
    "PDAC_Multi" : "/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/PDAC_FFPE/PDAC_FFPE.Multi.Results.20251029.parquet"
}

df_ind = pd.read_parquet(inds["PDAC_Multi"])
df_ind = compute_target_quality_score_ns(df_ind)
df_ind["sc_hazard_risk"] = np.array([sc_haz_dict.get(name, np.nan) for name in df_ind["gene_name"]])
df_ind["ff_hazard_risk"] = np.array([ff_haz_dict.get(name, np.nan) for name in df_ind["gene_name"]])

df_ind["LFC_On_50_vs_Top"] = np.log2(df_ind["On_Val_50"] + 1) - np.log2(df_ind["Top_Off_Target_Val"] + 1)
df_ind["Tox"] = np.max(df_ind[["Tox_Brain", "Tox_Heart", "Tox_Lung"]], axis = 1)
df_ind["LFC_On_50_vs_Tox"] = np.log2(df_ind["On_Val_50"] + 1) - np.log2(df_ind["Tox"] + 1)
df_ind["TI_Tox"] = np.min(df_ind[["TI_Brain", "TI_Heart", "TI_Lung"]], axis = 1)
df_ind = df_ind[[col for col in df_ind.columns if col != 'gene_name'] + ['gene_name']]

base_score = df_ind["TargetQ_Final_vNS"].copy()
base_score.loc[(df_ind["Positive_Final_v2"] <= 1) | (df_ind["Target_Val"] <= 0.1)] = 0
base_score += df_ind["Positive_Final_v2"] / 5
base_score += 10 * np.minimum(df_ind["LFC_On_50_vs_Tox"], 1)
base_score += 10 * df_ind["TI"]
base_score += 10 * df_ind["TI_Tox"]
base_score.loc[df_ind["TI_Tox"] < 0.1] -= 20
base_score += 5 * np.minimum(df_ind["On_Val_75"], 2)
base_score.loc[df_ind["sc_hazard_risk"] > 25] -= 5
base_score.loc[df_ind["ff_hazard_risk"] > 25] -= 5
base_score.loc[df_ind["TI_Tox"] == 0] -= 10
base_score.loc[df_ind["N_Off_Targets_1.0"] > 5] -= 10
df_ind["Training_Score"] = base_score
df_ind = df_ind.sort_values("Training_Score", ascending=False)
df_ind.head(25)


df_ind2 = df_ind[~df_ind["gene_name"].str.contains("LY6G6D")]
df_ind2.head(25)


df_ind[df_ind["gene_name"]=="NECTIN4_PODXL2"].iloc[0]


top_multis = ["TMPRSS4_SLC22A31", "CEACAM5_HPN", "DPP4_MUC21", "TREM1_SMPDL3B", 
"TREM1_HS6ST2", "TMPRSS4_ROS1", "TREM1_PODXL2", "TREM1_SEZ6L2", 
"MUC21_HPN", "MUC21_SLC22A31", "FOLR1_MUC21", "CEACAM5_SLC22A31", 
"TREM1_MUC21", "CDH3_SLC22A31", "TREM1_GPR39", "FUT3_SLC22A31", 
"MUC4_SLC22A31", "MUC21_SMPDL3B", "CDH3_SMPDL3B", "TREM1_ST6GALNAC1", 
"TREM1_TMPRSS4", "EPCAM_MUC21", "ANGPTL4_ROS1", "EMB_MUC21", 
"ICAM1_MUC21", "GPC4_MUC21", "CEACAM5_ROS1", "CEACAM5_SCTR", 
"SMPDL3B_SLC22A31", "CEACAM1_ROS1", "TREM1_LSR", "SLC34A2_SMPDL3B", 
"NT5E_SLC22A31", "CEACAM5_TREM1", "CDH3_TREM1"]

top_multis_reversed = [
"SLC22A31_TMPRSS4", "HPN_CEACAM5", "MUC21_DPP4", "SMPDL3B_TREM1", 
"HS6ST2_TREM1", "ROS1_TMPRSS4", "PODXL2_TREM1", "SEZ6L2_TREM1", 
"HPN_MUC21", "SLC22A31_MUC21", "MUC21_FOLR1", "SLC22A31_CEACAM5", 
"MUC21_TREM1", "SLC22A31_CDH3", "GPR39_TREM1", "SLC22A31_FUT3", 
"SLC22A31_MUC4", "SMPDL3B_MUC21", "SMPDL3B_CDH3", "ST6GALNAC1_TREM1", 
"TMPRSS4_TREM1", "MUC21_EPCAM", "ROS1_ANGPTL4", "MUC21_EMB", 
"MUC21_ICAM1", "MUC21_GPC4", "ROS1_CEACAM5", "SCTR_CEACAM5", 
"SLC22A31_SMPDL3B", "ROS1_CEACAM1", "LSR_TREM1", "SMPDL3B_SLC34A2", 
"SLC22A31_NT5E", "TREM1_CEACAM5", "TREM1_CDH3"
]

df_ind[df_ind["gene_name"].isin(top_multis_reversed + top_multis)]

tabs = set(tid.utils.tabs_genes())
df_ind2 = df_ind[df_ind["gene_name"].str.split("_").apply(lambda x: any(g in tabs for g in x))].copy()
df_ind3 = df_ind2[~df_ind2["gene_name"].str.contains("PRAME|VTCN1")]
df_ind3.head(50)


df_ind2.head(20)
df_ind3 = df_ind2[(df_ind2["Training_Score"] > 100) & (df_ind2["Positive_Final_v2"] > 25)]
df_ind3.sort_values("LFC_On_50_vs_Tox", ascending=False)

df_ind[df_ind["gene_name"]=="LRP8_PRLR"].iloc[0]


pd.concat(
    df_ind[df_ind["gene_name"]=="DPP4_MUC21"].iloc[0],
    df_ind[df_ind["gene_name"]=="TMPRSS4_SLC22A31"].iloc[0],
    df_ind[df_ind["gene_name"]=="ABCC3_SLC22A31"].iloc[0]
)





df_ind[df_ind["gene_name"]=="MUC21"].iloc[0]


df_ind[df_ind["Training_Score"] > 50]


df_ind[df_ind["gene_name"]=="MUC21"].iloc[0]

df_ind[(df_ind["TargetQ_Final_vNS"] > 90) & (df_ind["sc_hazard_risk"] < 15) & (df_ind["ff_hazard_risk"] < 15) & (df_ind["TI_Tox"] > 0.25)]








#Get Risk Scores
sc_haz = pd.read_parquet("SC_Single_Risk_Scores.20251017.parquet")
ff_haz = pd.read_parquet("FFPE_Single_Risk_Scores.20251017.parquet")
sc_haz.columns = ["gene_name", "sc_hazard_risk"]
ff_haz.columns = ["gene_name", "ff_hazard_risk"]

#Load Custom Scores
df_a = pd.read_csv(files('py_target_id').joinpath(f'data/training/Target_ID_Scoring_20221218 - LAML.csv'))
df_k = pd.read_csv(files('py_target_id').joinpath(f'data/training/Target_ID_Scoring_20221218 - KIRC.csv'))
df_c = pd.read_csv(files('py_target_id').joinpath(f'data/training/Target_ID_Scoring_20221218 - COAD_INT.csv'))

df_a = df_a[["gene_name", "Rank_AVG"]]
df_k = df_k[["gene_name", "Rank_AVG"]]
df_c = df_c[["gene_name", "Rank_AVG"]]

df_a["Rank_AVG"] = pd.to_numeric(df_a["Rank_AVG"], errors='coerce')
df_k["Rank_AVG"] = pd.to_numeric(df_k["Rank_AVG"], errors='coerce')
df_c["Rank_AVG"] = pd.to_numeric(df_c["Rank_AVG"], errors='coerce')

df_a.columns = ["gene_name", "custom_score"]
df_k.columns = ["gene_name", "custom_score"]
df_c.columns = ["gene_name", "custom_score"]

#Get All Surface Genes
df_surface = pd.DataFrame({"gene_name" : tid.utils.surface_genes()})

#Add Info
df_a2 = reduce(lambda left, right: pd.merge(left, right, how="left"), [df_surface, df_a, sc_haz, ff_haz])
df_k2 = reduce(lambda left, right: pd.merge(left, right, how="left"), [df_surface, df_k, sc_haz, ff_haz])
df_c2 = reduce(lambda left, right: pd.merge(left, right, how="left"), [df_surface, df_c, sc_haz, ff_haz])

#Read Single Results
AML = pd.read_parquet("/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/AML/AML.Single.Results.20251027.parquet")
KIRC = pd.read_parquet("/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/KIRC/KIRC.Single.Results.20251027.parquet")
CRC = pd.read_parquet("/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/CRC/CRC.Single.Results.20251027.parquet")

#Subset To Match
AML = AML[AML["gene_name"].isin(df_surface["gene_name"])]
KIRC = KIRC[KIRC["gene_name"].isin(df_surface["gene_name"])]
CRC = CRC[CRC["gene_name"].isin(df_surface["gene_name"])]

#The Goal Is To Score The Ones
df_a3 = pd.merge(AML, df_a2, how = "left")
df_k3 = pd.merge(KIRC, df_k2, how = "left")
df_c3 = pd.merge(CRC, df_c2, how = "left")

def calculate_custom_score(row):
    v = row['custom_score']
    if pd.isna(v):
        v = 4
        if row['TargetQ_Final_v1'] <= 60:
            v += 0.5
        if row['TargetQ_Final_v1'] <= 40:
            v += 0.5
        if row['TargetQ_Final_v1'] <= 20:
            v += 1
        if row['Positive_Final_v2'] <= 10:
            v += 1
        if row['sc_hazard_risk'] >= 10:
            v += 0.25
        if row['sc_hazard_risk'] >= 25:
            v += 0.75
        if row['ff_hazard_risk'] >= 10:
            v += 0.25
        if row['ff_hazard_risk'] >= 25:
            v += 0.75
    return v

df_a3['custom_score'] = df_a3.apply(calculate_custom_score, axis=1)
df_k3['custom_score'] = df_k3.apply(calculate_custom_score, axis=1)
df_c3['custom_score'] = df_c3.apply(calculate_custom_score, axis=1)

#Custom Edit
df_a3.loc[df_a3["gene_name"] == "PRSS21", "custom_score"] = 0
df_c3.loc[df_c3["gene_name"] == "LY6G6D", "custom_score"] = 0

# Get Info
df_a3[["gene_name", "custom_score"]].to_csv("AML_Custom_Scores.20251028.v2.csv")
df_k3[["gene_name", "custom_score"]].to_csv("KIRC_Custom_Scores.20251028.v2.csv")
df_c3[["gene_name", "custom_score"]].to_csv("CRC_Custom_Scores.20251028.v2.csv")
