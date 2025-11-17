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
from functools import reduce

#Inputs
df_aml = pd.read_csv("AML_Custom_Scores.20251028.csv", index_col=0)
df_crc = pd.read_csv("CRC_Custom_Scores.20251028.csv", index_col=0)
df_kirc = pd.read_csv("KIRC_Custom_Scores.20251028.csv", index_col=0)

AML = pd.read_parquet("/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/AML/AML.Single.Results.20251027.parquet")
CRC = pd.read_parquet("/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/CRC/CRC.Single.Results.20251027.parquet")
KIRC = pd.read_parquet("/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/KIRC/KIRC.Single.Results.20251027.parquet")

#Get Risk Scores
sc_haz = pd.read_parquet("SC_Single_Risk_Scores.20251017.parquet")
ff_haz = pd.read_parquet("FFPE_Single_Risk_Scores.20251017.parquet")
sc_haz.columns = ["gene_name", "sc_hazard_risk"]
ff_haz.columns = ["gene_name", "ff_hazard_risk"]

#Join Data
df_aml = reduce(lambda left, right: pd.merge(left, right, how="left"), [df_aml, AML, sc_haz, ff_haz])
df_crc = reduce(lambda left, right: pd.merge(left, right, how="left"), [df_crc, CRC, sc_haz, ff_haz])
df_kirc = reduce(lambda left, right: pd.merge(left, right, how="left"), [df_kirc, KIRC, sc_haz, ff_haz])

df_aml["Specificity_Combo"] = df_aml["N_Off_Targets"] + (1 - df_aml["Corrected_Specificity"])
df_crc["Specificity_Combo"] = df_crc["N_Off_Targets"] + (1 - df_crc["Corrected_Specificity"])
df_kirc["Specificity_Combo"] = df_kirc["N_Off_Targets"] + (1 - df_kirc["Corrected_Specificity"])

df_aml["SC_2nd_Target_Val"] = np.minimum(df_aml["SC_2nd_Target_Val"], 1.25)
df_crc["SC_2nd_Target_Val"] = np.minimum(df_crc["SC_2nd_Target_Val"], 1.25)
df_kirc["SC_2nd_Target_Val"] = np.minimum(df_kirc["SC_2nd_Target_Val"], 1.25)

####################################################################################################################################################
####################################################################################################################################################

# ============================================================================
# PREPARE DATA (Train on AML + KIRC, Test on CRC)
# ============================================================================
train = [df_aml, df_crc]
test = df_kirc

X_train = pd.concat(train).drop(['custom_score', 'gene_name'], axis=1)
y_train = pd.concat(train)['custom_score']

X_test = test.drop(['custom_score', 'gene_name'], axis=1)
y_test = test['custom_score']

# Select specific columns
feature_cols = [
    "Specificity_Combo",
    "N_Off_Targets_1.0", "SC_2nd_Target_Val", 
    "On_Val_50", "On_Val_75",
    "sc_hazard_risk", "ff_hazard_risk"
]
X_train = X_train[feature_cols]
X_test = X_test[feature_cols]

# Handle missing values (use train mean for both)
train_mean = X_train.mean()
X_train = X_train.fillna(train_mean)
X_test = X_test.fillna(train_mean)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# ============================================================================
# SCALE FEATURES
# ============================================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# STAGE 1: COARSE GRID SEARCH
# ============================================================================
print("\n" + "="*80)
print("STAGE 1: Coarse Grid Search")
print("="*80)

model = XGBRegressor(
    tree_method='hist',
    device='cuda',
    random_state=1
)

coarse_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.5, 0.8],
    'colsample_bytree': [0.5, 0.8]
}

coarse_search = GridSearchCV(
    estimator=model,
    param_grid=coarse_param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=2,
    n_jobs=1
)

print("Starting Coarse GridSearchCV...")
coarse_search.fit(X_train_scaled, y_train)

coarse_best_params = coarse_search.best_params_
coarse_best_score = coarse_search.best_score_

print(f"\nCoarse Search Best Parameters: {coarse_best_params}")
print(f"Coarse Search Best Score: {coarse_best_score:.6f}")

# ============================================================================
# STAGE 2: FINE GRID SEARCH
# ============================================================================
print("\n" + "="*80)
print("STAGE 2: Fine Grid Search (around best parameters)")
print("="*80)

fine_param_grid = {
    'n_estimators': [coarse_best_params['n_estimators'] - 50, 
                     coarse_best_params['n_estimators'], 
                     coarse_best_params['n_estimators'] + 50],
    'max_depth': [max(1, coarse_best_params['max_depth'] - 2),
                  coarse_best_params['max_depth'],
                  coarse_best_params['max_depth'] + 2],
    'learning_rate': [coarse_best_params['learning_rate'] * 0.5,
                      coarse_best_params['learning_rate'],
                      coarse_best_params['learning_rate'] * 1.5],
    'subsample': [max(0.1, coarse_best_params['subsample'] - 0.15),
                  coarse_best_params['subsample'],
                  min(1.0, coarse_best_params['subsample'] + 0.15)],
    'colsample_bytree': [max(0.1, coarse_best_params['colsample_bytree'] - 0.15),
                         coarse_best_params['colsample_bytree'],
                         min(1.0, coarse_best_params['colsample_bytree'] + 0.15)]
}

print(f"Fine parameter grid:")
for key, val in fine_param_grid.items():
    print(f"  {key}: {val}")

fine_search = GridSearchCV(
    estimator=model,
    param_grid=fine_param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=2,
    n_jobs=1
)

print("\nStarting Fine GridSearchCV...")
fine_search.fit(X_train_scaled, y_train)

best_params = fine_search.best_params_
print("\n" + "="*80)
print(f"Final Best Parameters: {best_params}")
print("="*80)

# Create sample weights - penalize errors on low scores more
# Lower scores get higher weights
sample_weights = 1 / (y_train + 0.1)  # +0.1 to avoid division by zero

# Normalize weights so they sum to training set size
sample_weights = sample_weights / sample_weights.sum() * len(y_train)

print(f"\nSample Weight Statistics:")
print(f"  Min: {sample_weights.min():.4f}, Max: {sample_weights.max():.4f}")
print(f"  Mean: {sample_weights.mean():.4f}")

# Retrain best model with sample weights
best_model = XGBRegressor(
    tree_method='hist',
    device='cuda',
    random_state=1,
    **best_params
)

print("\nTraining best model with sample weights...")
best_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)

# ============================================================================
# EVALUATE ON KIRC TEST SET
# ============================================================================
y_pred = best_model.predict(X_test_scaled)

correlation = np.corrcoef(y_test, y_pred)[0, 1]
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2))

print(f"\nKIRC Test Set Performance:")
print(f"  Correlation: {correlation:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R² Score: {r2:.4f}")

# Check bottom 10 targets
actual_bottom_indices = np.argsort(y_test.values)[:10]
pred_bottom_indices = np.argsort(y_pred)[:10]
overlap = len(set(actual_bottom_indices) & set(pred_bottom_indices))
print(f"  Overlap in bottom 10: {overlap}/10")

# Feature importance
importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nFeature Importance:")
print(importance)



inds = ["AML", "KIRC", "CRC"]

for ind in inds:
	df_ind = pd.read_parquet("/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/" + ind + "/" + ind + ".Single.Results.20251027.parquet")
	df_ind = df_ind[df_ind["gene_name"].isin(df["gene_name"])]
	df_ind = reduce(lambda left, right: pd.merge(left, right, how="left"), [df_ind, sc_haz, ff_haz])
	df_ind["Specificity_Combo"] = df_ind["N_Off_Targets"] + (1 - df_ind["Corrected_Specificity"])
    df_ind["SC_2nd_Target_Val"] = np.minimum(df_ind["SC_2nd_Target_Val"], 1.25)
	I = df_ind[X_train.columns]
	I_scaled = scaler.transform(I)
	df_ind["predicted"] = best_model.predict(I_scaled)  # All data scaled
	df_ind = df_ind.loc[df_ind["Positive_Final_v2"] > 15]
	print(ind)
	print(df_ind[["gene_name", "predicted", "TargetQ_Final_v1", "Positive_Final_v2"]].sort_values("predicted")[0:20])


inds = ["LUAD.Magellan", "TNBC.Magellan"]

for ind in inds:
	df_ind = pd.read_parquet("/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/" + ind + "/" + ind + ".Single.Results.20251027.parquet")
	df_ind = df_ind[df_ind["gene_name"].isin(df["gene_name"])]
	df_ind = reduce(lambda left, right: pd.merge(left, right, how="left"), [df_ind, sc_haz, ff_haz])
	df_ind["Specificity_Combo"] = df_ind["N_Off_Targets"] + (1 - df_ind["Corrected_Specificity"])
    df_ind["SC_2nd_Target_Val"] = np.minimum(df_ind["SC_2nd_Target_Val"], 1.25)
	I = df_ind[X_train.columns]
	I_scaled = scaler.transform(I)
	df_ind["predicted"] = best_model.predict(I_scaled)  # All data scaled
	df_ind = df_ind.loc[df_ind["Positive_Final_v2"] > 15]
	print(ind)
	print(df_ind[["gene_name", "predicted", "TargetQ_Final_v1", "Positive_Final_v2"]].sort_values("predicted")[0:20])


sc_m_haz = pd.read_parquet("SC_Multi_Risk_Scores.20251017.parquet")
ff_m_haz = pd.read_parquet("FFPE_Multi_Risk_Scores.20251017.parquet")
sc_m_haz.columns = ["gene_name", "sc_hazard_risk"]
ff_m_haz.columns = ["gene_name", "ff_hazard_risk"]

inds = ["TNBC.Magellan"]

for ind in inds:
    df_ind = pd.read_parquet("/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/" + ind + "/" + ind + ".Multi.Results.20251027.parquet")
    df_ind = reduce(lambda left, right: pd.merge(left, right, how="left"), [df_ind, sc_m_haz, ff_m_haz])
    df_ind["Specificity_Combo"] = df_ind["N_Off_Targets"] + (1 - df_ind["Corrected_Specificity"])
    df_ind["SC_2nd_Target_Val"] = np.minimum(df_ind["SC_2nd_Target_Val"], 1.25)
    I = df_ind[X_train.columns]
    I_scaled = scaler.transform(I)
    df_ind["predicted"] = best_model.predict(I_scaled)  # All data scaled
    df_ind = df_ind.loc[df_ind["Positive_Final_v2"] > 50]
    df_ind = df_ind[~df_ind["gene_name"].str.contains("PRAME|LY6K|VTCN1|PCDHB2|CALML5")]
    print(ind)
    print(df_ind[["gene_name", "predicted", "TargetQ_Final_v1", "Positive_Final_v2"]].sort_values("predicted")[0:50])

df_ind["gene_name"]

tabs = set(tid.utils.tabs_genes())
df_ind2 = df_ind[df_ind["gene_name"].str.split("_").apply(lambda x: any(g in tabs for g in x))].copy()
df_ind2["combo"] = (100 * (1 - np.maximum(np.minimum(df_ind2["predicted"],7),0) / 7)) + df_ind2["TargetQ_Final_v1"]
print(df_ind2[["gene_name", "combo", "predicted", "TargetQ_Final_v1", "Positive_Final_v2"]].sort_values("combo", ascending = False)[0:50])



df_ind.loc[df_ind["gene_name"]=="NECTIN4_PODXL2"].iloc[0]
df_ind.loc[df_ind["gene_name"]=="HIST1H1A_KRT8"].iloc[0]


pd.concat([
    df_ind.loc[df_ind["gene_name"]=="NECTIN4_PODXL2"].iloc[0],
    df_ind.loc[df_ind["gene_name"]=="NECTIN4_SYT12"].iloc[0]
], axis=1)



