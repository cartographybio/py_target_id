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

#Train On The Whole Thing
df = pd.concat([df_aml, df_crc, df_kirc])

####################################################################################################################################################
####################################################################################################################################################

#Run On
# Prepare features and target
X = df.copy().drop(['custom_score', 'gene_name'], axis=1)
X = X[[
	"Corrected_Log2_Fold_Change",
	"Corrected_Specificity", "N_Off_Targets", 
	"N_Off_Targets_0.1", "SC_2nd_Target_Val", 
	"On_Val_50", "On_Val_75",
	"sc_hazard_risk", "ff_hazard_risk"
	]]

y = df.copy()['custom_score']

# Keep only numeric columns
X = X.select_dtypes(include=['number'])

# Handle missing values
X = X.fillna(X.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# ============================================================================
# 3. SCALE FEATURES
# ============================================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled = scaler.transform(X)

# ============================================================================
# 4. STAGE 1: COARSE GRID SEARCH (Find good regions)
# ============================================================================
print("\n" + "="*80)
print("STAGE 1: Coarse Grid Search")
print("="*80)

model = XGBRegressor(
    tree_method='hist',
    device='cuda',
    random_state=1
)

# Coarse grid - larger ranges to find good regions
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
# 5. STAGE 2: FINE GRID SEARCH (Refine around best region)
# ============================================================================
print("\n" + "="*80)
print("STAGE 2: Fine Grid Search (around best parameters)")
print("="*80)

# Create fine grid centered on coarse best parameters
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

best_model = fine_search.best_estimator_

# ============================================================================
# 6. EVALUATE FINAL MODEL
# ============================================================================
y_pred = best_model.predict(X_test_scaled)

correlation = np.corrcoef(y_test, y_pred)[0, 1]
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2))

print(f"\nFinal Test Set Performance:")
print(f"  Correlation: {correlation:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R² Score: {r2:.4f}")

df["predicted"] = best_model.predict(X_scaled)  # All data scaled
df.sort_values("predicted")[0:20]

ind = pd.read_parquet("/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/LUAD.Magellan/LUAD.Magellan.Single.Results.20251027.parquet")
ind = ind[ind["gene_name"].isin(df["gene_name"])]
df_ind = reduce(lambda left, right: pd.merge(left, right, how="left"), [ind, sc_haz, ff_haz])
I = df_ind[X.columns]
I_scaled = scaler.transform(I)
df_ind["predicted"] = best_model.predict(I_scaled)  # All data scaled
df_ind[["gene_name", "predicted", "TargetQ_Final_v1", "Positive_Final_v2"]].sort_values("predicted")[0:20]

df_ind.loc[df_ind["gene_name"]=="SMPDL3B"].iloc[0]


importance = pd.DataFrame({
    'feature': X_train.columns,  # Use original DataFrame columns
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance)





K = df_kirc[X.columns]
K_scaled = scaler.transform(K)
df_kirc["predicted"] = best_model.predict(K_scaled)  # All data scaled
df_kirc.sort_values("predicted")[0:10]





KIRC["Predicted"] = y_pred_full
KIRC.sort_values("Predicted")[["gene_name"]]

N_scaled = scaler.transform(CRC[X.columns])  # Use transform, not fit_transform
n_pred_full = best_model.predict(N_scaled)  # All data scaled
CRC["Predicted"] = n_pred_full
CRC.sort_values("Predicted")[["gene_name"]]


CRC.sort_values("Predicted").iloc[0]
