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

#Create Training Data Sets

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
df_a2 = pd.merge(df_surface, df_a, how = "left")
df_k2 = pd.merge(df_surface, df_k, how = "left")
df_c2 = pd.merge(df_surface, df_c, how = "left")

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
            v += 1
        if row['TargetQ_Final_v1'] <= 40:
            v += 0.5
        if row['TargetQ_Final_v1'] <= 20:
            v += 0.5
        if row['Positive_Final_v2'] <= 5:
            v += 1
    return v

df_a3['custom_score'] = df_a3.apply(calculate_custom_score, axis=1)
df_k3['custom_score'] = df_k3.apply(calculate_custom_score, axis=1)
df_c3['custom_score'] = df_c3.apply(calculate_custom_score, axis=1)

#Custom Edit
df_a3.loc[df_a3["gene_name"] == "PRSS21", "custom_score"] = 0
df_c3.loc[df_c3["gene_name"] == "LY6G6D", "custom_score"] = 0

# Get Info
df_a3[["gene_name", "custom_score"]].to_csv("AML_Custom_Scores.20251028.csv")
df_k3[["gene_name", "custom_score"]].to_csv("KIRC_Custom_Scores.20251028.csv")
df_c3[["gene_name", "custom_score"]].to_csv("CRC_Custom_Scores.20251028.csv")






df_k3["custom_score"] <- lapply(seq(nrow(df_k3)), function(x){
	
	v <- df_k3[x, "custom_score"]

	if(is.na(v)){
		
		v <- 4
		
		if(df_k3["TargetQ_Final_v1"] <= 60){
			v <- v + 1
		}
		if(df_k3["TargetQ_Final_v1"] <= 40){
			v <- v + 0.5
		}
		if(df_k3["TargetQ_Final_v1"] <= 20){
			v <- v + 0.5
		}

		if(df_k3["Positive_Final_v2"] <= 5){
			v <- v + 1
		}

	}

	v

}) %>% unlist



df_k3["custom_score"]



KIRC[["custom_score"]] = KIRC[["custom_score"]].fillna(5)
CRC[["custom_score"]] = CRC[["custom_score"]].fillna(5)



#Train On
KIRC = pd.read_parquet("/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/KIRC/KIRC.Single.Results.20251027.parquet")
CRC = pd.read_parquet("/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/CRC/CRC.Single.Results.20251027.parquet")

#Load Custom Scores
df_k = pd.read_csv(files('py_target_id').joinpath(f'data/training/Target_ID_Scoring_20221218 - KIRC.csv'))
df_c = pd.read_csv(files('py_target_id').joinpath(f'data/training/Target_ID_Scoring_20221218 - COAD_INT.csv'))

df_k = df_k[["gene_name", "Rank_AVG"]]
df_c = df_c[["gene_name", "Rank_AVG"]]
df_k["Rank_AVG"] = pd.to_numeric(df_k["Rank_AVG"], errors='coerce')
df_c["Rank_AVG"] = pd.to_numeric(df_c["Rank_AVG"], errors='coerce')

df_k = df_k.loc[df_k["Rank_AVG"] <= 5]
df_c = df_c.loc[df_c["Rank_AVG"] <= 5]
df_k.columns = ["gene_name", "custom_score"]
df_c.columns = ["gene_name", "custom_score"]

#Surface
surface = tid.utils.surface_genes(tiers = [1,2,3,4], as_df = True)
surface.loc[surface["tier"] == "TABS", "tier"] = 2
surface["tier"] = surface["tier"].astype(int)

#Re-Shape Scoring
KIRC = pd.merge(KIRC, df_k, how = "left")
CRC = pd.merge(CRC, df_c, how = "left")

KIRC = pd.merge(KIRC, surface, how = "left")
CRC = pd.merge(CRC, surface, how = "left")

KIRC[["custom_score"]] = KIRC[["custom_score"]].fillna(5)
CRC[["custom_score"]] = CRC[["custom_score"]].fillna(5)

KIRC[["tier"]] = KIRC[["tier"]].fillna(5)
CRC[["tier"]] = CRC[["tier"]].fillna(5)

#Run On
# Prepare features and target
X = KIRC.copy().drop(['custom_score', 'gene_name'], axis=1)
X = X[[
	"Corrected_Log2_Fold_Change",
	"Corrected_Specificity", "N_Off_Targets", 
	"N_Off_Targets_0.1", "SC_2nd_Target_Val", 
	"Positive_Final_0.1", "Positive_Final_v2", "tier", "On_Val_75", "On_Val_50"
	]]

y = KIRC.copy()['custom_score']

# Keep only numeric columns
X = X.select_dtypes(include=['number'])

# Handle missing values
X = X.fillna(X.mean())

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

X_scaled = scaler.transform(X)  # Use transform, not fit_transform
y_pred_full = best_model.predict(X_scaled)  # All data scaled

KIRC["Predicted"] = y_pred_full
KIRC.sort_values("Predicted")[["gene_name"]]

N_scaled = scaler.transform(CRC[X.columns])  # Use transform, not fit_transform
n_pred_full = best_model.predict(N_scaled)  # All data scaled
CRC["Predicted"] = n_pred_full
CRC.sort_values("Predicted")[["gene_name"]]


CRC.sort_values("Predicted").iloc[0]

































X = X.select_dtypes(include=['number'])
X = X.fillna(X.mean())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# XGBoost with GPU acceleration (XGBoost 2.0+)
model = XGBRegressor(
    tree_method='hist',           # Updated for 2.0+
    device='cuda',                # Use CUDA instead of gpu_id
    random_state=1,
    n_jobs=-1
)

# Smaller, smarter grid
param_grid = {
    'n_estimators': [80, 100, 120, 150],
    'max_depth': [4, 5, 6],
    'learning_rate': [0.08, 0.1, 0.12],
    'subsample': [0.85, 0.9, 0.95],
    'colsample_bytree': [0.6, 0.7, 0.8]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=2,  # This will give you updates
    n_jobs=1
)

grid_search.fit(X_train_scaled, y_train)

best_params = grid_search.best_params_
print("Best Parameters:", best_params)

best_model = grid_search.best_estimator_

# Convert test data to GPU for prediction
import xgboost as xgb
X_test_dmatrix = xgb.DMatrix(X_test_scaled)
y_pred = best_model.predict(X_test_scaled)

correlation = np.corrcoef(y_test, y_pred)[0, 1]
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2))

print(f"Correlation: {correlation:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

X_scaled = scaler.transform(X)  # Use transform, not fit_transform
y_pred_full = best_model.predict(X_scaled)  # All data scaled

KIRC["Predicted"] = y_pred_full
KIRC.sort_values("Predicted")[["gene_name"]]

N_scaled = scaler.transform(CRC[X.columns])  # Use transform, not fit_transform
n_pred_full = best_model.predict(N_scaled)  # All data scaled
CRC["Predicted"] = n_pred_full
CRC.sort_values("Predicted")[["gene_name"]]






