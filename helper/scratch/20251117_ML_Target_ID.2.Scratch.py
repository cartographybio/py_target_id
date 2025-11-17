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

#Inputs
df_aml = pd.read_csv("AML_Custom_Scores.20251028.v2.csv", index_col=0)
df_crc = pd.read_csv("CRC_Custom_Scores.20251028.v2.csv", index_col=0)
df_kirc = pd.read_csv("KIRC_Custom_Scores.20251028.v2.csv", index_col=0)

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

df_aml = add_training_metrics_vectorized(df_aml)
df_crc = add_training_metrics_vectorized(df_crc)
df_kirc = add_training_metrics_vectorized(df_kirc)

model_id = "model_v1"
os.makedirs("models/" + model_id, exist_ok=True)

####################################################################################################################################################
# TEST MODEL HERE
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

# Select columns with "Training." in the name
feature_cols = [col for col in X_train.columns if 'Training.' in col]
X_train = X_train[feature_cols]
X_test = X_test[feature_cols]

# Handle missing values (use train mean for both)
train_mean = X_train.mean()
X_train = X_train.fillna(train_mean)
X_test = X_test.fillna(train_mean)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Features: {feature_cols}")

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

best_model = fine_search.best_estimator_

# ============================================================================
# EVALUATE TEST SET
# ============================================================================
y_pred = best_model.predict(X_test_scaled)

correlation = np.corrcoef(y_test, y_pred)[0, 1]
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2))

print(f"\nCRC Test Set Performance:")
print(f"  Correlation: {correlation:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R² Score: {r2:.4f}")

# Check bottom 10 targets
actual_bottom_indices = np.argsort(y_test.values)[:10]
pred_bottom_indices = np.argsort(y_pred)[:10]
overlap = len(set(actual_bottom_indices) & set(pred_bottom_indices))
print(f"  Overlap in bottom 10: {overlap}/10")

importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance)

####################################################################################################################################################
# END MODEL HERE
####################################################################################################################################################

# ============================================================================
# Can We Save these as ranked ordered list
# ============================================================================

inds = ["AML", "KIRC", "CRC", "LUAD.Magellan", "TNBC.Magellan"]

for ind in inds:
	df_ind = pd.read_parquet("/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/" + ind + "/" + ind + ".Single.Results.20251027.parquet")
	df_ind = df_ind[df_ind["gene_name"].isin(df_aml["gene_name"])]
	df_ind = reduce(lambda left, right: pd.merge(left, right, how="left"), [df_ind, sc_haz, ff_haz])
	df_ind = add_training_metrics_vectorized(df_ind)
	I = df_ind[X_train.columns]
	I_scaled = scaler.transform(I)
	df_ind["predicted"] = best_model.predict(I_scaled)  # All data scaled
	df_ind = df_ind.loc[df_ind["Positive_Final_v2"] > 15]
	print(ind)
	print(df_ind[["gene_name", "predicted", "TargetQ_Final_v1", "Positive_Final_v2"]].sort_values("predicted")[0:20])


inds = ["LUAD.Magellan", "TNBC.Magellan"]

for ind in inds:
	df_ind = pd.read_parquet("/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/" + ind + "/" + ind + ".Single.Results.20251027.parquet")
	df_ind = df_ind[df_ind["gene_name"].isin(df_aml["gene_name"])]
	df_ind = reduce(lambda left, right: pd.merge(left, right, how="left"), [df_ind, sc_haz, ff_haz])
    df_ind = add_training_metrics_vectorized(df_ind)
	I = df_ind[X_train.columns]
	I_scaled = scaler.transform(I)
	df_ind["predicted"] = best_model.predict(I_scaled)  # All data scaled
	df_ind = df_ind.loc[df_ind["Positive_Final_v2"] > 15]
	print(ind)
	print(df_ind[["gene_name", "predicted", "TargetQ_Final_v1", "Positive_Final_v2"]].sort_values("predicted")[0:20])

