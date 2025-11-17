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

inds = {
    "AML_Single" :     

}


AML = pd.read_parquet("/home/jgranja_cartography_bio/data/Custom_Analysis/20251008_Manifest_Analysis/AML/AML.Single.Results.20251027.parquet")





























































































































