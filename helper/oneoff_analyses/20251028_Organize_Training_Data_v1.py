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

#Create Training Data Sets

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
