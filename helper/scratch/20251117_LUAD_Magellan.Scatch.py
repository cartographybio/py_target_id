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

#Read
multi = pd.read_parquet('LUAD.Magellan.Multi.Results.20251104.parquet')

#Risk
risk = tid.utils.get_multi_risk_scores()

#Join
multi = pd.merge(multi, risk, on = "gene_name", how = "left")

#TQ
multi = tid.run.target_quality_v2_01(multi)

tabs = tid.utils.tabs_genes()

#Split Out Singles
split = multi["gene_name"].str.split("_", expand=True)
mask_single = split[0] == split[1]
single = multi[mask_single].copy()
single["gene_name"] = single["gene_name"].str.split("_").str[0]

#Improvement Scoring
multi2 = multi[(multi["Positive_Final_v2"] > 25) & (multi["TargetQ_Final_v2"] > 60)].copy().reset_index()
multi2 = pd.concat([multi2, multi2["gene_name"].str.split("_", expand=True)], axis=1)
multi2["TABS"] = 1*multi2[0].isin(tabs) + 1*multi2[1].isin(tabs)

#Multi Score
single0 = single.copy().set_index("gene_name").loc[multi2[0]].reset_index()
single1 = single.copy().set_index("gene_name").loc[multi2[1]].reset_index()

# Score_1: Expression Comparison
lfc_on_xy_50 = abs(np.log2(single0["On_Val_50"]+1) - np.log2(single1["On_Val_50"]+1))
Score_1 = np.select(
    [lfc_on_xy_50 < 0.5, 
     (lfc_on_xy_50 >= 0.5) & (lfc_on_xy_50 < 1),
     (lfc_on_xy_50 >= 1) & (lfc_on_xy_50 < 2)],
    [10, 8, 5],
    default=0
)
Score_1 = np.clip(Score_1, 0, 10)

# Score_2: Target Quality Improvement
diff_TQ = multi2["TargetQ_Final_v2"] - np.maximum(single0["TargetQ_Final_v2"], single1["TargetQ_Final_v2"])
Score_2 = np.minimum(diff_TQ, 50)/5
Score_2 = np.clip(Score_2, 0, 10)

# Score_3: Patients Lost
sngl_patients = np.maximum(single0["Positive_Final_v2"], single1["Positive_Final_v2"])
paitents_lost = (sngl_patients - multi2["Positive_Final_v2"]) / sngl_patients
Score_3 = 10 * (1 - paitents_lost)
Score_3 = np.clip(Score_3, 0, 10)

# Score 4: TI gain
sngl_TI = np.maximum(single0["TI"], single1["TI"])
Score_4 = 20 * (multi2["TI"] - sngl_TI)
Score_4 = np.clip(Score_4, 0, 10)

main_score = 1*(lfc_on_xy_50 < 1)
sub_score = Score_1 + Score_2 + Score_3 + Score_4
sub_score = sub_score / 40
multi2["TIS"] = main_score + sub_score

multi2[(multi2["TIS"] > 0.8) & (multi2["TABS"] == 1) & (multi2["Positive_Final_v2"] > 40)].head(50)[["Positive_Final_v2","TargetQ_Final_v2","TIS",0,1]]
multi2[(multi2["TIS"] > 0.8) & (multi2["TABS"] == 0) & (multi2["Positive_Final_v2"] > 40)].head(50)[["Positive_Final_v2","TargetQ_Final_v2","TIS",0,1]]
multi2[(multi2["TIS"] > 0.8) & (multi2["TABS"] == 2) & (multi2["Positive_Final_v2"] > 40)].head(50)[["Positive_Final_v2","TargetQ_Final_v2","TIS",0,1]]

multi2[multi2["gene_name"].isin(["CEACAM5_HPN"])][["Positive_Final_v2","TargetQ_Final_v2","TIS",0,1]]
which(multi2["gene_name"] == "EPCAM_MUC21")

lfc_on_xy_50[29]

total_cutoff = 35 * (lfc_on_xy_50 < 1) + 35 * (diff_TQ > 5)
total_sub = Score_1 + Score_2 + Score_3 + Score_4
total_sub = 30 * total_sub / 40
TIS = total_cutoff + total_sub




multi2[TIS > 70]





TIS = 100 * (50 * (lfc_on_xy_50 < 1) + 50 * (diff_TQ > 5) + (Score_1 + Score_2 + Score_3)) / 130
multi2[TIS > 75].head(25)



100 * (2 * Score_1 + Score_3 + Score_4) / 40

multi2[Score_1==10].head(20)



































