"""
compute_target_quality_v2
"""

__all__ = ['target_quality_v2_01']

import numpy as np
import pandas as pd

def target_quality_v2_01(  # NO SURFACE ASSUME ALL ARE SURFACE
    df: pd.DataFrame
) -> pd.DataFrame:
    
    # Check for required columns
    required_cols = ['Hazard_SC_v1', 'Hazard_FFPE_v1', 'Hazard_GTEX_v1']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Score_1: Off-target penalty (1.0 threshold)
    Score_1 = np.select(
        [df['N_Off_Targets'] <= 3, df['N_Off_Targets'] > 3],
        [df['N_Off_Targets'], 6 + (df['N_Off_Targets'] - 3) * 2], #New Change
        default=10
    )
    Score_1 = np.minimum(Score_1, 10)

    # Score_2: Off-target penalty (0.5 threshold)
    Score_2 = np.select(
        [df['N_Off_Targets_0.5'] <= 3, df['N_Off_Targets_0.5'] > 3],
        [df['N_Off_Targets_0.5'], 6 + (df['N_Off_Targets_0.5'] - 3) * 2], #New Change
        default=10
    )
    Score_2 = np.minimum(Score_2, 10)

    # Score_3: Corrected Specificity
    Score_3 = np.select(
        [df['Corrected_Specificity'] >= 0.75,
         (df['Corrected_Specificity'] >= 0.5) & (df['Corrected_Specificity'] < 0.75),
         (df['Corrected_Specificity'] >= 0.35) & (df['Corrected_Specificity'] < 0.5)],
        [0, 1, 3],
        default=10
    )

    # Score_4: Percent of Patients Above Val 0.5 Should Read as P_Pos_Val
    Score_4 = np.select(
        [df['P_Pos_Per'] > 0.25,
         (df['P_Pos_Per'] > 0.15) & (df['P_Pos_Per'] <= 0.25),
         (df['P_Pos_Per'] > 0.025) & (df['P_Pos_Per'] <= 0.15)],
        [0, 1, 3],
        default=10
    )

    # Penalty if Cohort is > 10
    if df["N"].values[0] > 10:
        Score_4 = np.where(df['N_Pos_Val'] == 1, 10, Score_4)

    # Score_5: Proportion of Patients with Specificity > 0.35
    off_set = np.where(df['N_Off_Targets'] <= 5, 0.1, 0)
    Score_5 = np.select(
        [df['P_Pos'] + off_set > 0.25,
         (df['P_Pos'] + off_set > 0.15) & (df['P_Pos'] + off_set <= 0.25),
         (df['P_Pos'] + off_set > 0.025) & (df['P_Pos'] + off_set <= 0.15)],
        [0, 1, 3],
        default=10
    )

    # Penalty if Cohort is > 10
    if df["N"].values[0] > 10:
        Score_5 = np.where(df['N_Pos'] == 1, 10, Score_5)

    # Score_6: 2nd Target expression
    Score_6 = np.select(
        [df['SC_2nd_Target_Val'] > 2,
         (df['SC_2nd_Target_Val'] > 1) & (df['SC_2nd_Target_Val'] <= 2),
         (df['SC_2nd_Target_Val'] > 0.5) & (df['SC_2nd_Target_Val'] <= 1),
         (df['SC_2nd_Target_Val'] > 0.1) & (df['SC_2nd_Target_Val'] <= 0.5)],
        [0, 1, 3, 5],
        default=10
    )
    Score_6 = Score_6 - np.maximum(np.minimum(df['SC_2nd_Target_Val'], 6) - 2, 0) * 2.5 # We want things High Exp

    # Score_7: Legacy baseline
    Score_7 = np.full(len(df), 0) # Assume All Surface

    # Compute final TargetQ score
    score_array = np.column_stack([Score_1, Score_2, Score_3, Score_5, Score_6, Score_7])
    penalty_array = np.column_stack([Score_1, Score_3])

    ###############
    # Target Quality V1 With New Adjustments
    ###############

    # Step 1: Sum all scores
    raw_scores0 = score_array.sum(axis=1)
    raw_scores = np.maximum(raw_scores0, 0) #New Scoring Has Some Negative Allowances

    # Step 2: Count penalties (Score == 10)
    penalty_count = (penalty_array == 10).sum(axis=1)

    # Step 3: Define constants
    penalty_weight = 0.1
    max_score = score_array.shape[1] * 10  # 6 scores × 10
    max_penalized_score = 1 + penalty_array.shape[1] * penalty_weight  # 1 + 3×0.1 = 1.3

    # Step 4: Calculate penalized score
    penalized_scores = raw_scores / max_score + penalty_weight * penalty_count

    # Step 5: Scale to 0-100
    target_quality = 100 - (100 / max_penalized_score) * penalized_scores
    
    ###############
    # Target Quality V2
    ###############

    # Calculate derived values on-the-fly
    LFC_On_50_vs_Tox = np.log2(df["On_Val_50"].values + 1) - np.log2(
        np.maximum(np.maximum(df["Tox_Brain"].values, df["Tox_Heart"].values), df["Tox_Lung"].values) + 1
    )
    TI_Tox = np.minimum(np.minimum(df["TI_Brain"].values, df["TI_Heart"].values), df["TI_Lung"].values)

    Simple_Tox = np.minimum(df["N_Off_Targets_1.0"] + 
        0.25 * df["N_Off_Targets_0.5"] + 
        0.25**2 * df["N_Off_Targets_0.25"] +
        0.25**3 * df["N_Off_Targets_0.1"] + 
        0.25**4 * df["N_Off_Targets_0.05"], 5)

    Simple_Tox = np.maximum(Simple_Tox - (df["SC_2nd_Target_Val"] - 1), 0)

    # Add bonuses (vectorized)
    #target_quality += df["Positive_Final_v2"].values / 5 # Not Sure If We Want This
    target_quality += 10 * np.minimum(LFC_On_50_vs_Tox, 1) + 2 * (np.minimum(LFC_On_50_vs_Tox, 5) - 1) #A little wiggle
    target_quality += 10 * df["TI"].values + 10 * df["TI_NonImmune"].values + 5 * (df["TI"].values > 0).astype(int)
    target_quality += 10 * TI_Tox
    target_quality += 2.5 * np.minimum(df["On_Val_75"].values, 2) + 2 * np.maximum(np.minimum(df["On_Val_75"].values, 5) - 2, 0) #A little wiggle
    target_quality += 10 * np.minimum(df["Specificity"].values, 2) #Only for ones with 0 off targets!

    # Apply penalties (vectorized with np.where)
    target_quality = np.where(df["N_Off_Targets_1.0"].values > 5, target_quality - 10, target_quality)
    target_quality = np.where(df["N_Off_Targets_1.0"].values > 15, target_quality - 10, target_quality)

    target_quality = np.where(df["Hazard_SC_v1"].values > 25, target_quality - 10, target_quality)
    target_quality = np.where(df["Hazard_FFPE_v1"].values > 25, target_quality - 10, target_quality)
    target_quality = np.where(TI_Tox < 0.1, target_quality - 20, target_quality)
    target_quality = np.where(TI_Tox == 0, target_quality - 10, target_quality)
    target_quality -= Simple_Tox

    # Bulk Expression Weighting
    target_quality = np.where(df["Hazard_GTEX_v1"].values > 25, target_quality - 5, target_quality)
    target_quality = np.where(df["GTEX_Tox_Tier1"].values > 1, target_quality - 10, target_quality)
    target_quality = np.where(df["GTEX_Tox_Tier2"].values > 4, target_quality - 5, target_quality)
    
    # Zero out bad targets
    target_quality = np.where(
        (df["Positive_Final_v2"].values <= 1) | (df["Target_Val"].values <= 0.1), 
        0, 
        target_quality
    )

    # Scale and clamp to 0-100
    df["TargetQ_Final_v2"] = np.maximum(target_quality + 50, 0) #For Pentalies Below 0
    df["TargetQ_Final_v2"] = 100 * df["TargetQ_Final_v2"] / (198.301384) #Value for CB21
    df = df[df.columns[df.columns != 'gene_name'].tolist() + ['gene_name']]

    return df.sort_values("TargetQ_Final_v2", ascending=False)

def target_quality_v1(
    df: pd.DataFrame,
    multi: False
) -> pd.DataFrame:
    """Compute target quality scores with surface protein evidence"""
    
    from py_target_id import utils

    if multi:
    
        try:
            # Load surface evidence
            surface_series = utils.surface_evidence()
            
            # Split gene pairs
            gene_splits = df['gene_name'].str.split('_', n=1, expand=True)
            
            # Map surface evidence for both genes
            gene1 = gene_splits[0].map(surface_series).fillna(1.0)
            gene2 = gene_splits[1].map(surface_series).fillna(1.0)
            
            # Take minimum (both genes must have surface evidence)
            df['Surface_Prob'] = np.minimum(gene1, gene2)
            
        except Exception as e:
            print(f"Warning: Could not load surface evidence ({e}), using default value 1.0")
            df['Surface_Prob'] = 1.0

    else :

        try:
            # Load surface evidence
            surface_series = utils.surface_evidence()
            df['Surface_Prob'] = df['gene_name'].map(surface_series).fillna(1.0)
        
        except Exception as e:
            print(f"Warning: Could not load surface evidence ({e}), using default value 1.0")
            df['Surface_Prob'] = 1.0

    # Score components
    df['Score_1'] = 10
    df.loc[df['N_Off_Targets'] <= 3, 'Score_1'] = df.loc[df['N_Off_Targets'] <= 3, 'N_Off_Targets']
    
    df['Score_2'] = 10
    df.loc[df['N_Off_Targets_0.5'] <= 3, 'Score_2'] = df.loc[df['N_Off_Targets_0.5'] <= 3, 'N_Off_Targets_0.5']
    
    df['Score_3'] = 10
    df.loc[df['Corrected_Specificity'] >= 0.75, 'Score_3'] = 0
    df.loc[(df['Corrected_Specificity'] >= 0.5) & (df['Corrected_Specificity'] < 0.75), 'Score_3'] = 1
    df.loc[(df['Corrected_Specificity'] >= 0.35) & (df['Corrected_Specificity'] < 0.5), 'Score_3'] = 3
    
    df['Score_4'] = 10
    df.loc[df['P_Pos_Val_0.5'] > 0.25, 'Score_4'] = 0
    df.loc[(df['P_Pos_Val_0.5'] > 0.15) & (df['P_Pos_Val_0.5'] <= 0.25), 'Score_4'] = 1
    df.loc[(df['P_Pos_Val_0.5'] > 0.025) & (df['P_Pos_Val_0.5'] <= 0.15), 'Score_4'] = 3
    df.loc[df['N_Pos_Val_0.5'] == 1, 'Score_4'] = 10
    
    df['Score_5'] = 10
    df.loc[df['P_Pos_Specific'] > 0.25, 'Score_5'] = 0
    df.loc[(df['P_Pos_Specific'] > 0.15) & (df['P_Pos_Specific'] <= 0.25), 'Score_5'] = 1
    df.loc[(df['P_Pos_Specific'] > 0.025) & (df['P_Pos_Specific'] <= 0.15), 'Score_5'] = 3
    df.loc[df['N_Pos_Specific'] == 1, 'Score_5'] = 10
    
    df['Score_6'] = 10
    df.loc[df['SC_2nd_Target_Val'] > 2, 'Score_6'] = 0
    df.loc[(df['SC_2nd_Target_Val'] > 1) & (df['SC_2nd_Target_Val'] <= 2), 'Score_6'] = 1
    df.loc[(df['SC_2nd_Target_Val'] > 0.5) & (df['SC_2nd_Target_Val'] <= 1), 'Score_6'] = 3
    df.loc[(df['SC_2nd_Target_Val'] > 0.1) & (df['SC_2nd_Target_Val'] <= 0.5), 'Score_6'] = 5
    
    df['Score_7'] = 10
    df.loc[df['Surface_Prob'] >= 0.5, 'Score_7'] = 0
    df.loc[(df['Surface_Prob'] >= 0.1875) & (df['Surface_Prob'] < 0.5), 'Score_7'] = 3
    df.loc[(df['Surface_Prob'] >= 0.125) & (df['Surface_Prob'] < 0.1875), 'Score_7'] = 7
    
    # Compute final TargetQ score
    score_columns = ['Score_1', 'Score_2', 'Score_3', 'Score_5', 'Score_6', 'Score_7']
    penalty_columns = ['Score_1', 'Score_2', 'Score_3']
    
    raw_scores = df[score_columns].sum(axis=1)
    penalty_count = (df[penalty_columns] == 10).sum(axis=1)
    penalized_scores = raw_scores / 60 + 0.25 * penalty_count
    df['TargetQ_Final_v1'] = (100 / 1.75) * (1.75 - penalized_scores)
    
    return df

