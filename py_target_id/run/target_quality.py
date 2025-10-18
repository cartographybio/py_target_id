"""
Target ID Multi v1 - GPU Optimized with Required Gene Pairs
"""

__all__ = ['target_quality_v1', 'target_quality_v2']

import torch
import numpy as np
import pandas as pd
import time
import gc
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

def target_quality_v1(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute target quality scores with surface protein evidence.
    
    Version 1: Original hand-crafted scoring including Surface Probability component.
    Scores range 0-100, where higher = better target quality.
    """
    
    import numpy as np
    from py_target_id import utils
    
    df = df.copy()
    
    # ===== LOAD SURFACE PROTEIN EVIDENCE =====
    # Surface proteins are more likely to be therapeutically relevant
    try:
        surface_series = utils.surface_evidence()
        df['Surface_Prob'] = df['gene_name'].map(surface_series).fillna(1.0)
    except Exception as e:
        print(f"Warning: Could not load surface evidence ({e}), using default value 1.0")
        df['Surface_Prob'] = 1.0
    
    # ===== SCORE 1: OFF-TARGET COUNT AT DEFAULT THRESHOLD =====
    # Lower off-targets = better specificity
    # Penalty component: contributes to final penalty if = 10
    df['Score_1'] = 10
    df.loc[df['N_Off_Targets'] <= 3, 'Score_1'] = df.loc[df['N_Off_Targets'] <= 3, 'N_Off_Targets']
    # Range: 0-10 (0=best, 10=worst)
    
    # ===== SCORE 2: OFF-TARGET COUNT AT 0.5 THRESHOLD =====
    # More stringent off-target threshold
    # Penalty component: contributes to final penalty if = 10
    df['Score_2'] = 10
    df.loc[df['N_Off_Targets_0.5'] <= 3, 'Score_2'] = df.loc[df['N_Off_Targets_0.5'] <= 3, 'N_Off_Targets_0.5']
    # Range: 0-10 (0=best, 10=worst)
    
    # ===== SCORE 3: CORRECTED SPECIFICITY =====
    # Higher specificity = better (fewer off-targets in malignant cells specifically)
    # Penalty component: contributes to final penalty if = 10
    df['Score_3'] = 10
    df.loc[df['Corrected_Specificity'] >= 0.75, 'Score_3'] = 0  # Excellent: >= 75%
    df.loc[(df['Corrected_Specificity'] >= 0.5) & (df['Corrected_Specificity'] < 0.75), 'Score_3'] = 1  # Good: 50-75%
    df.loc[(df['Corrected_Specificity'] >= 0.35) & (df['Corrected_Specificity'] < 0.5), 'Score_3'] = 3  # Fair: 35-50%
    # Range: 0-10 (0=best, 10=worst)
    
    # ===== SCORE 4: PATIENT POSITIVITY PERCENTAGE =====
    # Fraction of patients where target is positive
    # Higher positivity = present in more patients (better for CAR-T)
    df['Score_4'] = 10
    df.loc[df['P_Pos_Per'] > 0.25, 'Score_4'] = 0  # > 25% of patients: excellent
    df.loc[(df['P_Pos_Per'] > 0.15) & (df['P_Pos_Per'] <= 0.25), 'Score_4'] = 1  # 15-25%: good
    df.loc[(df['P_Pos_Per'] > 0.025) & (df['P_Pos_Per'] <= 0.15), 'Score_4'] = 3  # 2.5-15%: fair
    df.loc[df['N_Pos_Val'] == 1, 'Score_4'] = 10  # Only 1 patient positive: penalize
    # Range: 0-10 (0=best, 10=worst)
    
    # ===== SCORE 5: ABSOLUTE PATIENT POSITIVITY COUNT =====
    # Number of patients where target is positive (requires minimum threshold)
    df['Score_5'] = 10
    df.loc[df['P_Pos'] > 0.25, 'Score_5'] = 0  # > 25% of samples positive: excellent
    df.loc[(df['P_Pos'] > 0.15) & (df['P_Pos'] <= 0.25), 'Score_5'] = 1  # 15-25%: good
    df.loc[(df['P_Pos'] > 0.025) & (df['P_Pos'] <= 0.15), 'Score_5'] = 3  # 2.5-15%: fair
    df.loc[df['N_Pos'] == 1, 'Score_5'] = 10  # Only 1 positive sample: penalize heavily
    # Range: 0-10 (0=best, 10=worst)
    
    # ===== SCORE 6: SECOND-BEST TARGET EXPRESSION =====
    # How well the second-highest expressing gene on this target performs
    # Higher = better (multi-specific targets are broader hitting)
    df['Score_6'] = 10
    df.loc[df['SC_2nd_Target_Val'] > 2, 'Score_6'] = 0  # > 2.0: excellent secondary expression
    df.loc[(df['SC_2nd_Target_Val'] > 1) & (df['SC_2nd_Target_Val'] <= 2), 'Score_6'] = 1  # 1-2: good
    df.loc[(df['SC_2nd_Target_Val'] > 0.5) & (df['SC_2nd_Target_Val'] <= 1), 'Score_6'] = 3  # 0.5-1: moderate
    df.loc[(df['SC_2nd_Target_Val'] > 0.1) & (df['SC_2nd_Target_Val'] <= 0.5), 'Score_6'] = 5  # 0.1-0.5: weak
    # Range: 0-10 (0=best, 10=worst)
    
    # ===== SCORE 7: SURFACE PROTEIN PROBABILITY =====
    # Likelihood that this protein is actually on cell surface
    df['Score_7'] = 10
    df.loc[df['Surface_Prob'] >= 0.5, 'Score_7'] = 0  # >= 50% likely surface: excellent
    df.loc[(df['Surface_Prob'] >= 0.1875) & (df['Surface_Prob'] < 0.5), 'Score_7'] = 3  # 18.75-50%: fair
    df.loc[(df['Surface_Prob'] >= 0.125) & (df['Surface_Prob'] < 0.1875), 'Score_7'] = 7  # < 18.75%: poor
    # Range: 0-10 (0=best, 10=worst)
    
    # ===== COMBINE SCORES INTO FINAL TargetQ =====
    # Include all 6 score components (Score_7 is surface probability)
    score_columns = ['Score_1', 'Score_2', 'Score_3', 'Score_5', 'Score_6', 'Score_7']
    # Track penalty columns (scores that = 10 indicate major problems)
    penalty_columns = ['Score_1', 'Score_2', 'Score_3']
    
    # Sum all component scores (max = 60 for 6 scores × 10)
    raw_scores = df[score_columns].sum(axis=1)
    
    # Count how many penalty components are at their worst (= 10)
    # This heavily penalizes targets with multiple critical failures
    penalty_count = (df[penalty_columns] == 10).sum(axis=1)
    
    # Normalize: raw_scores/60 gives 0-1, plus 0.25 per penalty
    # Max penalized_score ≈ 1.75 (1.0 + 0.75 from 3 penalties)
    penalized_scores = raw_scores / 60 + 0.25 * penalty_count
    
    # Scale to 0-100: (1.75 - penalized_scores) gives range 0-1.75
    # Multiply by 100/1.75 to normalize to 0-100
    # Result: good targets (low penalized_scores) → high TargetQ, bad targets → low TargetQ
    df['TargetQ_Final_v1'] = (100 / 1.75) * (1.75 - penalized_scores)
    
    return df

def target_quality_v2(df: pd.DataFrame) -> pd.DataFrame:
    """Version 2: Scoring without Surface Probability component"""
    
    import numpy as np
    
    df = df.copy()
    
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
    df.loc[df['SC_2nd_Target_Val'] > 2, 'Score_4'] = 0
    df.loc[(df['SC_2nd_Target_Val'] > 1) & (df['SC_2nd_Target_Val'] <= 2), 'Score_4'] = 1
    df.loc[(df['SC_2nd_Target_Val'] > 0.5) & (df['SC_2nd_Target_Val'] <= 1), 'Score_4'] = 3
    df.loc[(df['SC_2nd_Target_Val'] > 0.1) & (df['SC_2nd_Target_Val'] <= 0.5), 'Score_4'] = 5
    
    df['Score_5'] = 10
    df.loc[df['SC_Hazard_Risk'] < 0.5, 'Score_5'] = 0
    df.loc[(df['SC_Hazard_Risk'] >= 0.5) & (df['SC_Hazard_Risk'] < 2), 'Score_5'] = 1
    df.loc[(df['SC_Hazard_Risk'] >= 2) & (df['SC_Hazard_Risk'] < 10), 'Score_5'] = 3
    df.loc[(df['SC_Hazard_Risk'] >= 10) & (df['SC_Hazard_Risk'] < 25), 'Score_5'] = 5
    
    df['Score_6'] = 10
    df.loc[df['FFPE_Hazard_Risk'] < 0.5, 'Score_6'] = 0
    df.loc[(df['FFPE_Hazard_Risk'] >= 0.5) & (df['FFPE_Hazard_Risk'] < 2), 'Score_6'] = 1
    df.loc[(df['FFPE_Hazard_Risk'] >= 2) & (df['FFPE_Hazard_Risk'] < 10), 'Score_6'] = 3
    df.loc[(df['FFPE_Hazard_Risk'] >= 10) & (df['FFPE_Hazard_Risk'] < 25), 'Score_6'] = 5
    
    # Compute final TargetQ score
    score_columns = ['Score_1', 'Score_2', 'Score_3', 'Score_4', 'Score_5', 'Score_6']
    penalty_columns = ['Score_1', 'Score_2', 'Score_3', 'Score_5', 'Score_6']
    
    # Sum raw scores (each score ranges 0-10, so max sum is 60)
    raw_scores = df[score_columns].sum(axis=1)
    
    # Count penalties: when a score = 10 (worst case)
    # Max penalty count is 5, contributing 1.25 to penalized_scores (5 * 0.25)
    penalty_count = (df[penalty_columns] == 10).sum(axis=1)
    
    # Normalize raw scores to 0-1 range, then add penalty term
    # penalized_scores ranges from 0 to ~2.25 (1.0 from raw/60 + 1.25 from penalties)
    penalized_scores = raw_scores / 60 + 0.25 * penalty_count
    
    # Calculate max possible penalized score dynamically
    max_penalized_score = 1.0 + (0.25 * len(penalty_columns))
    
    # Rescale to 0-100 range
    df['TargetQ_Final_v2'] = (100 / max_penalized_score) * (max_penalized_score - penalized_scores)
    
    # Hard penalty for very high hazard targets
    ix = (df['SC_Hazard_Risk'] > 40) | (df['FFPE_Hazard_Risk'] > 40)
    df.loc[ix, 'TargetQ_Final_v2'] = df.loc[ix, 'TargetQ_Final_v2'] - 10
    
    # Hard penalty if Target_Val > 2x SC_2nd_Target_Val AND SC_2nd_Target_Val < 0.5
    ix = (df['Target_Val'] > 2 * df['SC_2nd_Target_Val']) & (df['SC_2nd_Target_Val'] < 0.5)
    df.loc[ix, 'TargetQ_Final_v2'] = df.loc[ix, 'TargetQ_Final_v2'] - 10

    # Hard penalty on Specificity
    ix = (df["P_Pos"] <= 0.05) | (df["N_Pos"] <= 2) #Suggests Minimal
    df.loc[ix, 'TargetQ_Final_v2'] = df.loc[ix, 'TargetQ_Final_v2'] - 25
    
    # Ensure no negative scores
    df['TargetQ_Final_v2'] = np.maximum(df['TargetQ_Final_v2'], 0)
    
    return df