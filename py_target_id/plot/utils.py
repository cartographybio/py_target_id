"""
Manifest loading and processing functions.
"""

# Define what gets exported
__all__ = ['get_top_n_per_interval']

import numpy as np
import pandas as pd
def get_top_n_per_interval(df, x_col, y_col, label_col, n=2, interval=10):
    """
    Get top N points per x-axis interval for labeling
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    x_col : str
        Column name for x-axis values (e.g., 'P_Pos_Per')
    y_col : str
        Column name for y-axis values (e.g., 'TargetQ_Final_v1')
    label_col : str
        Column name for labels (e.g., 'gene_name')
    n : int
        Number of top points to select per interval
    interval : float
        Interval width as percentage (e.g., 10 for 10%)
    
    Returns
    -------
    pd.DataFrame
        Subset with top N points per interval
    """
    # Scale x values to 0-100 range
    x_values = df[x_col]
    x_min, x_max = x_values.min(), x_values.max()
    x_scaled = (x_values - x_min) / (x_max - x_min) * 100
    
    # Create bins
    bins = np.arange(0, 101, interval)
    df_copy = df.copy()
    df_copy['interval'] = pd.cut(x_scaled, bins=bins, labels=bins[:-1], include_lowest=True)
    
    # Get top N per interval
    top_per_interval = (df_copy
                       .groupby('interval', observed=True)
                       .apply(lambda x: x.nlargest(n, y_col))
                       .reset_index(drop=True))
    
    return top_per_interval[[x_col, y_col, label_col]]