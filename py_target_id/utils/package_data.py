"""
Manifest loading and processing functions.
"""

# Define what gets exported
__all__ = ['surface_genes', 'tabs_genes', 'valid_genes', 'surface_evidence', 'apical_genes']

"""
Data loading functions for py_target_id package.
Provides access to annotation data like surface proteins, TABS genes, etc.
"""

from importlib.resources import files
import pandas as pd
import numpy as np
from typing import List, Optional


def _get_data_path(subpath: str) -> str:
    """Get path to a data file in the package"""
    return files('py_target_id').joinpath(f'data/{subpath}')

def surface_genes(
    version: int = 2,
    tiers: List[int] = [1, 2],
    latest: bool = False,
    evidence: float = 0.1
) -> List[str]:
    """
    Load surface protein genes filtered by tier and evidence.
    
    Parameters
    ----------
    version : int, default=2
        Version of surface gene classification:
        - 1: Evidence-based filtering
        - 2: Tier-based classification
    tiers : list of int, default=[1, 2]
        Which tiers to include (only for version=2)
    latest : bool, default=False
        If True, fetch from Google Sheets (requires authentication)
        If False, use packaged data
    evidence : float, default=0.1
        Minimum evidence score threshold (only for version=1)
    
    Returns
    -------
    list
        List of surface gene names
    """
    if version == 1:
        sgenes_series = surface_evidence()
        sgenes = sgenes_series[sgenes_series >= evidence].index.tolist()
        return sgenes
    
    elif version == 2:
        if latest:
            # Would require gspread or similar library
            raise NotImplementedError(
                "Google Sheets integration not implemented. "
                "Use latest=False to load from package data."
            )
        else:
            path = _get_data_path('annotation/surface_tiers.20250804.csv')
            surface = pd.read_csv(path)
        
        # Custom edits - match R code exactly
        surface.loc[surface['gene_name'] == 'SLC22A31', 'Tier'] = 'Tier1'
        surface.loc[surface['gene_name'] == 'ROS1', 'Tier'] = 'Tier1'
        surface.loc[surface['gene_name'] == 'CRLF1', 'Tier'] = 'Tier3'
        surface.loc[surface['gene_name'] == 'C1orf210', 'Tier'] = 'Tier3'
        surface.loc[surface['gene_name'] == 'ZDHHC9', 'Tier'] = 'Tier3'
        surface.loc[surface['gene_name'] == 'KDELR3', 'Tier'] = 'Tier3'
        
        # 20250828 updates
        surface.loc[surface['gene_name'].isin([
            'SMIM22', 'RTN4RL2', 'LPCAT1', 'DPY19L1', 'TLCD1'
        ]), 'Tier'] = 'Tier3'
        
        # TABS genes update
        surface.loc[surface['gene_name'].isin([
            'AMHR2', 'CD70', 'CLEC6A', 'CNTN4', 'FOLH1', 'GPA33',
            'GRIN1', 'LOXL2', 'TLR7', 'TNF', 'TNFSF13', 'VEGFA'
        ]), 'Tier'] = 'Tier1'
        
        surface.loc[surface['gene_name'].isin([
            'MFI2', 'MUC5AC', 'ADAMTS5', 'LGALS9'
        ]), 'Tier'] = 'Tier1'
        
        # Filter by requested tiers
        tier_strings = [f'Tier{t}' for t in tiers]
        sgenes = surface.loc[surface['Tier'].isin(tier_strings), 'gene_name'].tolist()
        
        return sgenes
    
    else:
        raise ValueError(f"Unknown version: {version}")


def tabs_genes(version: int = 2) -> List[str]:
    """
    Load TABS (Therapeutically Applicable Body Site) genes.
    
    Parameters
    ----------
    version : int, default=2
        Version of TABS data:
        - 1: Original antibody count data (2023)
        - 2: Clinical antibody data (2025)
    
    Returns
    -------
    list
        Sorted list of unique TABS gene names
    """
    if version == 1:
        path = _get_data_path('annotation/TABs_Antibody_Count.20230323.xlsx')
        df = pd.read_excel(path)
        genes = df['final_gene'].dropna().tolist()
        genes = sorted(set(genes + ['SLC34A2']))
        return genes
    
    elif version == 2:
        path = _get_data_path('annotation/TABS_Antibody_Clinical.20250815.xlsx')
        df = pd.read_excel(path, sheet_name='all_clinical_symbol')
        genes = df['symbol'].dropna().tolist()
        genes = sorted(set(genes + ['SLC34A2', 'FOLR1']))
        return genes
    
    else:
        raise ValueError(f"Unknown version: {version}")


def valid_genes(version: str = "20240715") -> List[str]:
    """
    Load list of valid genes for analysis.
    
    Parameters
    ----------
    version : str, default="20240715"
        Version of the valid genes list to load
    
    Returns
    -------
    list
        List of valid gene names
    """
    if version == "20240715":
        # Assuming CSV format since .rds doesn't exist
        # Adjust based on your actual file format
        path = _get_data_path('annotation/sc_valid_genes.cellranger.20240715.csv')
        df = pd.read_csv(path)
        return df['gene_name'].tolist() if 'gene_name' in df.columns else df.iloc[:, 0].tolist()
    else:
        raise ValueError(f"Unknown version: {version}")


def surface_evidence(version: str = "20240715") -> pd.Series:
    """
    Load surface protein evidence scores.
    
    Parameters
    ----------
    version : str, default="20240715"
        Version of the surface evidence data to load
    
    Returns
    -------
    pd.Series
        Series with gene names as index and evidence scores as values
    """
    if version == "20240715":
        path = _get_data_path('annotation/surface_evidence.v1.20240715.csv')
        df = pd.read_csv(path, index_col=0)
        # Assuming first column is gene names, second is evidence score
        return df.iloc[:, 0] if df.shape[1] == 1 else df.squeeze()
    else:
        raise ValueError(f"Unknown version: {version}")


def apical_genes(version: str = "20240715") -> List[str]:
    """
    Load apical localization genes.
    
    Parameters
    ----------
    version : str, default="20240715"
        Version of the apical genes data to load
    
    Returns
    -------
    list
        List of apical gene names
    """
    if version == "20240715":
        path = _get_data_path('annotation/apical.v1.20240715.csv')
        df = pd.read_csv(path, index_col=0)
        # Return genes where value == 1
        if df.shape[1] == 1:
            return df[df.iloc[:, 0] == 1].index.tolist()
        else:
            # Assuming binary column exists
            return df[df.iloc[:, 0] == 1].index.tolist()
    else:
        raise ValueError(f"Unknown version: {version}")

