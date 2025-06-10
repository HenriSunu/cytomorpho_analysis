
import pandas as pd
import numpy as np

from pandas import DataFrame, Series
from cell_pipeline.constants import (
    CELL_LINEAGES,
    CELL_LABEL_MAP,
    APL_INCLUDED_CLASSES,
    APL_RESULTS,
    MDS_INCLUDED_CLASSES,
    MDS_RESULTS,
)


def engineer_statistics(
    stat_df: DataFrame,
) -> DataFrame:
    
    # Nuclear-cytoplasmic ratio
    stat_df["cell_N:C-ratio"] = (
        (stat_df["cell_area"] - stat_df["cyt_area"]) / stat_df["cell_area"]
    )

    return stat_df

def get_result_counts(
    result_df: DataFrame,
) -> Series:
    """
    Extracts counts of cells with each result by cell type and cell lineages.
    """
    
    raw_series = pd.Series(dtype=int)

    # Basic cell differential, loop over to get 0 values too
    cell_types = list(CELL_LABEL_MAP.values())
    cell_counts = result_df["CLS-LBL"].value_counts()
    for ct in cell_types:
        if ct in cell_counts.index:
            raw_series[ct] = cell_counts[ct]
        else:
            raw_series[ct] = 0

    # Cell lineages
    for lineage in CELL_LINEAGES.keys():
        raw_series[lineage] = result_df[result_df["CLS-LBL"].isin(CELL_LINEAGES[lineage])].shape[0]

    # APL counts
    for cls in APL_INCLUDED_CLASSES:
        if cls in cell_types:
            cls_df = result_df[result_df["CLS-LBL"] == cls]
        else: 
            cls_df = result_df[result_df["CLS-LBL"].isin(CELL_LINEAGES[cls])]
        
        for result in APL_RESULTS:
            raw_series[f"{cls}-{result.split('-')[1]}"] = (cls_df[result] > 0.5).sum()

    # MDS counts
    for cls in MDS_INCLUDED_CLASSES:
        if cls in cell_types:
            cls_df = result_df[result_df["CLS-LBL"] == cls]
        else:
            cls_df = result_df[result_df["CLS-LBL"].isin(CELL_LINEAGES[cls])]

        for result in MDS_RESULTS:
            raw_series[f"{cls}-{result.split('-')[1]}"] = (cls_df[result] > 0.5).sum()

    # Mitotic counts
    for ct in cell_types:
        cls_df = result_df[result_df["CLS-LBL"] == ct]
        raw_series[f"{ct}-MITOTIC"] = (cls_df["MITOTIC"] > 0.5).sum()

    # Vacuoli counts
    for ct in cell_types:
        cls_df = result_df[result_df["CLS-LBL"] == ct]
        raw_series[f"{ct}-VACUOLI"] = (cls_df["VACUOLI"] > 0.5).sum()

    return raw_series