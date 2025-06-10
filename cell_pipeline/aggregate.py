import pandas as pd
import numpy as np
from pandas import DataFrame, Series

from cell_pipeline.feature_engineering import get_result_counts
from cell_pipeline.constants import (
    CELL_STATISTICS_CLASSES,
    CELL_LINEAGES,
    CELL_LABEL_MAP,
    APL_INCLUDED_CLASSES,
    APL_RESULTS,
    MDS_INCLUDED_CLASSES,
    MDS_RESULTS,
)

def aggregate_statistics(
    stat_df: DataFrame,
    minimum_to_aggregate: int,
) -> Series:
    """
    Aggregates cell-wise statistics to sample-wise statistics by cell type.
    """

    agg_ser = pd.Series(dtype=np.float32)
    for ct in CELL_STATISTICS_CLASSES:
        ct_df = stat_df[stat_df["CLS-LBL"] == ct].drop("CLS-LBL", axis=1)
        median_index = [f"{ct}_{col}_median" for col in ct_df.columns]
        range95_index = [f"{ct}_{col}_95_range" for col in ct_df.columns]

        if ct_df.shape[0] < minimum_to_aggregate:
            empty_ser = pd.Series(
                np.full(2*ct_df.shape[1], np.nan),
                index=median_index + range95_index
            )
            agg_ser = pd.concat([agg_ser, empty_ser], axis=0)
        else:
            median_ser = ct_df.apply(np.nanmedian, axis=0)
            range95_ser = ct_df.apply(
                lambda x: np.nanpercentile(x, 97.5) - np.nanpercentile(x, 2.5),
                axis=0,
                result_type="reduce"
            )
            
            median_ser.index = pd.Index(median_index)
            range95_ser.index = pd.Index(range95_index)

            agg_ser = pd.concat([agg_ser, median_ser, range95_ser], axis=0)
    
    if not isinstance(agg_ser, Series):
        raise TypeError("Expected series")
    
    return agg_ser


def min_division(
    numerator: int,
    denominator: int,
    min: int,
) -> float | None:
    """
    Returns the fraction if the denominator is larger than min, and None otherwise
    """
    if denominator >= min:
        return numerator / denominator
    else:
        return None


def aggregate_results(
    result_df: DataFrame,
    minimum_to_aggregate: int,
) -> Series:
    """
    Aggregates cell-wise results to sample-wise results by cell type.
    """
    
    # Get raw cell differential counts
    raw_results_ser = get_result_counts(result_df)
    
    # Calculate proportions
    proportions_ser = pd.Series()
    # Basic cell differential
    cell_types = np.array(list(CELL_LABEL_MAP.values()))[1:-1]
    for ct in cell_types:
        proportions_ser[f"Living_cells-{ct}_proportion"] = (
            min_division(raw_results_ser[ct], raw_results_ser["Living_cells"], minimum_to_aggregate)
        )
    
    # Lineage proportions
    for lineage in CELL_LINEAGES.keys():
        proportions_ser[f"Living_cells-{lineage}_proportion"] = (
            min_division(raw_results_ser[lineage], raw_results_ser["Living_cells"], minimum_to_aggregate)
        )

    # APL
    for ct in APL_INCLUDED_CLASSES:
        for apl_res in APL_RESULTS:
            apl_res = apl_res.split("-")[1]
            proportions_ser[f"{ct}-{apl_res}_proportion"] = (
                min_division(raw_results_ser[f"{ct}-{apl_res}"], raw_results_ser[ct], minimum_to_aggregate)
            )

    # MDS
    for ct in MDS_INCLUDED_CLASSES:
        for mds_res in MDS_RESULTS:
            mds_res = mds_res.split("-")[1]
            proportions_ser[f"{ct}-{mds_res}_proportion"] = (
                min_division(raw_results_ser[f"{ct}-{mds_res}"], raw_results_ser[ct], minimum_to_aggregate)
            )

    # Mitotic
    for ct in cell_types:
        proportions_ser[f"{ct}-MITOTIC_proportion"] = (
            min_division(raw_results_ser[f"{ct}-MITOTIC"], raw_results_ser[ct], minimum_to_aggregate)
        )

    # Vacuoli
    for ct in cell_types:
        proportions_ser[f"{ct}-VACUOLI_proportion"] = (
            min_division(raw_results_ser[f"{ct}-VACUOLI"], raw_results_ser[ct], minimum_to_aggregate)
        )

    return proportions_ser