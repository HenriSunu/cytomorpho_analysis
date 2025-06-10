import pandas as pd
import numpy as np

from h5py import Group
from pandas import DataFrame


from cell_pipeline.constants import (
    STATISTICS,
    CELL_LABEL_MAP,
    ALL_RESULTS,
)

def extract_cell_statistics(
    statistics: Group,
    stat_df: DataFrame
) -> DataFrame:
    """
    Extracts cell statistics from the HDF5 file for all cells.
    """

    # Cytoplasm
    cyto = statistics["CYTOPLASM"]
    for field in STATISTICS["CYTOPLASM"]:
        stat_df[f"cyt_{field}"] = cyto[field][:]

    # Nucleus
    nuc = statistics["NUCLEUS"]
    for field in STATISTICS["NUCLEUS"]:
        stat_df[f"nuc_{field}"] = nuc[field][:]

    # Entire cell
    cell = statistics["ENTIRE"]
    for field in STATISTICS["ENTIRE"]:
        stat_df[f"cell_{field}"] = cell[field][:]

    return stat_df

def extract_cell_results(
    sample: Group,
) -> DataFrame:
    """
    Extracts cell results from the HDF5 file for all cells.
    """

    result_df = pd.DataFrame()

    # CLS-LBL
    result_df["CLS-LBL"] = [CELL_LABEL_MAP[x] for x in sample["CLS-LBL"][:]]
    result_df.index.name = "CELL-ID"

    # Results
    for res in ALL_RESULTS:
        result = sample[res][:]

        if result.ndim == 1:
            result_df[res] = result
        else:
            for i in np.arange(result.shape[1]):
                result_df[f"{res}-{i}"] = result[:, i]
    
    return result_df