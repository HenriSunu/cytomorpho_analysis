import h5py

import pandas as pd
import numpy as np

from h5py import Group
from pandas import DataFrame

from cell_pipeline.cell_filters import filter_cells
from cell_pipeline.aggregate import (
    aggregate_statistics,
    aggregate_results
)
from cell_pipeline.extract import (
    extract_cell_statistics,
    extract_cell_results
)
from cell_pipeline.feature_engineering import (
    get_result_counts,
    engineer_statistics
)

from cell_pipeline.constants import (
    CELL_LABEL_MAP
)

def _get_statistics(
    sample: Group,
    min_dist_from_edge: int,
    max_aspect_ratio: float,
    filter_with_erythroblast_quality: bool,
    filter_with_granulocyte_quality: bool,
    filter_artefacts: bool,
    filter_unknowns: bool,
    filter_with_quality: bool,
    min_megakaryocyte_size: tuple,
) -> DataFrame:
    """
    Extracts cell statistics for one sample from the HDF5 file.
    """

    stat_df = pd.DataFrame()
    stat_df["CLS-LBL"] = [CELL_LABEL_MAP[x] for x in sample["CLS-LBL"][:]]
    stat_df.index.name = "CELL-ID"

    stat_group = sample["STATISTICS"]

    if isinstance(stat_group, Group):
        stat_df = extract_cell_statistics(
            stat_group,
            stat_df
        )
    else:
        raise ValueError("Invalid group in HDF5 file. Expected a Group object. Check the STATISTICS subgroup.")

    stat_df = engineer_statistics(stat_df)

    # Apply filters
    stat_df = filter_cells(
        stat_df,
        sample,
        min_dist_from_edge,
        max_aspect_ratio,
        filter_with_erythroblast_quality,
        filter_with_granulocyte_quality,
        filter_artefacts,
        filter_unknowns,
        filter_with_quality,
        min_megakaryocyte_size,
    )

    return stat_df


def read_hdf5_statistics(
    hdf5_path: str,
    mgg_db: str,
    album_id: str,
    aggregate: bool = True,
    minimum_to_aggregate: int = 30,
    min_dist_from_edge: int = 1,
    max_aspect_ratio: float = np.inf,
    filter_with_erythroblast_quality: bool = True,
    filter_with_granulocyte_quality: bool = True,
    filter_artefacts: bool = True,
    filter_unknowns: bool = True,
    filter_with_quality: bool = True,
    min_megakaryocyte_size: tuple = (0.146, 0.122)
) -> DataFrame:
    """
    Reads cell statistics for one sample from the HDF5 file.

    Args:
        hdf5_path (str): Path to the HDF5 file.
        mgg_db (str): MGG database name.
        album_id (str): album_id.
        aggregate (bool): If True, aggregate statistics for cells by cell type.
        minimum_to_aggregate (int): Minimum number of cells required to aggregate statistics for cell type.
        min_dist_from_edge (int): Cells closer than this distance to the image edge will be filtered out.
        max_aspect_ratio (float): Cells with an aspect ratio larger than this value will be filtered out.
        filter_with_erythroblast_quality (bool): If True, filter low quality erythroblasts.
        filter_with_granulocyte_quality (bool): If True, filter low quality granulocytes.
        filter_artefacts (bool): If True, filter artefacts.
        filter_unknowns (bool): If True, filter unknown cells.
        filter_with_quality (bool): If True, filter cells with additional cell statistics-based models.
        min_megakaryocyte_size (tuple): (width, height) Megakaryocytes smaller than this size will be filtered out. The units are relative to the image sizes.

    Returns:
        DataFrame: if aggregate is True, a single-row DataFrame with the aggregated statistics
            for each cell type, otherwise a DataFrame with statistics for each cell.

    """
    with open(hdf5_path, "rb", buffering=0) as file, h5py.File(file) as f:
        group = f[f"{mgg_db}/{album_id}"]

        if isinstance(group, Group):
            stat_df = _get_statistics(
                group,
                min_dist_from_edge,
                max_aspect_ratio,
                filter_with_erythroblast_quality,
                filter_with_granulocyte_quality,
                filter_artefacts,
                filter_unknowns,
                filter_with_quality,
                min_megakaryocyte_size,
            )
        else:
            raise ValueError("Invalid group in HDF5 file. Expected a Group object. Check the mgg_database and album_id?")

    if aggregate:
        agg_statistics = aggregate_statistics(
            stat_df,
            minimum_to_aggregate,
        )
        stat_df = agg_statistics.to_frame().T
        stat_df.index = pd.MultiIndex.from_tuples([(mgg_db, album_id)])

        return stat_df
    
    return stat_df


def _get_results(
    sample: Group,
    min_dist_from_edge: int,
    max_aspect_ratio: float,
    filter_with_erythroblast_quality: bool,
    filter_with_granulocyte_quality: bool,
    filter_artefacts: bool,
    filter_unknowns: bool,
    filter_with_quality: bool,
    min_megakaryocyte_size: tuple,
) -> DataFrame:
    """
    Extracts cell results for one sample from the HDF5 file.
    """
    
    result_df = extract_cell_results(sample)

    # Apply filters
    result_df = filter_cells(
        result_df,
        sample,
        min_dist_from_edge,
        max_aspect_ratio,
        filter_with_erythroblast_quality,
        filter_with_granulocyte_quality,
        filter_artefacts,
        filter_unknowns,
        filter_with_quality,
        min_megakaryocyte_size,
    )

    return result_df


def read_hdf5_results(
    hdf5_path: str,
    mgg_db: str,
    album_id: str,
    aggregate: str | bool = True,
    minimum_to_aggregate: int = 30,
    min_dist_from_edge: int = 0,
    max_aspect_ratio: float = np.inf,
    filter_with_erythroblast_quality: bool = True,
    filter_with_granulocyte_quality: bool = False,
    filter_artefacts: bool = True,
    filter_unknowns: bool = True,
    filter_with_quality: bool = False,
    min_megakaryocyte_size: tuple = (0.146, 0.122)
) -> DataFrame:
    """
    Reads cell results for one sample from the HDF5 file.

    Args:
        hdf5_path (str): Path to the HDF5 file.
        mgg_db (str): MGG database name.
        album_id (str): album_id.
        aggregate (str | bool): If True, aggregate results, if "proportions", aggregate results,
            if "all", aggregate results and additionally return raw counts and extra proportions, if False do not aggregate, in this case
            note that all results are inferred for all cells regardless of if they are in the target domain.
        minimum_to_aggregate (int): Minimum number of cells required to aggregate results. Specifically, when considering a division, the number of samples in the denominator much be at least this.
        min_dist_from_edge (int): Cells closer than this distance to the image edge will be filtered out.
        max_aspect_ratio (float): Cells with an aspect ratio larger than this value will be filtered out.
        filter_with_erythroblast_quality (bool): If True, filter low quality erythroblasts.
        filter_with_granulocyte_quality (bool): If True, filter low quality granulocytes.
        filter_artefacts (bool): If True, filter artefacts.
        filter_unknowns (bool): If True, filter unknown cells.
        filter_with_quality (bool): If True, filter cells with additional cell statistics-based models.
        min_megakaryocyte_size (tuple): (width, height) Megakaryocytes smaller than this size will be filtered out. The units are relative to the image sizes.

    Returns:
        DataFrame: if aggregate is True, "proportions" or "all", a single-row DataFrame
            with the aggregated results. Otherwise a DataFrame with raw results for each cell within the sample.
    """

    with open(hdf5_path, "rb", buffering=0) as file, h5py.File(file) as f:
        group = f[f"{mgg_db}/{album_id}"]

        if isinstance(group, Group):
            result_df = _get_results(
                group,
                min_dist_from_edge,
                max_aspect_ratio,
                filter_with_erythroblast_quality,
                filter_with_granulocyte_quality,
                filter_artefacts,
                filter_unknowns,
                filter_with_quality,
                min_megakaryocyte_size,
            )
        else:
            raise ValueError("Invalid group in HDF5 file. Expected a Group object. Check the mgg_database and album_id?")

    if aggregate not in [True, False, "proportions", "all"]:
        raise ValueError(
            "Invalid aggregate type. Expected True, False, 'proportions', or 'all'."
        )

    if aggregate is True or aggregate == "proportions":
        # Only return features in terms of proportions
        agg_results = aggregate_results(
            result_df,
            minimum_to_aggregate,
        )

        result_df = agg_results.to_frame().T
        result_df.index = pd.MultiIndex.from_tuples([(mgg_db, album_id)])

        return result_df
    elif aggregate == "all":
        # Also return raw counts
        agg_results = aggregate_results(
            result_df,
            minimum_to_aggregate,
        )

        raw_results = get_result_counts(result_df)

        result_ser = pd.concat([agg_results, raw_results])
        result_df = result_ser.to_frame().T
        result_df.index = pd.MultiIndex.from_tuples([(mgg_db, album_id)])

        return result_df
    else:
        # Do not aggregate
        return result_df