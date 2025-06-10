import h5py
import argparse

import numpy as np
import pandas as pd

from cell_pipeline.reader import (
    read_hdf5_statistics,
    read_hdf5_results
)

from pandas import DataFrame
from tqdm import tqdm

"""
Wrapper functions for running the HDF5-reader with some good default arguments.
"""

def get_aggregated_data(
    hdf5_path: str,
    mgg_db: str | None = None,
    album_id: str | None = None,
    min_dist_from_edge_stats: int = 1,
    min_dist_from_edge_results: int = 0
) -> DataFrame:
    """
    Reads cell results and statistics from HDF5 file and calculates aggregate
    features that can be used for modeling. 
    
    Uses the default arguments. If more control is needed, call read_hdf5_statistics and read_hdf5_results directly.

    Args:
        hdf5_path (str): Path to a HDF5-file, if mgg_database or album_id is None, then all samples from the HDF5-file are extracted. If album_id and mgg_database are provided, then only one sample is extracted.
        mgg_db (str): mgg_database
        album_id (str): album_id
        min_dist_from_edge_stats (int): Cells closer than this distance to the image edge will be filtered out before calculating statistics.
        min_dist_from_edge_results (int): Cells closer than this distance to the image edge will be filtered out before calculating results or the cell differential.
        
    Returns:
        DataFrame: DataFrame with the extracted features either for one sample or all samples.
    """

    if mgg_db is None and album_id is None:
        # Read all samples
        with open(hdf5_path, "rb", buffering=0) as file, h5py.File(file) as f:
            mgg_dbs = list(f.keys())
            samples = []
            for mgg_db in mgg_dbs:
                samples.extend([(mgg_db, aid) for aid in list(f[mgg_db].keys())])
    elif mgg_db is not None and album_id is None:
        # Read all samples from mgg_db
        with open(hdf5_path, "rb", buffering=0) as file, h5py.File(file) as f:
            samples = [(mgg_db, aid) for aid in list(f[mgg_db].keys())]
    else:
        # Read one sample
        samples = [(mgg_db, album_id)]

    result_df = pd.DataFrame()
    for mgg_db, aid in tqdm(samples, desc="Reading samples", unit="sample", miniters=10):
        stats = read_hdf5_statistics(
            hdf5_path,
            mgg_db=mgg_db,
            album_id=aid,
            aggregate=True,
            minimum_to_aggregate=5, # Using a lower value than in the manuscript for the sake of example
            min_dist_from_edge=min_dist_from_edge_stats,
            max_aspect_ratio=np.inf,
            filter_with_erythroblast_quality=True,
            filter_with_granulocyte_quality=True,
            filter_artefacts=True,
            filter_unknowns=True,
            filter_with_quality=True,
            min_megakaryocyte_size=(0.146, 0.122)
        )

        results = read_hdf5_results(
            hdf5_path,
            mgg_db=mgg_db,
            album_id=aid,
            aggregate=True,
            minimum_to_aggregate=5, # Using a lower value than in the manuscript for the sake of example
            min_dist_from_edge=min_dist_from_edge_results,
            max_aspect_ratio=np.inf,
            filter_with_erythroblast_quality=True,
            filter_with_granulocyte_quality=False,
            filter_artefacts=True,
            filter_unknowns=True,
            filter_with_quality=False,
            min_megakaryocyte_size=(0.146, 0.122)
        )

        sample_df = pd.concat([stats, results], axis=1)
        result_df = pd.concat([result_df, sample_df], axis=0)

    return result_df


def get_raw_data(
    hdf5_path: str,
    mgg_db: str,
    album_id: str,
    strict_filtering: bool = False,
    min_dist_from_edge_stats: int = 1,
    min_dist_from_edge_results: int = 0
) -> tuple[DataFrame, DataFrame]:
    """
    Reads all statistics and results from a HDF5 file and returns the raw cell-wise data.
    This can be useful for debugging, and for exploring the range of cytomorphologies
    within a sample.

    Can only be run for one sample at a time due to dataset size.

    Uses the default arguments. If more control is needed, call read_hdf5_statistics and read_hdf5_results directly.

    Args:
        hdf5_path (str): Path to a HDF5-file
        mgg_db (str): mgg_database
        album_id (str): album_id
        strict_filtering (bool): If False (default), strict filtering is applied before calculating statistics, with more relaxed filtering for results. This is the default behavior of the aggregation process, but output dataframes will definitely be of different sizes. If True, strict filtering is applied for both statistics and results.
        min_dist_from_edge_stats (int): Cells closer than this distance to the image edge will be filtered out before calculating statistics.
        min_dist_from_edge_results (int): Cells closer than this distance to the image edge will be filtered out before calculating results or the cell differential.
       
    Returns:
        (DataFrame: Cell-wise statistics, DataFrame: Cell-wise results)
    """

    stats = read_hdf5_statistics(
        hdf5_path,
        mgg_db=mgg_db,
        album_id=album_id,
        aggregate=False,
        minimum_to_aggregate=5, # Using a lower value than in the manuscript for the sake of example
        min_dist_from_edge=min_dist_from_edge_stats,
        max_aspect_ratio=np.inf,
        filter_with_erythroblast_quality=True,
        filter_with_granulocyte_quality=True,
        filter_artefacts=True,
        filter_unknowns=True,
        filter_with_quality=True,
        min_megakaryocyte_size=(0.146, 0.122)
    )

    results = read_hdf5_results(
        hdf5_path,
        mgg_db=mgg_db,
        album_id=album_id,
        aggregate=False,
        minimum_to_aggregate=5, # Using a lower value than in the manuscript for the sake of example
        min_dist_from_edge=min_dist_from_edge_results,
        max_aspect_ratio=np.inf,
        filter_with_erythroblast_quality=True,
        filter_with_granulocyte_quality=strict_filtering,
        filter_artefacts=True,
        filter_unknowns=True,
        filter_with_quality=strict_filtering,
        min_megakaryocyte_size=(0.146, 0.122)
    )
    
    return stats, results