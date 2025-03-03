from pandas import DataFrame
import numpy as np
import pandas as pd
import onnxruntime as rt
from h5py import File

from constants import (
    CELL_LABEL_MAP,
    CELL_TYPES_TO_FILTER,
    ERYTHROBLAST_QUALITY_LABEL_MAP,
    GRANULOCYTE_QUALITY_LABEL_MAP,
    QUALITY_FILTER_STATISTICS,
    QUALITY_FILTER_RESULTS,
    CELL_TYPE_TO_MODEL,
)

CELL_MODEL_PATH = "quality_models"

def _detection_box_near_edge_filter(
    data: File,
    results: DataFrame,
    min_dist_from_edge: int = 0,
) -> DataFrame:
    """
    Find cells that are too close to the image edge.
    """

    # Extract bounding boxes
    boxes = data["DET-BOX"][:]  # Shape: [Number of detections, 4 (x0, y0, x1, y1)]

    # Extract image sizes (one image size per detection)
    img_sizes = data["IMGSIZE"][:]  # Shape: [Number of samples, 2 (height, width)]

    # Calculate the distances to the image borders
    left_border_distances = boxes[:,0]
    right_border_distances = img_sizes[:,1] - boxes[:,2]
    top_border_distances = boxes[:,1]
    bottom_border_distances = img_sizes[:,0] - boxes[:,3]

    # Check that all distances to the borders are greater than the margin
    keep = (
        (left_border_distances >= min_dist_from_edge)
        & (right_border_distances >= min_dist_from_edge)
        & (top_border_distances >= min_dist_from_edge)
        & (bottom_border_distances >= min_dist_from_edge)
    )

    results["KEEPVAL"] = results["KEEPVAL"] & keep

    return results

def _aspect_ratio_filter(
    data: File,
    results: DataFrame,
    max_aspect_ratio: float = np.inf,
) -> DataFrame:
    """
    Find cells whose aspect_ratio is too large.

    Note: Cells with an aspect ratio of over than 3 were discarded before storing
    the data into the HDF5-file.
    """

    boxes = data["DET-BOX"][:]

    widths = boxes[:,2] - boxes[:,0]
    heights = boxes[:,3] - boxes[:,1]

    keep = (widths / heights <= max_aspect_ratio) & (
        heights / widths <= max_aspect_ratio
    )
    results["KEEPVAL"] = results["KEEPVAL"] & keep
    return results

def _low_quality_erythroblast_filter(results: DataFrame) -> DataFrame:
    """
    Find low-quality erythroblasts. Low-quality erythroblasts should always be removed
    before calculating the cell differential count.
    """

    keep = (results["LOW-QUA-ERY"] == ERYTHROBLAST_QUALITY_LABEL_MAP[0]) | results[
        "LOW-QUA-ERY"
    ].isna()

    results["KEEPVAL"] = results["KEEPVAL"] & keep
    results = results.drop(columns="LOW-QUA-ERY")
    return results

def _low_quality_granulocyte_filter(results: DataFrame) -> DataFrame:
    """
    Find low-quality granulocytes. Low-quality granulocytes should be removed
    before calculating the cell statistics, but can be included for cell differential
    count.
    """

    keep = (results["LOW-QUA-GRA"] == GRANULOCYTE_QUALITY_LABEL_MAP[0]) | results[
        "LOW-QUA-GRA"
    ].isna()

    results["KEEPVAL"] = results["KEEPVAL"] & keep
    results = results.drop(columns="LOW-QUA-GRA")
    return results

def _include_artefacts(results: DataFrame) -> DataFrame:
    keep = results["CLS-LBL"] == CELL_LABEL_MAP[0]
    results["KEEPVAL"] = results["KEEPVAL"] | keep
    return results

def _filter_statistics(
    data: File,
    results: DataFrame,
):
    cell_indices_to_keep = results["DET-IDX"]

    # Models take inputs in the order of ENTIRE, CYTOPL, NUCLEI, RESULTS
    # Further order in the QUALITY_FILTER_STATISTICS and QUALITY_FILTER_RESULTS
    # For colors, all 3 are used in order, for glcm, all 5 are used in order
    input_stats = QUALITY_FILTER_STATISTICS
    input_results = QUALITY_FILTER_RESULTS
    model_path = CELL_MODEL_PATH

    # Extract features
    input_dict = {}
    for component, stats in input_stats.items():
        cell_indices = data["STATISTICS"][component]["DET-IDX"]
        keep = np.isin(cell_indices, cell_indices_to_keep)

        for stat in stats:
            stat_values = data["STATISTICS"][component][stat][keep]

            if "glcm" in stat:
                for i in np.arange(5):
                    input_dict[f"{component}-{stat}-{i}"] = stat_values[:,i]
            elif "color" in stat:
                for i in np.arange(3):
                    input_dict[f"{component}-{stat}-{i}"] = stat_values[:,i]
            else:
                input_dict[f"{component}-{stat}"] = stat_values

    input_df = pd.DataFrame(input_dict, index=cell_indices_to_keep)

    # Extract results
    for res in input_results:
        res_values = data["RESULTS"][res][cell_indices_to_keep]
        input_df[res] = res_values
    
    drop_dict = {}
    for ct in CELL_TYPES_TO_FILTER:
        ct_ind = results[results["CLS-LBL"] == ct]["DET-IDX"]
        ct_df = input_df.loc[ct_ind,:]

        # Predict bad/good quality for ct_df
        sess = rt.InferenceSession(
            f"{model_path}/{CELL_TYPE_TO_MODEL[ct]}_quality_model.onnx",
            providers=["CPUExecutionProvider"],
        )

        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name

        pred_onx = sess.run([label_name], {input_name: ct_df.values.astype(np.float32)})[0]

        bad_ind = ct_ind[pred_onx == 0]
        results["KEEPVAL"] = results["KEEPVAL"] & ~np.isin(results["DET-IDX"], bad_ind)
        
        drop_dict[f"{ct}_dropped"] = bad_ind.shape[0]
        drop_dict[f"{ct}_total"] = ct_df.shape[0]

    return results

def _filter_multinuclear(
    data: File,
    results: DataFrame,
):
    # Filter multinuclear cells
    nuclei_indices = data["STATISTICS/NUCLEI"]["DET-IDX"]
    nuclei_indices, counts = np.unique(nuclei_indices, return_counts=True)
    mononuclear_cell_indices = nuclei_indices[counts == 1]

    keep = np.isin(results["DET-IDX"], mononuclear_cell_indices)
    results["KEEPVAL"] = results["KEEPVAL"] & keep

    return results

def filter_cells(
    results: DataFrame,
    data: File,
    min_dist_from_edge: int = 0,
    max_aspect_ratio: float = np.inf,
    filter_with_erythroblast_quality = True,
    filter_with_granulocyte_quality = True,
    keep_artefacts: bool = False,
    keep_unknowns: bool = False,
    filter_with_keepval: bool = True,
    filter_with_quality: bool = False,
    filter_out_multinuclear: bool = False
) -> DataFrame:

    if not filter_with_keepval:
        results["KEEPVAL"] = True
        return results

    # Defaulting to removing all artefacts.
    if keep_artefacts:
        keep = results["CLS-LBL"] == CELL_LABEL_MAP[0]
        results["KEEPVAL"] = results["KEEPVAL"] | keep
    else:
        keep = results["CLS-LBL"] != CELL_LABEL_MAP[0]
        results["KEEPVAL"] = results["KEEPVAL"] & keep

    # filter unknowns
    if not keep_unknowns:
        keep = results["CLS-LBL"] != CELL_LABEL_MAP[-1]
        results["KEEPVAL"] = results["KEEPVAL"] & keep

    # find cell detections that are too close to the image edge
    results = _detection_box_near_edge_filter(data, results, min_dist_from_edge)

    # find cells whose aspect_ratio is too large
    results = _aspect_ratio_filter(data, results, max_aspect_ratio)

    # find low-quality erythroblasts
    if filter_with_erythroblast_quality:
        results = _low_quality_erythroblast_filter(results)

    # find low-quality granulocytes
    if filter_with_granulocyte_quality:
        results = _low_quality_granulocyte_filter(results)

    # find multinuclear cells
    if filter_out_multinuclear:
        results = _filter_multinuclear(data, results)

    # filter cells with logistic regression-based quality models, to be used before statistic aggregation.
    if filter_with_quality:
        # apply all filters, then applying the extra quality models is easier
        results = results[results["KEEPVAL"] == True]
        results = _filter_statistics(data, results)

    results = results[results["KEEPVAL"] == True].drop(columns="KEEPVAL")
    return results
