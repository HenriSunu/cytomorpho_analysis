import warnings

import pandas as pd
import numpy as np
import numpy.typing as npt

from pandas import DataFrame
from h5py import Group

from cell_pipeline.constants import (
    CELL_LABEL_MAP,
    CELL_TYPES_TO_FILTER,
    CELL_TYPE_TO_MODEL,
    QUALITY_FILTER_STATISTICS,
    QUALITY_FILTER_RESULTS,
    CELL_MODEL_PATH,
    ERYTHROBLAST_QUALITY_CLASSES,
    GRANULOCYTE_QUALITY_CLASSES,
)

def _get_filter_input_data(
    data: Group
) -> DataFrame:
    """
    Get the input data for the filtering models.
    """

    filter_df = pd.DataFrame()
    for result in QUALITY_FILTER_RESULTS:
        filter_df[result] = data[result][:]

    for component in QUALITY_FILTER_STATISTICS:
        for statistic in QUALITY_FILTER_STATISTICS[component]:
            filter_df[f"{component}_{statistic}"] = data[f"STATISTICS/{component}"][statistic][:]
    
    return filter_df


def _filter_statistics(
    data: Group,
    cell_df: DataFrame,
) -> npt.NDArray[np.bool_]:
    """
    Apply extra models to filter cells based on quality.

    Filtered cells are those for which calculating geometric statistics does not make sense.
    For example, cells squished between other cells
    """

    # Bit unorthodox, import in the function so that
    # onnxruntime doesn't become a dependency when
    # this extra filtering is not used.
    import onnxruntime as rt

    filter_df = _get_filter_input_data(data)
    filter_out = np.zeros(len(cell_df), dtype=bool)

    for ct in CELL_TYPES_TO_FILTER:
        ct_ind = (cell_df["CLS-LBL"] == ct).to_numpy().nonzero()[0]
        ct_df = filter_df.loc[ct_ind,:]

        # Predict bad/good quality for cell type
        if CELL_MODEL_PATH == "":
            raise UserWarning("Provide a path to .onnx models in constants.py when filter_with_quality=True")

        ct_model = CELL_TYPE_TO_MODEL[ct]
        if ct_model != ct:
            warnings.warn(
                f"Applying {ct_model} model to filter {ct} cells"
            )

        sess = rt.InferenceSession(
            f"{CELL_MODEL_PATH}/{ct}_quality_model.onnx",
            providers=["CPUExecutionProvider"]
        )

        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name

        pred_onx = sess.run([label_name], {input_name: ct_df.values.astype(np.float32)})[0]
        bad_ind = ct_ind[pred_onx == 0]
        filter_out[bad_ind] = True

    return filter_out


def _filter_small_megakaryocytes(
    data: Group,
    cls_lbl: npt.NDArray[np.str_],
    min_size: tuple,
) -> npt.NDArray[np.bool_]:
    """
    Filter out megakaryocytes that are too small.
    """

    # extract bounding boxes
    boxes = data["DET-BOX"][:]  # Shape: [Number of detections, 4 (x0, y0, x1, y1)]

    # extract image sizes
    image_sizes = data["IMGSIZE"][:]

    # calculate width and height of the bounding boxes
    widths = (boxes[:, 2] - boxes[:, 0]) / image_sizes[:, 0]
    heights = (boxes[:, 3] - boxes[:, 1]) / image_sizes[:, 1]

    # Cells that are smaller
    small_cells = (widths < min_size[0]) & (heights < min_size[1])

    # and are megakaryocytes
    megas = cls_lbl == "Megakaryocytes"
    filter_out = small_cells & megas

    return filter_out


def _filter_erythroblast_quality(
    data: Group,
    cls_lbl: npt.NDArray[np.str_],
) -> npt.NDArray[np.bool_]:

    low_quality = data["LOW-QUALITY"][:]
    erythroblasts = np.isin(cls_lbl, ERYTHROBLAST_QUALITY_CLASSES)

    filter_out = (low_quality > 0.5) & erythroblasts

    return filter_out

def _filter_granulocyte_quality(
    data: Group,
    cls_lbl: npt.NDArray[np.str_],
) -> npt.NDArray[np.bool_]:
    
    low_quality = data["LOW-QUALITY"][:]
    granulocytes = np.isin(cls_lbl, GRANULOCYTE_QUALITY_CLASSES)

    filter_out = (low_quality > 0.5) & granulocytes

    return filter_out

def _filter_aspect_ratio(
    data: Group,
    max_aspect_ratio: float
) -> npt.NDArray[np.bool_]:
    """
    Find cells whose aspect ratio is too large. (In terms of the bounding box)
    """

    # extract bounding boxes
    boxes = data["DET-BOX"][:]  # Shape: (N, 4 (x0, y0, x1, y1))

    # calculate width and height of the bounding boxes
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]

    filter_out = (widths / heights > max_aspect_ratio) | (heights / widths > max_aspect_ratio)

    return filter_out


def _filter_edge_boxes(
    data: Group,
    min_dist_from_edge: int
) -> npt.NDArray[np.bool_]:
    """
    Find cells that are too close to the edge of the image.
    """

    # Extract bounding boxes
    boxes = data["DET-BOX"][:]  # Shape: (N, 4 (x0, y0, x1, y1))

    # Extract image sizes (one image size per detection)
    img_sizes = data["IMGSIZE"][:]  # Shape: (N, 2 (width, height))

    # Calculate the distances to the image borders
    dist_arr = np.array(
        [
            boxes[:, 0], #left
            img_sizes[:, 0] - boxes[:, 2], #right
            boxes[:, 1], #top
            img_sizes[:, 1] - boxes[:, 3], #bottom
        ]
    )

    # Check for distances to the borders that are too small
    filter_out = np.min(dist_arr, axis=0) < min_dist_from_edge
    return filter_out


def filter_cells(
    cell_df: DataFrame,
    data: Group,
    min_dist_from_edge: int = 0,
    max_aspect_ratio: float = np.inf,
    filter_with_erythroblast_quality: bool = True,
    filter_with_granulocyte_quality: bool = False,
    filter_artefacts: bool = True,
    filter_unknowns: bool = True,
    filter_with_quality: bool = False,
    min_megakaryocyte_size: tuple = (0.146, 0.122),
) -> DataFrame:
    """
    Filter individual cells before calculating cell differentials or statistics for samples.
    """

    filter_mask = np.zeros(len(cell_df), dtype=bool)
    cls_lbl = np.array([CELL_LABEL_MAP[x] for x in data["CLS-LBL"]])
    
    # Filter out artefacts
    if filter_artefacts:
        filter_mask |= cls_lbl == "Artefacts"
    # Filter out unknonws
    if filter_unknowns:
        filter_mask |= cls_lbl == "Unknowns"
    # Filter out cells too close to the 100x image edge
    if min_dist_from_edge > 0:
        filter_mask |= _filter_edge_boxes(data, min_dist_from_edge)

    # Filter bounding boxes with a large aspect ratio
    if max_aspect_ratio != np.inf:
        filter_mask |= _filter_aspect_ratio(data, max_aspect_ratio)

    # Filter low quality eyrthroblasts
    if filter_with_erythroblast_quality:
        filter_mask |= _filter_erythroblast_quality(data, cls_lbl)
    
    # Filter low quality granulocytes
    if filter_with_granulocyte_quality:
        filter_mask |= _filter_granulocyte_quality(data, cls_lbl)

    # Filter out small megakaryocytes
    if min_megakaryocyte_size:
        filter_mask |= _filter_small_megakaryocytes(data, cls_lbl, min_megakaryocyte_size)
    
    if filter_with_quality:
        # Potential for noticeable speedup, only apply .onnx models to
        # cell that were not filtered in previous steps.
        filter_mask |= _filter_statistics(data, cell_df)
        cell_df = cell_df.loc[~filter_mask,:]
    else:
        # Apply all filters
        cell_df = cell_df.loc[~filter_mask,:]

    return cell_df
