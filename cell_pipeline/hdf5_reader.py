import h5py
import numpy as np
import pandas as pd
from pandas import DataFrame
from h5py import Dataset, File
from scipy.stats import iqr
from skimage.color import rgb2gray

from cell_filters import filter_cells
from constants import (
    SAMPLE_ID_COLUMN_NAME,
    CELL_LABEL_MAP,
    CELL_LINEAGES,
    ERYTHROBLAST_QUALITY_LABEL_MAP,
    GRANULOCYTE_QUALITY_LABEL_MAP,
    GRANULOCYTE_QUALITY_INCLUDED_CELLS,
    PROMYELOCYTE_CLASS_LABEL_MAP,
    PROMYELOCYTE_CLASS_INCLUDED_CELLS,
    MGK_CLS_LABEL_MAP,
    ERYTHROBLAST_CLASS_LABEL_MAP,
    ERYTHROBLAST_CLASS_INCLUDED_CELLS,
    GRANULARITY_LABEL_MAP,
    GRANULARITY_CLASS_INCLUDED_CELLS,
    EOSINOPHIL_MAT_LABEL_MAP,
    EOSINOPHIL_CLASS_INCLUDED_CELLS,
    STATISTICS_TO_EXTRACT,
    STATISTICS_TO_ENGINEER,
    COLOR_CHANNEL_LABELS,
    STATISTIC_COMPONENT_LABEL_MAP,
    MONOCYTE_SUBCLASS_MAP,
    RAW_RESULTS_TO_EXTRACT,
    CELL_TYPES_TO_FILTER
)

def _get_cell_lineage_specific_column_name_for_vacuolization(cell_lineage):
    return f"Vacuolated {cell_lineage}"

def _get_cell_lineage_specific_column_name_for_multinuclear_results(cell_lineage):
    return f"Multinuclear {cell_lineage}"

def get_target_columns() -> list:
    """
    Get column names required in the final dataframe.
    """

    columns = [SAMPLE_ID_COLUMN_NAME] + list(CELL_LABEL_MAP.values())

    label_maps_to_add = [
        GRANULOCYTE_QUALITY_LABEL_MAP,
        PROMYELOCYTE_CLASS_LABEL_MAP,
        MGK_CLS_LABEL_MAP,
        ERYTHROBLAST_CLASS_LABEL_MAP,
        GRANULARITY_LABEL_MAP,
        EOSINOPHIL_MAT_LABEL_MAP,
    ]
    for label_map in label_maps_to_add:
        labels = list(label_map.values())
        columns.extend(labels)

    # add vacuolization and multinuclear labels to the column-list
    for cell_lineage in CELL_LINEAGES.keys():
        if cell_lineage not in ["Living cells", "Artefacts", "Unknowns"]:
            columns.append(
                _get_cell_lineage_specific_column_name_for_vacuolization(cell_lineage)
            )
            columns.append(
                _get_cell_lineage_specific_column_name_for_multinuclear_results(
                    cell_lineage
                )
            )
    return columns

def _extract_class_labels(
    group: Dataset,
    result_table: DataFrame,
    unknown_confidence_threshold: float,
) -> DataFrame:
    """
    Extract main cell classification results.
    """

    # extract cell labels and probabilities
    int_labels = group["CLS-LBL"]
    probabilities = group["CLS-PRB"][np.arange(len(group)), int_labels]

    vectorized_map = np.vectorize(lambda x: CELL_LABEL_MAP[x])
    string_labels = vectorized_map(int_labels)

    # change cells with low probability value to 'Unknowns'
    unknown_cells = probabilities < unknown_confidence_threshold
    string_labels[unknown_cells] = "Unknowns"

    # create new sub-result table for cell labels
    sub_result_table = pd.DataFrame(data={"CLS-LBL": string_labels})

    # subclassify monocytes to monocytes, promonocytes and macrophages, if monocytes exist
    if np.any(sub_result_table["CLS-LBL"] == "Monocytes"):
        monocyte_results = group["MON-MAT"][sub_result_table["CLS-LBL"] == "Monocytes",:]
        vectorized_map = np.vectorize(lambda x: MONOCYTE_SUBCLASS_MAP[x])
        monocyte_classes = vectorized_map(np.argmax(monocyte_results, axis=1))
        sub_result_table.loc[sub_result_table["CLS-LBL"] == "Monocytes", "CLS-LBL"] = monocyte_classes

    result_table = pd.concat([result_table, sub_result_table], axis=1)
    return result_table

def _extract_erythroblast_quality_results(
    group: Dataset,
    result_table: DataFrame,
    confidence_threshold: float,
) -> DataFrame:
    """
    Extract erythroblast quality results.
    """
    included_cells = ["Erythroblasts"]

    sub_result_table = pd.DataFrame(
        data={"LOW-QUA-ERY": np.array(group["QUA-ERY"] > confidence_threshold)}
    )

    result_table = pd.concat([result_table, sub_result_table], axis=1)

    # set values to NaN, if cell was not an erythroblast
    result_table.loc[
        ~result_table["CLS-LBL"].isin(included_cells), "LOW-QUA-ERY"
    ] = np.nan

    result_table["LOW-QUA-ERY"] = result_table["LOW-QUA-ERY"].map(
        {
            True: ERYTHROBLAST_QUALITY_LABEL_MAP[1],
            False: ERYTHROBLAST_QUALITY_LABEL_MAP[0],
        }
    )
    return result_table

def _extract_granulocyte_quality_results(
    group: Dataset,
    result_table: DataFrame,
    confidence_threshold: float,
) -> DataFrame:
    """
    Extract granularity quality results.
    """
    sub_result_table = pd.DataFrame(
        data={"LOW-QUA-GRA": np.array(group["QUA-GRA"] > confidence_threshold)}
    )

    result_table = pd.concat([result_table, sub_result_table], axis=1)

    # set values to NaN, if cell was not in the target domain
    result_table.loc[
        ~result_table["CLS-LBL"].isin(GRANULOCYTE_QUALITY_INCLUDED_CELLS), "LOW-QUA-GRA"
    ] = np.nan

    result_table["LOW-QUA-GRA"] = result_table["LOW-QUA-GRA"].map(
        {
            True: GRANULOCYTE_QUALITY_LABEL_MAP[1],
            False: GRANULOCYTE_QUALITY_LABEL_MAP[0],
        }
    )

    return result_table

def _extract_promyelocyte_results(
    group: Dataset,
    result_table: DataFrame,
    confidence_threshold: dict,
) -> DataFrame:
    """
    Extract promyelocyte results.
    """

    sub_result_table = pd.DataFrame(
        data={
            "APL-ABN": np.array(group["APL-ABN"] > confidence_threshold["APL-ABN"]),
            "APL-HYP": np.array(group["APL-HYP"] > confidence_threshold["APL-HYP"]),
            "APL-INC": np.array(group["APL-INC"] > confidence_threshold["APL-INC"]),
        }
    )

    result_table = pd.concat([result_table, sub_result_table], axis=1)

    result_table["APL-ABN"] = result_table["APL-ABN"].map(
        {True: PROMYELOCYTE_CLASS_LABEL_MAP["APL-ABN"], False: np.nan}
    )
    result_table["APL-HYP"] = result_table["APL-HYP"].map(
        {True: PROMYELOCYTE_CLASS_LABEL_MAP["APL-HYP"], False: np.nan}
    )
    result_table["APL-INC"] = result_table["APL-INC"].map(
        {True: PROMYELOCYTE_CLASS_LABEL_MAP["APL-INC"], False: np.nan}
    )

    # set values to NaN, if cell was not in the target domain
    result_table.loc[
        ~result_table["CLS-LBL"].isin(PROMYELOCYTE_CLASS_INCLUDED_CELLS),
        ["APL-ABN", "APL-HYP", "APL-INC"],
    ] = np.nan

    return result_table

def _extract_megakaryocyte_results(
    group: Dataset,
    result_table: DataFrame
) -> DataFrame:
    """
    Extract megakaryocyte classification and quality results.
    """

    mgk_class_results = np.argmax(group["MGK-CLS"], axis=1)

    mgk_class_mapper = np.vectorize(lambda x: MGK_CLS_LABEL_MAP[x])
    mgk_class_labels = mgk_class_mapper(mgk_class_results)

    sub_result_table = pd.DataFrame(
        data={
            "MGK-CLS": mgk_class_labels,
        }
    )

    result_table = pd.concat([result_table, sub_result_table], axis=1)

    # set values to NaN, if cell type not megakaryocyte
    result_table.loc[
        ~result_table["CLS-LBL"].isin(["Megakaryocytes"]), ["MGK-CLS", "MGK-QUA"]
    ] = np.nan

    return result_table

def _extract_vacuolization_results(
    group: Dataset,
    result_table: DataFrame,
    confidence_threshold: float,
) -> DataFrame:
    """
    Extract vacuolization results. Vacuolization is calculated at cell lineage level.
    """

    sub_result_table = pd.DataFrame(
        data={
            "VACUOLI": np.array(group["VACUOLI"] > confidence_threshold),
        }
    )

    result_table = pd.concat([result_table, sub_result_table], axis=1)

    replacement_mapping = {}
    for cell_lineage, cell_types in CELL_LINEAGES.items():
        if cell_lineage not in ["Living cells", "Artefacts", "Unknowns"]:
            for cell_type in cell_types:
                replacement_mapping[
                    cell_type
                ] = _get_cell_lineage_specific_column_name_for_vacuolization(
                    cell_lineage
                )

    result_table["VACUOLI"] = np.where(
        result_table["VACUOLI"],
        result_table["CLS-LBL"].map(replacement_mapping),
        np.nan,
    )

    return result_table

def _extract_erythroblast_results(
    group: Dataset,
    result_table: DataFrame,
    confidence_threshold: dict
) -> DataFrame:
    """
    Extract erythroblast results.
    """

    sub_result_table = pd.DataFrame(
        data={
            "MDS-CYT": np.array(group["MDS-CYT"] > confidence_threshold["MDS-CYT"]),
            "MDS-DYS": np.array(group["MDS-DYS"] > confidence_threshold["MDS-DYS"]),
            "MDS-MUL": np.array(group["MDS-MUL"] > confidence_threshold["MDS-MUL"]),
            "MDS-MEG": np.array(group["MDS-MEG"] > confidence_threshold["MDS-MEG"]),
        }
    )

    result_table = pd.concat([result_table, sub_result_table], axis=1)

    # set values to NaN, if cell type not erythroblast
    result_table.loc[
        ~result_table["CLS-LBL"].isin(ERYTHROBLAST_CLASS_INCLUDED_CELLS),
        ["MDS-CYT", "MDS-DYS", "MDS-MUL", "MDS-MEG"],
    ] = np.nan

    result_table["MDS-CYT"] = result_table["MDS-CYT"].map(
        {True: ERYTHROBLAST_CLASS_LABEL_MAP["MDS-CYT"], False: np.nan}
    )
    result_table["MDS-DYS"] = result_table["MDS-DYS"].map(
        {True: ERYTHROBLAST_CLASS_LABEL_MAP["MDS-DYS"], False: np.nan}
    )
    result_table["MDS-MUL"] = result_table["MDS-MUL"].map(
        {True: ERYTHROBLAST_CLASS_LABEL_MAP["MDS-MUL"], False: np.nan}
    )
    result_table["MDS-MEG"] = result_table["MDS-MEG"].map(
        {True: ERYTHROBLAST_CLASS_LABEL_MAP["MDS-MEG"], False: np.nan}
    )

    return result_table

def _extract_granularity_results(
    group: Dataset,
    result_table: DataFrame,
) -> DataFrame:
    """
    Extract granularity results.
    """
    granularity_results = np.argmax(group["GRANULA"], axis=1)

    granularity_mapper = np.vectorize(lambda x: GRANULARITY_LABEL_MAP[x])
    granularity_labels = granularity_mapper(granularity_results)

    sub_result_table = pd.DataFrame(data={"GRANULA": granularity_labels})

    result_table = pd.concat([result_table, sub_result_table], axis=1)

    # set values to NaN, if cell was not in the target domain
    result_table.loc[
        ~result_table["CLS-LBL"].isin(GRANULARITY_CLASS_INCLUDED_CELLS), "GRANULA"
    ] = np.nan

    return result_table

def _extract_multinuclear_results(data: File, result_table: DataFrame) -> DataFrame:
    cell_indices = data["DET-IDX"]
    statistics = data["STATISTICS"]

    nucleis = statistics["NUCLEI"]["DET-IDX"]
    indices, counts = np.unique(nucleis, return_counts=True)
    multinuclear_cell_indices = indices[counts > 1]

    bool_array = np.full(cell_indices.shape, False, dtype=object)
    bool_array[multinuclear_cell_indices] = True
    result_table["MULTINUCLEAR"] = bool_array

    replacement_mapping = {}
    for cell_lineage, cell_types in CELL_LINEAGES.items():
        if cell_lineage not in ["Living cells", "Artefacts", "Unknowns"]:
            for cell_type in cell_types:
                replacement_mapping[
                    cell_type
                ] = _get_cell_lineage_specific_column_name_for_multinuclear_results(
                    cell_lineage
                )

    result_table["MULTINUCLEAR"] = np.where(
        result_table["MULTINUCLEAR"],
        result_table["CLS-LBL"].map(replacement_mapping),
        np.nan,
    )

    return result_table

def _extract_eosinophil_results(
    group: Dataset,
    result_table: DataFrame,
    confidence_threshold: float,
) -> DataFrame:
    """
    Extract eosinophil maturity results. 0 = Mature, 1 = Immature.
    """
    sub_result_table = pd.DataFrame(
        data={
            "EOS-MAT": np.array(group["EOS-MAT"] > confidence_threshold)
        }
    )

    result_table = pd.concat([result_table, sub_result_table], axis=1)

    # set values to NaN, if cell type is not Eosinophil
    result_table.loc[
        ~result_table["CLS-LBL"].isin(EOSINOPHIL_CLASS_INCLUDED_CELLS), "EOS-MAT"
    ] = np.nan

    result_table["EOS-MAT"] = result_table["EOS-MAT"].map(
        {
            True: EOSINOPHIL_MAT_LABEL_MAP[1],
            False: EOSINOPHIL_MAT_LABEL_MAP[0],
        }
    )

    return result_table

def _calculate_cell_differential_counts(data: DataFrame) -> DataFrame:
    """
    Calculate cell differential counts for each sample.
    """
    melted_data = data.melt()
    results = melted_data["value"].value_counts().to_frame().T

    return results

def _aggregate_cell_statistics(data, cell_labels, label):

    df = pd.DataFrame({"value": data, "cell_label": cell_labels})

    # Group by 'label' and calculate the median and 95% range of 'value' for each group
    results = {}
    for cell_label, group in df.groupby("cell_label")["value"]:
        # discard unknown cells
        if cell_label != CELL_LABEL_MAP[-1]:
            values = group.values
            results.update(
                {
                    f"{cell_label}_{label}_median": [np.median(values)],
                    f"{cell_label}_{label}_95_range": [iqr(values, rng=(2.5, 97.5))]
                }
            )
    return results

def _engineer_statistics(
    statistics: Dataset,
    keep: list,
    new_statistic: str,
    cell_component: str,
):
    if new_statistic == "nc-ratio":
        en_area = statistics["ENTIRE"]["area"][keep]
        cyt_area = statistics["CYTOPL"]["area"][keep]
        nc_ratio = (en_area - cyt_area)/en_area
        return nc_ratio
    
    if new_statistic == "luminance":
        comp_color = statistics[cell_component]["color-mean"][keep]
        comp_luminance = rgb2gray(comp_color)
        return comp_luminance

def _extract_cell_statistics(
    statistics: Dataset,
    cell_indices_to_keep: list,
    cell_id_to_label_map: dict,
):
    results = {}

    for cell_component, covariates in STATISTICS_TO_EXTRACT.items():
        cell_indices = statistics[cell_component]["DET-IDX"]
        keep = np.isin(cell_indices, cell_indices_to_keep)

        for covariate in covariates:
            if covariate in STATISTICS_TO_ENGINEER[cell_component]:
                # Engineered statistics
                covariate_data = _engineer_statistics(statistics, keep, covariate, cell_component)
            else:
                # Statistics straight from file
                covariate_data = statistics[cell_component][covariate]
                covariate_data = covariate_data[keep]

            cell_indices_to_keep = cell_indices[keep]
            cell_labels = [
                cell_id_to_label_map.get(cell_id) for cell_id in cell_indices_to_keep
            ]

            # glcm statistics have five values, one for each pixel range
            if covariate_data.shape[-1] == 5 and covariate.startswith("glcm"):
                for i in range(covariate_data.shape[-1]):
                    covariate_label = f"{STATISTIC_COMPONENT_LABEL_MAP[cell_component]}-{covariate}-{i+1}"
                    results.update(
                        _aggregate_cell_statistics(
                            covariate_data[:,i].flatten(),
                            cell_labels,
                            covariate_label
                        )
                    )

            # color statistics has three values, one for each color channel (RGB)
            elif covariate_data.shape[-1] == 3 and covariate.startswith("color"):
                for i in range(covariate_data.shape[-1]):
                    covariate_label = f"{STATISTIC_COMPONENT_LABEL_MAP[cell_component]}-{covariate.replace('color', COLOR_CHANNEL_LABELS[i])}"
                    results.update(
                        _aggregate_cell_statistics(
                            covariate_data[:, i].flatten(),
                            cell_labels,
                            covariate_label,
                        )
                    )

            elif covariate_data.ndim == 1:
                covariate_label = (
                    f"{STATISTIC_COMPONENT_LABEL_MAP[cell_component]}-{covariate}"
                )
                results.update(
                    _aggregate_cell_statistics(
                        covariate_data,
                        cell_labels,
                        covariate_label,
                    )
                )

            else:
                raise ValueError(f"Did not expect covariate: {covariate}")

    return pd.DataFrame(data=results)

def _extract_raw_cell_results(
    cell_results: Dataset,
    cell_indices_to_keep: list,
    cell_id_to_label_map: dict,
):
    """
    Extract cell results without aggregation
    """

    results = {}
    for result in RAW_RESULTS_TO_EXTRACT:

        cell_indices = cell_results["DET-IDX"]
        keep = np.isin(cell_indices, cell_indices_to_keep)

        res_vec = cell_results["RESULTS"][result][keep]

        # For results that are not of shape [N,]
        if isinstance(res_vec[0], np.ndarray):
            for i in np.arange(len(res_vec[0])):
                results[f"{result}_{i}"] = [r[i] for r in res_vec]
        else:
            results[result] = cell_results["RESULTS"][result][keep]

    result_df = pd.DataFrame(data=results, index=cell_indices_to_keep)
    # Results are one for one cell, unlike statistics, so we can just get labels outside of the loop
    cell_labels = [
            cell_id_to_label_map.get(cell_id) for cell_id in cell_indices_to_keep
        ]
    result_df["CLS-LBL"] = cell_labels
    return result_df

def _extract_raw_cell_statistics(
    statistics: Dataset,
    cell_indices_to_keep: list,
    cell_id_to_label_map: dict,
):
    """
    Same as _extract_cell_statistics but without aggregating statistics.
    """
    results = {}

    for cell_component, covariates in STATISTICS_TO_EXTRACT.items():
        cell_indices = statistics[cell_component]["DET-IDX"]
        keep = np.isin(cell_indices, cell_indices_to_keep)
        cell_indices_to_keep = cell_indices[keep]

        cell_labels = [
            cell_id_to_label_map.get(cell_id) for cell_id in cell_indices_to_keep
        ]

        for covariate in covariates:
            if covariate in STATISTICS_TO_ENGINEER[cell_component]:
                # Engineered statistics
                covariate_data = _engineer_statistics(statistics, keep, covariate, cell_component)
            else:
                # Statistics straight from file
                covariate_data = statistics[cell_component][covariate]
                covariate_data = covariate_data[keep]

            if covariate_data.shape[-1] == 5 and covariate.startswith("glcm"):
                for i in range(covariate_data.shape[-1]):
                    covariate_label = f"{STATISTIC_COMPONENT_LABEL_MAP[cell_component]}-{covariate}-{i+1}"
                    results[covariate_label] = covariate_data[:,i].flatten()

            elif covariate_data.shape[-1] == 3 and covariate.startswith("color"):
                for i in range(covariate_data.shape[-1]):
                    covariate_label = f"{STATISTIC_COMPONENT_LABEL_MAP[cell_component]}-{covariate}-{i+1}"
                    results[covariate_label] = covariate_data[:,i].flatten()

            elif covariate_data.ndim == 1:
                covariate_label = f"{STATISTIC_COMPONENT_LABEL_MAP[cell_component]}-{covariate}"
                results[covariate_label] = covariate_data
                
            else:
                raise ValueError(f"Did not expect covariate: {covariate}")
    
    result_df = pd.DataFrame(data=results, index=cell_indices_to_keep)

    # As we analyze mononuclear cells, the cell_labels are the same each run
    # of the loop, and we can use the last one.
    result_df["CLS-LBL"] = cell_labels
    return result_df

def extract_sample_differential_counts(
    morph_data: Dataset,
    unknown_confidence_threshold: float = 0.25,
    binary_model_confidence_thresholds: dict = {
        "QUA-ERY": 0.5,
        "QUA-GRA": 0.5,
        "APL": {
            "APL-ABN": 0.5,
            "APL-HYP": 0.5,
            "APL-INC": 0.5,
        },
        "VACUOLI": 0.5,
        "EOS-MAT": 0.5,
        "MDS": {
            "MDS-CYT": 0.5,
            "MDS-DYS": 0.5,
            "MDS-MUL": 0.5,
            "MDS-MEG": 0.5,
        },
    },
    min_dist_from_edge: int = 0,
    max_aspect_ratio: float = np.inf,
    filter_with_erythroblast_quality: bool = True,
    filter_with_granulocyte_quality: bool = False,
    keep_artefacts: bool = False,
    keep_unknowns: bool = False,
    filter_with_keepval: bool = True,
    filter_with_quality: bool = False,
    filter_out_multinuclear: bool = False
) -> DataFrame:

    results = morph_data["RESULTS"]

    result_table = pd.DataFrame(data={"KEEPVAL": results["KEEPVAL"]})

    result_table = _extract_class_labels(
        results, result_table, unknown_confidence_threshold
    )
    
    result_table = _extract_erythroblast_quality_results(
        results, result_table, binary_model_confidence_thresholds["QUA-ERY"]
    )
    result_table = _extract_granulocyte_quality_results(
        results, result_table, binary_model_confidence_thresholds["QUA-GRA"]
    )
    result_table = _extract_promyelocyte_results(
        results, result_table, binary_model_confidence_thresholds["APL"]
    )
    result_table = _extract_megakaryocyte_results(results, result_table)
    result_table = _extract_vacuolization_results(
        results, result_table, binary_model_confidence_thresholds["VACUOLI"]
    )
    result_table = _extract_erythroblast_results(
        results, result_table, binary_model_confidence_thresholds["MDS"]
    )
    result_table = _extract_granularity_results(results, result_table)
    result_table = _extract_multinuclear_results(morph_data, result_table)
    result_table = _extract_eosinophil_results(
        results, result_table, binary_model_confidence_thresholds["EOS-MAT"]
    )

    # filter cell detections, do not use quality filtering models (default)
    result_table = filter_cells(
        result_table,
        morph_data,
        min_dist_from_edge,
        max_aspect_ratio,
        filter_with_erythroblast_quality,
        filter_with_granulocyte_quality,
        keep_artefacts,
        keep_unknowns,
        filter_with_keepval,
        filter_with_quality,
        filter_out_multinuclear
    )

    # aggregate cell specific results to sample specific results
    result_table = _calculate_cell_differential_counts(result_table)

    return result_table

def extract_sample_cell_statistics(
    morph_data: Dataset,
    unknown_confidence_threshold: float = 0.25,
    binary_model_confidence_thresholds: dict = {
        "QUA-ERY": 0.5,
        "QUA-GRA": 0.5,
    },
    min_dist_from_edge: int = 1,
    max_aspect_ratio: float = np.inf,
    filter_with_erythroblast_quality: bool = True,
    filter_with_granulocyte_quality: bool = True,
    keep_artefacts: bool = False,
    keep_unknowns: bool = False,
    filter_with_keepval: bool = True,
    filter_with_quality: bool = True,
    filter_out_multinuclear: bool = True
) -> DataFrame:

    results = morph_data["RESULTS"]

    result_table = pd.DataFrame(
        data={"KEEPVAL": results["KEEPVAL"], "DET-IDX": morph_data["DET-IDX"]}
    )
    
    result_table = _extract_class_labels(
        results,
        result_table,
        unknown_confidence_threshold,
    )

    result_table = _extract_erythroblast_quality_results(
        results, result_table, binary_model_confidence_thresholds["QUA-ERY"]
    )
    result_table = _extract_granulocyte_quality_results(
        results, result_table, binary_model_confidence_thresholds["QUA-GRA"]
    )

    # filter cell detections, filter cells for statistic quality.
    result_table = filter_cells(
        result_table,
        morph_data,
        min_dist_from_edge,
        max_aspect_ratio,
        filter_with_erythroblast_quality,
        filter_with_granulocyte_quality,
        keep_artefacts,
        keep_unknowns,
        filter_with_keepval,
        filter_with_quality,
        filter_out_multinuclear
    )

    cell_id_to_label_map = dict(zip(result_table["DET-IDX"], result_table["CLS-LBL"]))
    cell_indices_to_keep = result_table["DET-IDX"].to_list()
    statistics = morph_data["STATISTICS"]

    result_table = _extract_cell_statistics(
        statistics,
        cell_indices_to_keep,
        cell_id_to_label_map,
    )
    
    # add information on how many cells were used to calculate cell statistics
    result_table.insert(0, "n_cells_in_stat_calculations", len(cell_indices_to_keep))

    # by cell-type
    cells = np.array(list(cell_id_to_label_map.values()))
    for ct in CELL_TYPES_TO_FILTER:
        cc = cells[cells == ct].shape[0]
        result_table.insert(0, f"{ct}_in_stats_calculations", cc)

    return result_table

def extract_sample_raw_cell_results(
    morph_data: Dataset,
    unknown_confidence_threshold: float = 0.25,
    binary_model_confidence_thresholds: dict = {
        "QUA-ERY": 0.5,
        "QUA-GRA": 0.5,
    },
    min_dist_from_edge: int = 0,
    max_aspect_ratio: float = np.inf,
    filter_with_erythroblast_quality: bool = True,
    filter_with_granulocyte_quality: bool = True,
    keep_artefacts: bool = False,
    keep_unknowns: bool = False,
    filter_with_keepval: bool = True,
    filter_with_quality: bool = True,
    filter_out_multinuclear: bool = True
) -> DataFrame:
    """
    Extracts cell results of a single sample without aggregating

    should return the same set of cells as extract_sample_raw_cell_statistics
    when min_dist_from_edge = 1
    """

    results = morph_data["RESULTS"]

    result_table = pd.DataFrame(
        data={"KEEPVAL": results["KEEPVAL"], "DET-IDX": morph_data["DET-IDX"]}
    )

    result_table = _extract_class_labels(
        results,
        result_table,
        unknown_confidence_threshold
    )

    result_table = _extract_erythroblast_quality_results(
        results, result_table, binary_model_confidence_thresholds["QUA-ERY"]
    )
    result_table = _extract_granulocyte_quality_results(
        results, result_table, binary_model_confidence_thresholds["QUA-GRA"]
    )

    # filter cell detections, filter cells for statistic quality.
    result_table = filter_cells(
        result_table,
        morph_data,
        min_dist_from_edge,
        max_aspect_ratio,
        filter_with_erythroblast_quality,
        filter_with_granulocyte_quality,
        keep_artefacts,
        keep_unknowns,
        filter_with_keepval,
        filter_with_quality,
        filter_out_multinuclear
    )

    cell_id_to_label_map = dict(zip(result_table["DET-IDX"], result_table["CLS-LBL"]))
    cell_indices_to_keep = result_table["DET-IDX"].to_list()

    # extract results
    result_df = _extract_raw_cell_results(
        morph_data,
        cell_indices_to_keep,
        cell_id_to_label_map
    )

    result_df.index.name = "DET-IDX"
    return result_df

def extract_sample_raw_cell_statistics(
    morph_data: Dataset,
    unknown_confidence_threshold: float = 0.25,
    binary_model_confidence_thresholds: dict = {
        "QUA-ERY": 0.5,
        "QUA-GRA": 0.5,
    },
    min_dist_from_edge: int = 1,
    max_aspect_ratio: float = np.inf,
    filter_with_erythroblast_quality: bool = True,
    filter_with_granulocyte_quality: bool = True,
    keep_artefacts: bool = False,
    keep_unknowns: bool = False,
    filter_with_keepval: bool = True,
    filter_with_quality: bool = True,
    filter_out_multinuclear: bool = True
) -> DataFrame:
    """
    Extracts cell statistics of a single sample without aggregating.
    """

    results = morph_data["RESULTS"]

    result_table = pd.DataFrame(
        data={"KEEPVAL": results["KEEPVAL"], "DET-IDX": morph_data["DET-IDX"]}
    )

    result_table = _extract_class_labels(
        results,
        result_table,
        unknown_confidence_threshold
    )

    result_table = _extract_erythroblast_quality_results(
        results, result_table, binary_model_confidence_thresholds["QUA-ERY"]
    )
    result_table = _extract_granulocyte_quality_results(
        results, result_table, binary_model_confidence_thresholds["QUA-GRA"]
    )

    # filter cell detections, filter cells for statistic quality.
    result_table = filter_cells(
        result_table,
        morph_data,
        min_dist_from_edge,
        max_aspect_ratio,
        filter_with_erythroblast_quality,
        filter_with_granulocyte_quality,
        keep_artefacts,
        keep_unknowns,
        filter_with_keepval,
        filter_with_quality,
        filter_out_multinuclear
    )

    cell_id_to_label_map = dict(zip(result_table["DET-IDX"], result_table["CLS-LBL"]))
    cell_indices_to_keep = result_table["DET-IDX"].to_list()
    statistics = morph_data["STATISTICS"]

    result_df = _extract_raw_cell_statistics(
        statistics,
        cell_indices_to_keep,
        cell_id_to_label_map
    )

    result_df.index.name = "DET-IDX"
    return result_df
