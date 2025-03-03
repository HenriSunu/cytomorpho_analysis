import pandas as pd
from pandas import DataFrame

from constants import (
    CELL_LABEL_MAP,
    CELL_LINEAGES,
    GRANULOCYTE_QUALITY_LABEL_MAP,
    GRANULOCYTE_QUALITY_INCLUDED_CELLS,
    PROMYELOCYTE_CLASS_LABEL_MAP,
    PROMYELOCYTE_CLASS_INCLUDED_CELLS,
    MGK_CLS_LABEL_MAP,
    MGK_INCLUDED_CELLS,
    ERYTHROBLAST_CLASS_LABEL_MAP,
    ERYTHROBLAST_CLASS_INCLUDED_CELLS,
    GRANULARITY_LABEL_MAP,
    GRANULARITY_CLASS_INCLUDED_CELLS,
    EOSINOPHIL_MAT_LABEL_MAP,
)

def _get_cell_lineage_specific_column_name_for_vacuolization(cell_lineage):
    return f"Vacuolated {cell_lineage}"

def _get_cell_lineage_specific_column_name_for_multinuclear_results(cell_lineage):
    return f"Multinuclear {cell_lineage}"

def _calculate_cell_percentages(data: DataFrame) -> DataFrame:
    """
    Calculate percentage values for living cells, artefacts and unknown cells out of
    all cells, and percentage values for living cell types out of all living cells.
    """

    all_cell_labels = list(CELL_LABEL_MAP.values())
    living_cell_labels = CELL_LINEAGES["Living cells"]

    data["Living cells"] = data[living_cell_labels].sum(axis=1)
    data["All cells"] = data[all_cell_labels].sum(axis=1)
    data["Granulopoietic cells"] = data[CELL_LINEAGES["Granulopoietic cells"]].sum(
        axis=1
    )
    data["Erythropoietic cells"] = data[CELL_LINEAGES["Erythropoietic cells"]].sum(
        axis=1
    )

    data["Living cells_Percentage"] = 100 * data["Living cells"] / data["All cells"]
    data["Artefacts_Percentage"] = 100 * data["Artefacts"] / data["All cells"]
    data["Unknowns_Percentage"] = 100 * data["Unknowns"] / data["All cells"]
    data["Granulopoietic cells_Percentage"] = (
        100 * data["Granulopoietic cells"] / data["Living cells"]
    )
    data["Erythropoietic cells_Percentage"] = (
        100 * data["Erythropoietic cells"] / data["Living cells"]
    )

    proportions = data[living_cell_labels].div(data["Living cells"], axis=0) * 100
    proportions.columns = [f"{col}_Percentage" for col in living_cell_labels]
    data = pd.concat([data, proportions], axis=1)

    return data

def _calculate_me_ratio(data: DataFrame) -> DataFrame:
    """
    Calculate M:E ratio (myeloid to nucleated erythroid ratio)
    """
    
    myeloid_cells = data["Granulopoietic cells"].astype("float") + data["Blasts"].astype("float")
    data["M:E ratio"] = myeloid_cells / data["Erythropoietic cells"].astype("float")

    # If there are no erythropoietic cells, set the M:E ratio to None
    msk = data["Erythropoietic cells"] == 0
    data.loc[msk, "M:E ratio"] = None

    return data

def _calculate_granulocyte_quality_percentages(data: DataFrame) -> DataFrame:
    """
    Calculate the percentage of low-quality and normal-quality granulocytes out of
    all granulocytes.
    """
    denominator = data[GRANULOCYTE_QUALITY_INCLUDED_CELLS].sum(axis=1)
    for label in GRANULOCYTE_QUALITY_LABEL_MAP.values():
        new_column_name = f"{label}_Percentage"
        data[new_column_name] = 0
        non_zero_mask = denominator != 0
        data.loc[non_zero_mask, new_column_name] = (
            100 * data[label] / denominator[non_zero_mask]
        )

    return data

def _calculate_promyelocyte_percentages(data: DataFrame) -> DataFrame:
    """
    Calculate the percentage of promyelocyte-subclass results out of blasts,
    promyelocytes and myelocytes.
    """
    denominator = data[PROMYELOCYTE_CLASS_INCLUDED_CELLS].sum(axis=1)
    for label in PROMYELOCYTE_CLASS_LABEL_MAP.values():
        new_column_name = f"{label}_Percentage"
        data[new_column_name] = 0
        non_zero_mask = denominator != 0
        data.loc[non_zero_mask, new_column_name] = (
            100 * data[label] / denominator[non_zero_mask]
        )

    return data

def _calculate_megakaryocyte_percentages(data: DataFrame) -> DataFrame:
    """
    Calculate the percentage of megakaryocytes by class.
    """

    # calculate the total number and percentage of dysplastic megakaryocytes
    mgk_dysplasia_label = "Megakaryocytes-Class=Dysplastic"
    data[mgk_dysplasia_label] = data[[MGK_CLS_LABEL_MAP[1], MGK_CLS_LABEL_MAP[2]]].sum(
        axis=1
    )

    # 1) Dysplastic and normal megakaryocytes
    dysplasia_classes = [
        MGK_CLS_LABEL_MAP[0],
        MGK_CLS_LABEL_MAP[1],
        MGK_CLS_LABEL_MAP[2],
    ]
    denominator_dysplasia = data[dysplasia_classes].sum(axis=1)
    dysplasia_classes_to_iterate = [
        MGK_CLS_LABEL_MAP[0],
        MGK_CLS_LABEL_MAP[1],
        MGK_CLS_LABEL_MAP[2],
        mgk_dysplasia_label,
    ]
    for label in dysplasia_classes_to_iterate:
        column_name = f"{label}_Percentage"
        data[column_name] = 0
        non_zero_mask = denominator_dysplasia != 0
        data.loc[non_zero_mask, column_name] = (
            100 * data[label] / denominator_dysplasia[non_zero_mask]
        )

    # 2) Broken megakaryocytes
    denominator_quality = data[MGK_INCLUDED_CELLS].sum(axis=1)
    column_name = f"{MGK_CLS_LABEL_MAP[3]}_Percentage"
    data[column_name] = 0
    non_zero_mask = denominator_quality != 0
    data.loc[non_zero_mask, column_name] = (
        100 * data[label] / denominator_quality[non_zero_mask]
    )

    return data

def _calculate_vacuolization_percentages(data: DataFrame) -> DataFrame:
    """
    Calculate the percentage of vacuolated cells in each cell lineage.
    """
    for cell_lineage, cell_types in CELL_LINEAGES.items():
        denominator = data[cell_types].sum(axis=1)
        if cell_lineage not in ["Living cells", "Artefacts", "Unknowns"]:
            column_name = _get_cell_lineage_specific_column_name_for_vacuolization(
                cell_lineage
            )
            new_column_name = f"{column_name}_Percentage"
            data[new_column_name] = 0
            non_zero_mask = denominator != 0
            data.loc[non_zero_mask, new_column_name] = (
                100 * data[column_name] / denominator[non_zero_mask]
            )

    return data

def _calculate_erythroblast_percentages(data: DataFrame) -> DataFrame:
    """
    Calculate the percentage of erythroblast-subclasses out of all erythroblasts.
    """
    denominator = data[ERYTHROBLAST_CLASS_INCLUDED_CELLS].sum(axis=1)
    for label in ERYTHROBLAST_CLASS_LABEL_MAP.values():
        new_column_name = f"{label}_Percentage"
        data[new_column_name] = 0
        non_zero_mask = denominator != 0
        data.loc[non_zero_mask, new_column_name] = (
            100 * data[label] / denominator[non_zero_mask]
        )

    return data

def _calculate_granularity_percentages(data: DataFrame) -> DataFrame:
    """
    Calculate the percentage of granularity results out of all granulocytes.
    """
    denominator = data[GRANULARITY_CLASS_INCLUDED_CELLS].sum(axis=1)
    for label in GRANULARITY_LABEL_MAP.values():
        new_column_name = f"{label}_Percentage"
        data[new_column_name] = 0
        non_zero_mask = denominator != 0
        data.loc[non_zero_mask, new_column_name] = (
            100 * data[label] / denominator[non_zero_mask]
        )

    return data

def _calculate_multinuclear_percentages(data: DataFrame) -> DataFrame:
    """
    Calculate the percentage of multinuclear cells out of relevant subgroups.
    """
    for cell_lineage, cell_types in CELL_LINEAGES.items():
        denominator = data[cell_types].sum(axis=1)
        if cell_lineage not in ["Living cells", "Artefacts", "Unknowns"]:
            column_name = (
                _get_cell_lineage_specific_column_name_for_multinuclear_results(
                    cell_lineage
                )
            )
            new_column_name = f"{column_name}_Percentage"
            data[new_column_name] = 0
            non_zero_mask = denominator != 0
            data.loc[non_zero_mask, new_column_name] = (
                100 * data[column_name] / denominator[non_zero_mask]
            )

    return data

def _calculate_eosinophil_percentages(data: DataFrame) -> DataFrame:
    """
    Calculate the percentage of mature and immature cells out of all living cells.
    Additionally calculate proportion of mature eosinophils
    """
    living_cells = data.loc[:,CELL_LINEAGES["Living cells"]].sum(axis=1)
    eosinophils = data["Eosinophils"]

    # Proportion of mature eosinophils out of eosinophils
    new_column_name = f"{EOSINOPHIL_MAT_LABEL_MAP[0]}_Percentage"
    non_zero_mask = eosinophils != 0
    data.loc[non_zero_mask, new_column_name] = (
        100 * data[EOSINOPHIL_MAT_LABEL_MAP[0]] / eosinophils[non_zero_mask]
    )

    # Proportion of mature and immature eosinophils out of all living cells
    new_column_name = "Mature_Eosinophils_Percentage"
    non_zero_mask = living_cells != 0
    data.loc[non_zero_mask, new_column_name] = (
        100 * data[EOSINOPHIL_MAT_LABEL_MAP[0]] / living_cells[non_zero_mask]
    )

    new_column_name = "Immature_Eosinophils_Percentage"
    data.loc[non_zero_mask, new_column_name] = (
        100 * data[EOSINOPHIL_MAT_LABEL_MAP[1]] / living_cells[non_zero_mask]
    )

    return data

def calculate_percentage_values(data: DataFrame) -> DataFrame:
    """
    Calculate percentage values for different cell results.
    """
    data = _calculate_cell_percentages(data)
    data = _calculate_me_ratio(data)
    data = _calculate_granulocyte_quality_percentages(data)
    data = _calculate_promyelocyte_percentages(data)
    data = _calculate_megakaryocyte_percentages(data)
    data = _calculate_vacuolization_percentages(data)
    data = _calculate_erythroblast_percentages(data)
    data = _calculate_granularity_percentages(data)
    data = _calculate_multinuclear_percentages(data)
    data = _calculate_eosinophil_percentages(data)

    data = data.reindex(sorted(data.columns), axis=1)

    return data