SAMPLE_ID_COLUMN_NAME = "Sample"

CELL_LABEL_MAP = {
    0: "Artefacts",
    1: "Basophils",
    2: "Blasts",
    3: "Eosinophils",
    4: "Erythroblasts",
    5: "Lymphocytes",
    6: "Megakaryocytes",
    7: "Metamyelocytes",
    8: "Monocytes",
    9: "Myelocytes",
    10: "Neutrophils",
    11: "Plasma cells",
    12: "Proerythroblasts",
    13: "Promyelocytes",
    14: "Promonocytes",
    15: "Macrophages",
    -1: "Unknowns",  # This is an extra class not present in the cell classification model
}

MONOCYTE_SUBCLASS_MAP = {
    0: "Monocytes",
    1: "Promonocytes",
    2: "Macrophages",
}

CELL_TYPES_TO_FILTER = [
    "Basophils",
    "Blasts",
    "Eosinophils",
    "Erythroblasts",
    "Lymphocytes",
    "Megakaryocytes",
    "Metamyelocytes",
    "Monocytes",
    "Myelocytes",
    "Neutrophils",
    "Plasma cells",
    "Proerythroblasts",
    "Promyelocytes",
    "Promonocytes",
    "Macrophages",
]

# Which model to use for which cell type class
CELL_TYPE_TO_MODEL = {
    "Basophils": "Basophils",
    "Blasts": "Blasts",
    "Eosinophils": "Eosinophils",
    "Erythroblasts": "Erythroblasts",
    "Lymphocytes": "Lymphocytes",
    "Megakaryocytes": "Megakaryocytes",
    "Metamyelocytes": "Metamyelocytes",
    "Monocytes": "Monocytes",
    "Myelocytes": "Myelocytes",
    "Neutrophils": "Neutrophils",
    "Plasma cells": "Plasma cells",
    "Proerythroblasts": "Proerythroblasts",
    "Promyelocytes": "Promyelocytes",
    "Promonocytes": "Monocytes",
    "Macrophages": "Monocytes",
}

CELL_LINEAGES = {
    "Blasts": ["Blasts"],
    "Granulopoietic cells": [
        "Promyelocytes",
        "Myelocytes",
        "Metamyelocytes",
        "Basophils",
        "Neutrophils",
        "Eosinophils",
    ],
    "Erythropoietic cells": ["Proerythroblasts", "Erythroblasts"],
    "Monocytic cells": ["Monocytes", "Promonocytes", "Macrophages"],
    "Lymphocytes": ["Lymphocytes"],
    "Plasma cells": ["Plasma cells"],
    "Megakaryocytes": ["Megakaryocytes"],
    "Artefacts": ["Artefacts"],
    "Unknowns": ["Unknowns"],
    "Living cells": [
        "Blasts",
        "Proerythroblasts",
        "Erythroblasts",
        "Promyelocytes",
        "Myelocytes",
        "Metamyelocytes",
        "Basophils",
        "Neutrophils",
        "Eosinophils",
        "Monocytes",
        "Lymphocytes",
        "Plasma cells",
        "Megakaryocytes",
        "Promonocytes",
        "Macrophages"
    ],
}

ERYTHROBLAST_QUALITY_LABEL_MAP = {
    0: "Erythroblasts-Quality=Normal",
    1: "Erythroblasts-Quality=Low",
}
ERYTHROBLAST_QUALITY_INCLUDED_CELLS = ["Erythroblasts"]

GRANULOCYTE_QUALITY_LABEL_MAP = {
    0: "Granulocytes-Quality=Normal",
    1: "Granulocytes-Quality=Low",
}
GRANULOCYTE_QUALITY_INCLUDED_CELLS = [
    "Myelocytes",
    "Metamyelocytes",
    "Neutrophils",
    "Basophils",
    "Eosinophils",
]

MGK_CLS_LABEL_MAP = {
    0: "Megakaryocytes-Class=Normal",
    1: "Megakaryocytes-Class=Multinuclear",
    2: "Megakaryocytes-Class=Hypolobulated",
    3: "Megakaryocytes-Class=Broken",
}

MGK_QUA_LABEL_MAP = {
    0: "Megakaryocytes-Quality=Late-Broken",
    1: "Megakaryocytes-Quality=Unclear",
    2: "Megakaryocytes-Quality=Normal",
}
MGK_INCLUDED_CELLS = ["Megakaryocytes"]

GRANULARITY_LABEL_MAP = {
    0: "Granulopoietic cells-Granularity=Low",
    1: "Granulopoietic cells-Granularity=Normal",
    2: "Granulopoietic cells-Granularity=High",
}
GRANULARITY_CLASS_INCLUDED_CELLS = [
    "Blasts",
    "Promyelocytes",
    "Myelocytes",
    "Metamyelocytes",
    "Neutrophils",
]

EOSINOPHIL_MAT_LABEL_MAP = {
    0: "Eosinophils-Maturation=Mature",
    1: "Eosinophils-Maturation=Immature",
}
EOSINOPHIL_CLASS_INCLUDED_CELLS = ["Eosinophils"]

ERYTHROBLAST_CLASS_LABEL_MAP = {
    "MDS-CYT": "Erythroblasts-Class=Abnormal cytoplasm",
    "MDS-DYS": "Erythroblasts-Class=Dysmorphic",
    "MDS-MUL": "Erythroblasts-Class=Multinuclear",
    "MDS-MEG": "Erythroblasts-Class=Megaloblastic",
}
ERYTHROBLAST_CLASS_INCLUDED_CELLS = ["Erythroblasts"]

PROMYELOCYTE_CLASS_LABEL_MAP = {
    "APL-ABN": "APL-Class=Abnormal",
    "APL-HYP": "APL-Class=Hypergranular",
    "APL-INC": "APL-Class=Inclusion",
}
PROMYELOCYTE_CLASS_INCLUDED_CELLS = ["Blasts", "Promyelocytes", "Myelocytes"]

MULTINUCLEAR_LABEL_MAP = {"MULTINUCLEAR": "Multinuclear living cells"}

# Cell statistics
STATISTICS_TO_EXTRACT = {
    "ENTIRE": [
        "area",
        "convexity",
        "roundness",
        "eccentricity",
        "solidity",
        "perimeter",
        "compactness",
        "nc-ratio"
    ],
    "CYTOPL": [
        "area",
        "color-mean",
        "color-std",
        "luminance",
        "glcm-homogen",
        "glcm-energy"
    ],
    "NUCLEI": [
        "area",
        "color-mean",
        "color-std",
        "convexity",
        "roundness",
        "eccentricity",
        "solidity",
        "perimeter",
        "compactness",
        "luminance",
        "glcm-homogen",
        "glcm-energy"
    ],
}

STATISTICS_TO_ENGINEER = {
    "ENTIRE": [
        "nc-ratio"
    ],
    "CYTOPL":[
        "luminance"
    ],
    "NUCLEI":[
        "luminance"
    ]
}

STATISTIC_COMPONENT_LABEL_MAP = {
    "ENTIRE": "cell",
    "CYTOPL": "cytoplasm",
    "NUCLEI": "nuclei",
}

COLOR_CHANNEL_LABELS = {
    0: "red",
    1: "green",
    2: "blue",
}

QUALITY_FILTER_STATISTICS = {
    "ENTIRE": [
        "area",
        "convexity",
        "roundness",
        "eccentricity",
        "solidity",
        "perimeter",
        "compactness",
    ],
    "CYTOPL": [
        "area",
        "color-mean",
        "color-std",
        "glcm-homogen",
        "glcm-energy",
    ],
    "NUCLEI": [
        "area",
        "color-mean",
        "color-std",
        "convexity",
        "roundness",
        "eccentricity",
        "solidity",
        "perimeter",
        "compactness",
        "glcm-homogen",
        "glcm-energy",
    ]
}

QUALITY_FILTER_RESULTS = [
    "APL-ABN",
    "APL-HYP",
    "APL-INC",
    "MDS-CYT",
    "MDS-DYS",
    "MDS-MUL",
    "MDS-MEG",
    "QUA-GRA",
    "QUA-ERY",
]

RAW_RESULTS_TO_EXTRACT = [
    "APL-ABN",
    "APL-HYP",
    "MDS-CYT",
    "MDS-DYS",
    "MDS-MUL",
    "MDS-MEG",
    "MGK-CLS",
    "VACUOLI",
    "GRANULA",
    "EOS-MAT",
]
