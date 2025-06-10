from pathlib import Path
CELL_MODEL_PATH = f"{Path(__file__).resolve().parent}/cell_quality_models"

CELL_LABEL_MAP = {
    0: "Artefacts",
    1: "Basophils",
    2: "Blasts",
    3: "Eosinophils_immature",
    4: "Eosinophils",
    5: "Erythroblasts",
    6: "Lymphocytes",
    7: "Macrophages",
    8: "Megakaryocytes",
    9: "Metamyelocytes",
    10: "Monocytes",
    11: "Myelocytes",
    12: "Neutrophils",
    13: "Plasma cells",
    14: "Proerythroblasts",
    15: "Promonocytes",
    16: "Promyelocytes",
    -1: "Unknowns",  # This is an extra class not present in the cell classification model
}

CELL_STATISTICS_CLASSES = [
    "Basophils",
    "Blasts",
    "Eosinophils_immature",
    "Eosinophils",
    "Erythroblasts",
    "Lymphocytes",
    "Macrophages",
    "Megakaryocytes",
    "Metamyelocytes",
    "Monocytes",
    "Myelocytes",
    "Neutrophils",
    "Plasma cells",
    "Proerythroblasts",
    "Promonocytes",
    "Promyelocytes",
]

STATISTICS = {
    "CYTOPLASM": [
        "area"
    ],
    "NUCLEUS": [
        "area",
        "perimeter",
        "solidity",
        "eccentricity",
        "compactness",
    ],
    "ENTIRE": [
        "area",
        "perimeter",
        "solidity",
        "eccentricity",
        "compactness",
    ]
}

CELL_LINEAGES = {
    "Living_cells": [
        "Basophils",
        "Blasts",
        "Eosinophils_immature",
        "Eosinophils",
        "Erythroblasts",
        "Lymphocytes",
        "Macrophages",
        "Megakaryocytes",
        "Metamyelocytes",
        "Monocytes",
        "Myelocytes",
        "Neutrophils",
        "Plasma cells",
        "Proerythroblasts",
        "Promonocytes",
        "Promyelocytes",
    ],
    "Granulopoetic_cells": [
        "Promyelocytes",
        "Myelocytes",
        "Metamyelocytes",
        "Basophils",
        "Neutrophils",
        "Eosinophils",
        "Eosinophils_immature",
    ],
    "Monocytic_cells": [
        "Promonocytes",
        "Monocytes",
        "Macrophages",
    ],
    "Erythropoetic_cells": [
        "Proerythroblasts",
        "Erythroblasts",
    ],
    "Lymphoid_cells": [
        "Lymphocytes",
        "Plasma cells",
    ],
    "Granulocytes": [
        "Neutrophils",
        "Eosinophils",
        "Basophils",
    ],
    "APL": [
        "Blasts",
        "Promyelocytes",
        "Myelocytes",
    ]
}

ALL_RESULTS = [
    "APL-HYPERGRANULATED",
    "APL-BILOBED",
    "APL-AUER",
    "MDS-ASYNCHRONY",
    "MDS-DYSMORPHIC",
    "MDS-MULTINUCLEATED",
    "MITOTIC",
    "VACUOLI",
]

CELL_TYPES_TO_FILTER = [
    "Basophils",
    "Blasts",
    "Eosinophils",
    "Eosinophils_immature",
    "Erythroblasts",
    "Lymphocytes",
    "Metamyelocytes",
    "Monocytes",
    "Myelocytes",
    "Neutrophils",
    "Plasma cells",
    "Proerythroblasts",
    "Promyelocytes",
    "Promonocytes",
]

# Which model to use for which cell type class
CELL_TYPE_TO_MODEL = {
    "Basophils": "Basophils",
    "Blasts": "Blasts",
    "Eosinophils": "Eosinophils",
    "Eosinophils_immature": "Eosinophils_immature",
    "Erythroblasts": "Erythroblasts",
    "Lymphocytes": "Lymphocytes",
    "Metamyelocytes": "Metamyelocytes",
    "Monocytes": "Monocytes",
    "Myelocytes": "Myelocytes",
    "Neutrophils": "Neutrophils",
    "Plasma cells": "Plasma cells",
    "Proerythroblasts": "Proerythroblasts",
    "Promyelocytes": "Promyelocytes",
    "Promonocytes": "Promonocytes",
}

# Caution: the order is important here!
QUALITY_FILTER_STATISTICS = {
    "CYTOPLASM": [
        "area",
    ],
    "NUCLEUS": [
        "area",
        "compactness",
        "roundness",
    ],
    "ENTIRE": [
        "area",
        "compactness",
        "roundness",
    ],
}

QUALITY_FILTER_RESULTS = [
    "LOW-QUALITY",
]

APL_RESULTS = [
    "APL-HYPERGRANULATED",
    "APL-BILOBED",
    "APL-AUER",
]

APL_INCLUDED_CLASSES = [
    "Blasts",
    "Promyelocytes",
    "Myelocytes",
    "APL"
]

MDS_RESULTS = [
    "MDS-ASYNCHRONY",
    "MDS-DYSMORPHIC",
    "MDS-MULTINUCLEATED",
]

MDS_INCLUDED_CLASSES = [
    "Erythroblasts",
]

ERYTHROBLAST_QUALITY_CLASSES = [
    "Erythroblasts"
]

GRANULOCYTE_QUALITY_CLASSES = [
    "Neutrophils",
    "Eosinophils",
    "Basophils",
    "Eosinophils_immature",
    "Myelocytes",
    "Metamyelocytes",
]