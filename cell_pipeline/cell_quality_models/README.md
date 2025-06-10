## Cell type-specific quality classifiers.

Binary random forest classifiers for filterting cells prior to cell statistic calculation.

Megakaryocyte model is missing on purpose as there is a separate classifier for that.
Macrophage model is missing on purpose as the class is very rare.

Goal: cells classified with 0 "Quality Bad" would be those that are squished between other cells, meaning that calculating their statistics does not make much sense as the reason for the morphology is mechanical pressure from surrounding cells, not the cell itself.

These models should generally not be used before calculating any cell differentials or when running classification models, but are used as a filter before calculating statistics (area, roundness, etc.)

### Input:
- LOW-QUALITY
- cytoplasm area
- nucleus area
- nucleus compactness
- nucleus roundness
- cell area
- cell compactness
- cell roundness

### Output:
- 1 -> Quality Good
- 0 -> Quality Bad

