# Cytomorpholgy analysis

Scripts by: Henri Sundquist & Mikko Purhonen

This repository includes representative toy data output by the image analysis pipeline as well as an example of how this data was utilized in the manuscript *Computational Cytomorphological Analysis Predicts Response to Venetoclax and Hypomethylating Agents in Acute Myeloid Leukemia*

### toy_data.hdf5
Includes 100 simulated samples each containing 500 cells over three classes (blasts, erythroblasts, neutrophils).

### analysis_example.ipynb
Includes code for
- Reading the data on a single-cell level from the toy dataset for exploratory analysis
- Reading sample-wise aggregated features, filtering those features, and applying repeated and nested cross-validation to predict simulated PFS times.

### Setting up environment
```
conda env create -n cytomorpho-analysis -f env.yml
```