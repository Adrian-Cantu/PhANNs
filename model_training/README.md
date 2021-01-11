<!-- #region -->
# Model training

## Requirements

All requirement are listed on the `environment.yml` file, but the easier way to install them is using anaconda

```
conda env create -f environment.yml
```

Then activate the environment

```
conda activate tf2
```

Beside the requirement in the environment file Perl and a modified version of CD-hit (  [available here](https://github.com/Adrian-Cantu/cdhit) ) are required.


## Build your own model

Training a model consist of several steps :

1. Download structural sequences from NCBI (and non structural from the PhANNs webserver)
2. Generate the curating list and do manual curation on them.
3. Generate fasta files from the curating list.
4. Remove non-structural proteins that cluster with structural ones.
5. Sequence clustering, split train/test/validation sets ,and cluster expansion.
6. Feature extraction .
7. Training and 10-fold cross-validation
8. Interpret results and Generate figures.
9. Compare to a simple logistic regression model.

All python scripts in this directory indicate the order they should be run to train a new model. Note that they are meant to be used as a jupyter notebook with the **jupytext** plugin (included in the environment file).   

## If you want to:

### Update the database, using the same curation:
Run step 1, skip 2 and then run steps 3-9

### Update manual curation
Download and uncompress the [Raw sequences](https://edwards.sdsu.edu/phanns/download/rawDB.tgz) in the ``01_fasta/`` directory.
Update the curating list in ``03_curated_fasta/`` and run steps 3-9.

### Reproduce the figures in the paper
Download and uncompress the [Expanded clusters](https://edwards.sdsu.edu/phanns/download/expandedDB.tgz) in the ``05_2_expanded_clusters`` directory and run steps 6-9.  

## Included files
Most intermediary file are either too large or of little use by themselves to be included in this repository. Sequence files at from all stages of processing are available in the [PhANNs web server](https://edwards.sdsu.edu/phanns/downloads).

We have included a few files that make reproducing the models and figures in the paper possible:
1. In the **01_fasta** directory, the original list of manual curation terms used.
2. **07_models/all_results_df.p** 10-fold cross-validation results, as a pickled pandas data-frame.
3. **09_logistic_models/log_kfold_df.p** 10-fold cross-validation results of a logistic regression model trained on the same data, as a pickled pandas data-frame.
4. **09_logistic_models/40_derep_results.p** 10-fold cross-validation results of a ANN model trained on the data before cluster expansion, as a pickled pandas data-frame.
5. **08_figures/CM_predicted_test_Y.p** and **08_figures/CM_test_Y.p**,  real and predicted class for the test set using the tetra_sc_tri_p ensemble. Used to construct the confusion matrix.

We also included  image files for figures that are used in the [PhANNs paper](https://www.biorxiv.org/content/10.1101/2020.04.03.023523v1).

<!-- #endregion -->
