# model training

## requirements

Beside the requirement in the environment file. Perl and a modified version of CD-hit (  [available here](https://github.com/Adrian-Cantu/cdhit) ) are required.

training a model consist of 4 steps :

1. Download sequences from NCBI
2. Sequence per-processing (Manual Curation, clustering, split, and expansion)
3. Feature extraction
4. Training and 10-fold cross-validation
5. Test

All python scripts in this directory indicate the order they should be run to train a new model. Note that they are meant to be used as a jupyter notebook with the **jupytext** plugin (included in the environment file).   