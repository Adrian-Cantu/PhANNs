# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import os
import sys
sys.path.append("..")
import phage_init

# %%
import ann_data

# %%
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Dropout
from keras.models import load_model
from sklearn.metrics import classification_report
import numpy
#load the saved matrices
import pickle

# %%
err=pickle.load(open( os.path.join(phage_init.data_dir,"error.p"), "rb" ))

# %%
sorted_errors = numpy.sort(err)[0]
print(sorted_errors)
sorted_reverse = sorted_errors[::-1]
print(sorted_reverse)

# %%
import matplotlib.pyplot as plt
import seaborn as sns
plot_range=8000
sns.lineplot(numpy.arange(plot_range), sorted_reverse[0:plot_range])

# %%
features_index=ann_data.get_feature_names('tri_p')

# %%
top_imp_i=err[0].argsort()[-20:][::-1]

# %%
top_imp=[features_index[i] for i in top_imp_i]
print(sorted_reverse[0:20])
print(top_imp)
