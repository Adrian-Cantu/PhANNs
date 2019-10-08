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
n_members = 10
models = list()
for model_number in range(n_members):
    # load model
    model =  load_model( os.path.join(phage_init.model_dir,'tri_p_'+"{:02d}".format(model_number+1)+'.h5') )
    # store in memory
    models.append(model)


# %%
(test_X,test_Y)=ann_data.get_formated_test("tri_p")

# %%
yhats = [model.predict(test_X) for model in models]


# %%
yhats_v=numpy.array(yhats)

# %%
summed=numpy.sum(yhats_v, axis=0)

# %%
outcomes = numpy.argmax(summed, axis=1)

# %%
outcomes

# %%
labels_names=["Major capsid","Minor capsid","Baseplate","Major tail","Minor tail","Portal","Tail fiber",
             "Tail shaft","Collar","Head-Tail joining","Others"]
test_Y_index = test_Y.argmax(axis=1)
print(classification_report(test_Y_index, outcomes, target_names=labels_names ))
