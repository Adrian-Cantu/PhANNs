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
model = load_model( os.path.join(phage_init.model_dir,'tri_p_single.h5') )

# %%
import pandas as pd
id_df=pickle.load(open( os.path.join(phage_init.data_dir,"raw_df.p"), "rb" ))

# %%
(test_X,test_Y)=ann_data.get_formated_test("tri_p")
test_Y_index = test_Y.argmax(axis=1)
test_Y_predicted = model.predict_classes(test_X)

# %%
#pickle.dump(test_X,open( os.path.join(phage_init.data_dir,"test_X.p"), "wb" ),protocol=4 )
#pickle.dump(test_Y_index,open( os.path.join(phage_init.data_dir,"test_Y_index.p"), "wb" ),protocol=4 )

# %%
labels_names=["Major capsid","Minor capsid","Baseplate","Major tail","Minor tail","Portal","Tail fiber",
             "Tail shaft","Collar","Head-Tail joining","Others"]
print(classification_report(test_Y_index, test_Y_predicted, target_names=labels_names ))

# %%
from collections import Counter
zz=Counter(test_Y_index)
sample_w=[zz[i] for i in range(0,11,1)]

# %%
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
plt.show()
CM=confusion_matrix(test_Y_index, test_Y_predicted)
CM_n=CM/numpy.array(sample_w)[:,None]
scale_up=1.8
plt.figure(figsize=[6.4*scale_up, 4.8*scale_up])
plt.imshow(CM_n, interpolation='nearest')
#plt.title('CM all')
plt.colorbar()
tick_marks = numpy.arange(len(labels_names))
plt.xticks(tick_marks, labels_names, rotation=90)
plt.yticks(tick_marks, labels_names)
fmt = '.2f'
for i, j in itertools.product(range(CM_n.shape[0]), range(CM_n.shape[1])):
        plt.text(j, i, format(CM_n[i, j], fmt),horizontalalignment="center",verticalalignment='center',
                color="white" if CM_n[i, j] < 0.50 else "black")
plt.savefig('tri_p_CM.pdf',bbox_inches="tight")
plt.show()

# %%
row_class=10
col_class=0

# %%
kk=numpy.intersect1d(numpy.where(test_Y_index == row_class)[0],numpy.where( test_Y_predicted == col_class)[0])
print(kk)

# %%
test_id=ann_data.get_test_id()
kk2=test_id[kk]

# %%
id_df.loc[id_df['sec_code'].isin(kk2)]

# %%
#numpy.intersect1d(numpy.where(test_Y_index == 1)[0],numpy.where( test_Y_predicted == 1)[0])
