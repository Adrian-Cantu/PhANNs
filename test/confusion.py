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
from keras import backend as K
from sklearn.metrics import classification_report
import numpy
#load the saved matrices
import pickle
import keras

# %%
(test_X,test_Y)=ann_data.get_formated_test("tetra_sc_tri_p")

# %%
pickle.dump(test_X, open( os.path.join(phage_init.model_dir,"web_test_X.p"), "wb" ) )

# %%
test_X.shape

# %%
#stop using gpu to not run into memoru issues
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# %%

n_members = 10
models = list()
yhats = numpy.empty((test_X.shape[0],10,11), dtype=numpy.float)
for model_number in range(n_members):
    # load model
        print('loading ...' + os.path.join(phage_init.model_dir,'tetra_sc_tri_p_'+"{:02d}".format(model_number+1)+'.h5'))
        model =  load_model( os.path.join(phage_init.model_dir,'tetra_sc_tri_p_'+"{:02d}".format(model_number+1)+'.h5') )
    # store in memory
        models.append(model)
    #row=model.predict(test_X)
    #yhats[model_number,:]=row
    #K.clear_session()

# %%
#model =  load_model( os.path.join(phage_init.model_dir,'tetra_sc_tri_p_01.h5') )

# %%
#yhats = numpy.empty((test_X.shape[0],10,11), dtype=numpy.float)

# %%
#row=model.predict(test_X)

# %%
#row.shape

# %%
pickle.dump(models, open( os.path.join(phage_init.model_dir,"deca_model.p"), "wb" ) )

# %%
test_Y_index = test_Y.argmax(axis=1)

# %%
yhats = [model.predict(test_X) for model in models]
yhats_v=numpy.array(yhats)
predicted_Y=numpy.sum(yhats_v, axis=0)
predicted_Y_index = numpy.argmax(predicted_Y, axis=1)

# %%
labels_names=["Major capsid","Minor capsid","Baseplate","Major tail","Minor tail","Portal","Tail fiber",
             "Tail shaft","Collar","Head-Tail joining","Others"]
print(classification_report(test_Y_index, predicted_Y_index, target_names=labels_names ))

# %%
from collections import Counter
zz=Counter(test_Y_index)
sample_w=[zz[i] for i in range(0,11,1)]
print(zz)
print(sample_w)
print()

# %%
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
plt.show()
CM=confusion_matrix(test_Y_index, predicted_Y_index)
CM_n=CM/numpy.array(sample_w)[:,None]
scale_up=1.4
plt.figure(figsize=[6.4*scale_up, 4.8*scale_up])
plt.imshow(CM_n, interpolation='nearest')
#plt.title('CM all')
plt.colorbar()
tick_marks = numpy.arange(len(labels_names))
plt.xticks(tick_marks, labels_names, rotation=90,fontsize='15')
plt.yticks(tick_marks, labels_names,fontsize='15')
fmt = '.2f'
for i, j in itertools.product(range(CM_n.shape[0]), range(CM_n.shape[1])):
        plt.text(j, i, format(CM_n[i, j], fmt),horizontalalignment="center",verticalalignment='center',
                color="white" if CM_n[i, j] < 0.50 else "black")
plt.savefig('tetra_sc_tri_p_CM.png',bbox_inches="tight")
plt.show()
