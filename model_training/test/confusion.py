# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.0
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from sklearn.metrics import classification_report
import numpy
#load the saved matrices
import pickle
import tensorflow.keras as keras

# %%

# %%
#pickle.dump(test_X, open( os.path.join(phage_init.model_dir,"web_test_X.p"), "wb" ) )

# %%
#pickle.dump(test_Y, open( os.path.join(phage_init.model_dir,"web_test_Y.p"), "wb" ) )

# %%
#test_X.shape

# %%
#stop using gpu to not run into memoru issues
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# %%

n_members = 10
models = list()
#yhats = numpy.empty((test_X.shape[0],10,11), dtype=numpy.float)
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
(test_X,test_Y)=ann_data.get_formated_test("tetra_sc_tri_p")

# %%
test_Y.shape

# %%
test_Y_index = test_Y.argmax(axis=1)

# %%
(kk_X,kk_Y)=ann_data.get_formated_test("all")
(kkt_X,kkt_Y)=ann_data.get_formated_train("all")

# %%
kk_X.shape

# %%
kkt_X.shape

# %%
kk_X.shape[0]+kkt_X.shape[0]

# %%
#model.predict(test_X,verbose=1)


# %%
yhats = [model.predict(test_X,verbose=2) for model in models]

# %%

yhats_v=numpy.array(yhats)
predicted_Y=numpy.sum(yhats_v, axis=0)
predicted_Y_index = numpy.argmax(predicted_Y, axis=1)

# %%
pickle.dump(predicted_Y_index, open( os.path.join(phage_init.model_dir,"CM_predicted_test_Y.p"), "wb" ) )
pickle.dump(test_Y_index, open( os.path.join(phage_init.model_dir,"CM_test_Y.p"), "wb" ) )

# %%
labels_names=["Major capsid","Minor capsid","Baseplate","Major tail","Minor tail","Portal","Tail fiber",
             "Tail shaft","Collar","Head-Tail joining","Others"]
print(classification_report(test_Y_index, predicted_Y_index, target_names=labels_names ))

# %%

# %%
report=classification_report(test_Y_index, predicted_Y_index , target_names=labels_names,output_dict=True )

# %%
report

# %%
import pandas as pd
id_df=pickle.load(open( os.path.join(phage_init.data_dir,"raw_df.p"), "rb" ))

# %%
row_class=1
col_class=1

# %%
kk=numpy.intersect1d(numpy.where(test_Y_index == row_class)[0],numpy.where( predicted_Y_index == col_class)[0])
print(kk)

# %%
test_id=ann_data.get_test_id()
kk2=test_id[kk]

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
#plt.title('CM : ' + df)
#plt.title('CM all')
plt.colorbar()
tick_marks = numpy.arange(len(labels_names))
plt.xticks(tick_marks, labels_names, rotation=90,fontsize='15')
plt.yticks(tick_marks, labels_names,fontsize='15')
fmt = '.2f'
for i, j in itertools.product(range(CM_n.shape[0]), range(CM_n.shape[1])):
        plt.text(j, i, format(CM_n[i, j], fmt),horizontalalignment="center",verticalalignment='center',
                color="white" if CM_n[i, j] < 0.50 else "black")
plt.ylabel('True Class',fontsize='20')
plt.xlabel('Predicted Class',fontsize='20')
plt.clim(0,1)
plt.savefig('tetra_sc_tri_p_CM.png',bbox_inches="tight")
plt.show()

# %%
import pickle
import os
mf=pickle.load(open( os.path.join(phage_init.data_dir,"mean_final.p"), "rb" ))
sd=pickle.load(open( os.path.join(phage_init.data_dir,"std_final.p"), "rb" ))

# %%
mf_s=numpy.concatenate((mf[400:8400],mf[8792:11193],mf[-8:]))
sd_s=numpy.concatenate((sd[400:8400],sd[8792:11193],sd[-8:]))

# %%
pickle.dump(mf_s, open( os.path.join(phage_init.model_dir,"mean_part.p"), "wb" ))
pickle.dump(sd_s, open( os.path.join(phage_init.model_dir,"std_part.p"), "wb" ))
