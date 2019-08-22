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
(train_X,train_Y)=ann_data.get_formated_train('all')

# %%
(test_X,test_Y)=ann_data.get_formated_test('all')

# %%
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Dropout
from keras.models import load_model
from sklearn.metrics import classification_report
from keras.optimizers import Adam
import numpy
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

# %%
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# %%
ff_num=train_X.shape[1]
model = Sequential()
opt=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.add(Dense(ff_num, input_dim=ff_num, kernel_initializer='random_uniform',activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(11,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
print(model.summary())

# %%
from collections import Counter
train_Y_index = train_Y.argmax(axis=1)
train_count=Counter(train_Y_index)
train_w_temp=[train_Y.shape[0]/train_count[i] for i in range(0,11,1)]
train_weights = dict(zip(range(0,11,1),train_w_temp) )
print(train_weights)


# %%
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5, min_delta=0.02 )
mc = ModelCheckpoint(os.path.join(phage_init.model_dir,'test_single.h5'), monitor='loss', 
                         mode='min', save_best_only=True, verbose=1)

# %%
model.fit(train_X, train_Y, epochs=100,verbose=1, batch_size=5000,class_weight=train_weights,callbacks=[es,mc])

# %%
K.clear_session()
del model

# %%
from collections import Counter
def train_model(model_name):
    (train_X,train_Y)=ann_data.get_formated_train(model_name)
    train_Y_index = train_Y.argmax(axis=1)
    f_num=train_X.shape[1]
    train_count=Counter(train_Y_index)
    train_w_temp=[train_Y.shape[0]/train_count[i] for i in range(0,11,1)]
    train_weights = dict(zip(range(0,11,1),train_w_temp) )
    #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, min_delta=0.02 )
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5, min_delta=0.02 )
    mc = ModelCheckpoint(os.path.join(phage_init.model_dir,model_name+'_single.h5'), monitor='loss', 
                         mode='min', save_best_only=True, verbose=1)
    model = Sequential()
    opt=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.add(Dense(f_num, input_dim=f_num, kernel_initializer='random_uniform',activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(200,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(200,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(11,activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit(train_X, train_Y, epochs=90, batch_size=5000, verbose=1,class_weight=train_weights,callbacks=[es,mc]
              #validation_split=0.1,
              )
    #model.save( os.path.join(phage_init.model_dir,model_name+'_single.h5') )
    K.clear_session()
    del model


# %%
all_models=['tetra_sc','tetra_sc_p','di','di_p','tri','tri_p','di_sc','di_sc_p','tri_sc','tri_sc_p','all']
#all_models=['all']
for this_model in all_models:
    print('training ' + this_model)
    train_model(this_model)
