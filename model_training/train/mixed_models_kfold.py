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
from keras.optimizers import Adam
import numpy
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from collections import Counter
K.clear_session()

# %%
from collections import Counter
from sklearn.model_selection import StratifiedKFold
def train_kfold(model_name,train_X,train_Y):
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=43)
    #kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=43)
    cvscores = []
    model_number=0
    print("Doing cross validation on "+model_name)
    #(train_X,train_Y)=ann_data.get_formated_train(model_name)
    train_Y_index = train_Y.argmax(axis=1)
    f_num=train_X.shape[1]
    train_count=Counter(train_Y_index)
    train_w_temp=[train_Y.shape[0]/train_count[i] for i in range(0,11,1)]
    train_weights = dict(zip(range(0,11,1),train_w_temp) )
    for train, test in kfold.split(train_X, train_Y_index):
        model_number=model_number+1
        print('\tModel '+ str(model_number))
        train_YY = numpy.eye(11)[train_Y_index[train]]
        test_YY  = numpy.eye(11)[train_Y_index[test]]
        es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5, min_delta=0.02 )
        mc = ModelCheckpoint(os.path.join(phage_init.model_dir,model_name+'_'+"{:02d}".format(model_number)+'.h5'), monitor='loss', 
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
        model.fit(train_X[train], train_YY, epochs=120, batch_size=5000, verbose=1,
                  class_weight=train_weights,callbacks=[es,mc])
        test_Y_predicted = model.predict_classes(train_X[test])
#        df=add_to_df(df,train_Y_index[test], test_Y_predicted,this_model)
#        scores = model.evaluate(train_X[test], test_YY, verbose=0)
#        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#        cvscores.append(scores[1] * 100)
#        model.save( os.path.join(phage_init.model_dir,model_name+'_'+"{:02d}".format(model_number)+'.h5') )
#        pickle.dump(train_X[test], open( os.path.join(phage_init.kfold_dir,"{:02d}".format(model_number)+'_test_X.p' ), "wb"))
#        pickle.dump(test_YY, open( os.path.join(phage_init.kfold_dir,"{:02d}".format(model_number)+'_test_Y.p' ), "wb"))
        
        K.clear_session()
        del model
#    return df
    return 1


# %%
(train_X1,train_Y)=ann_data.get_formated_train("tetra_sc")
(train_X2,train_Y)=ann_data.get_formated_train("tri_p")

# %%
train_X=numpy.concatenate((train_X1,train_X2),axis=1)

# %% [raw]
# train_kfold('tetra_sc_tri_p',train_X,train_Y)
