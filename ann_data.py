import os
import sys
import phage_init
import numpy

import pickle
final=pickle.load(open( os.path.join(phage_init.data_dir,"zscore_all_final.p"), "rb" ))

tt=200000  
f_num=final.shape[1]-11
train_id=final[0:tt,0]
train_X_total=final[0:tt,1:f_num]
train_Y_total=final[0:tt,f_num:]
test_id=final[tt:,0]
test_X_total=final[tt:,1:f_num]
test_Y_total=final[tt:,f_num:]





di_train       = train_X_total[:,0:400]
tri_train      = train_X_total[:,400:8400]
di_sc_train    = train_X_total[:,8400:8449] 
tri_sc_train   = train_X_total[:,8449:8792]
tetra_sc_train = train_X_total[:,8792:11193]
tt_train       = train_X_total[:,11193:]

di_test        = test_X_total[:,0:400]
tri_test       = test_X_total[:,400:8400]
di_sc_test     = test_X_total[:,8400:8449] 
tri_sc_test    = test_X_total[:,8449:8792]
tetra_sc_test  = test_X_total[:,8792:11193]
tt_test        = test_X_total[:,11193:]

def get_formated_train(model_name):
    train_x=[]
    train_y=train_Y_total
    if model_name == 'di':
        train_x=di_train
    elif model_name == 'di_p':
        train_x=numpy.concatenate((di_train,tt_train),axis=1)
    elif model_name == 'tri':
        train_x=tri_train
    elif model_name == 'tri_p':
        train_x=numpy.concatenate((tri_train,tt_train),axis=1)
    elif model_name == 'di_sc':
        train_x=di_sc_train
    elif model_name == 'di_sc_p':
        train_x=numpy.concatenate((di_sc_train,tt_train),axis=1)
    elif model_name == 'tri_sc':
        train_x=tri_sc_train
    elif model_name == 'tri_sc_p':
        train_x=numpy.concatenate((tri_sc_train,tt_train),axis=1)
    elif model_name == 'tetra_sc':
        train_x=tetra_sc_train
    elif model_name == 'tetra_sc_p':
        train_x=numpy.concatenate((tetra_sc_train,tt_train),axis=1)
    elif model_name == 'all':
        train_x=train_X_total
    return (train_x,train_y)

def get_formated_test(model_name):
    test_x=[]
    test_y=test_Y_total
    if model_name == 'di':
        test_x=di_test
    elif model_name == 'di_p':
        test_x=numpy.concatenate((di_test,tt_test),axis=1)
    elif model_name == 'tri':
        test_x=tri_test
    elif model_name == 'tri_p':
        test_x=numpy.concatenate((tri_test,tt_test),axis=1)
    elif model_name == 'di_sc':
        test_x=di_sc_test
    elif model_name == 'di_sc_p':
        test_x=numpy.concatenate((di_sc_test,tt_test),axis=1)
    elif model_name == 'tri_sc':
        test_x=tri_sc_test
    elif model_name == 'tri_sc_p':
        test_x=numpy.concatenate((tri_sc_test,tt_test),axis=1)
    elif model_name == 'tetra_sc':
        test_x=tetra_sc_test
    elif model_name == 'tetra_sc_p':
        test_x=numpy.concatenate((tetra_sc_test,tt_test),axis=1)
    elif model_name == 'all':
        test_x=test_X_total
    return (test_x,test_y)


 
