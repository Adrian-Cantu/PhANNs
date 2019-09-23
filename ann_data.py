# %%
import os
import sys
import phage_init
import numpy

# %%
import pickle
#final=pickle.load(open( os.path.join(phage_init.data_dir,"zscore_all_final.p"), "rb" ))


# %%
#pickle.load(open( os.path.join(phage_init.data_dir,"zscore_all_final.p"), "rb" ))

# %%
def get_formated_train(model_name):
    train_x=[]
    train_y=pickle.load(open( os.path.join(phage_init.data_dir,"train_Y.p"), "rb" ))
    if model_name == 'di':
        train_x=pickle.load(open( os.path.join(phage_init.data_dir,"di_train.p"), "rb" ))
    elif model_name == 'di_p':
        train_x=numpy.concatenate((pickle.load(open( os.path.join(phage_init.data_dir,"di_train.p"), "rb" )),pickle.load(open( os.path.join(phage_init.data_dir,"tt_train.p"), "rb" ))),axis=1)
    elif model_name == 'tri':
        train_x=pickle.load(open( os.path.join(phage_init.data_dir,"tri_train.p"), "rb" ))
    elif model_name == 'tri_p':
        train_x=numpy.concatenate((pickle.load(open( os.path.join(phage_init.data_dir,"tri_train.p"), "rb" )),pickle.load(open( os.path.join(phage_init.data_dir,"tt_train.p"), "rb" ))),axis=1)
    elif model_name == 'di_sc':
        train_x=pickle.load(open( os.path.join(phage_init.data_dir,"di_sc_train.p"), "rb" ))
    elif model_name == 'di_sc_p':
        train_x=numpy.concatenate((pickle.load(open( os.path.join(phage_init.data_dir,"di_sc_train.p"), "rb" )),pickle.load(open( os.path.join(phage_init.data_dir,"tt_train.p"), "rb" ))),axis=1)
    elif model_name == 'tri_sc':
        train_x=pickle.load(open( os.path.join(phage_init.data_dir,"tri_sc_train.p"), "rb" ))
    elif model_name == 'tri_sc_p':
        train_x=numpy.concatenate((pickle.load(open( os.path.join(phage_init.data_dir,"tri_sc_train.p"), "rb" )),pickle.load(open( os.path.join(phage_init.data_dir,"tt_train.p"), "rb" ))),axis=1)
    elif model_name == 'tetra_sc':
        train_x=pickle.load(open( os.path.join(phage_init.data_dir,"tetra_sc_train.p"), "rb" ))
    elif model_name == 'tetra_sc_p':
        train_x=numpy.concatenate((pickle.load(open( os.path.join(phage_init.data_dir,"tetra_sc_train.p"), "rb" )),pickle.load(open( os.path.join(phage_init.data_dir,"tt_train.p"), "rb" ))),axis=1)
    elif model_name == 'all':
        train_x=train_x=numpy.concatenate((pickle.load(open( os.path.join(phage_init.data_dir,"di_train.p"), "rb" )),pickle.load(open( os.path.join(phage_init.data_dir,"tri_train.p"), "rb" )),pickle.load(open( os.path.join(phage_init.data_dir,"di_sc_train.p"), "rb" )),pickle.load(open( os.path.join(phage_init.data_dir,"tri_sc_train.p"), "rb" )),pickle.load(open( os.path.join(phage_init.data_dir,"tetra_sc_train.p"), "rb" )),pickle.load(open( os.path.join(phage_init.data_dir,"tt_train.p"), "rb" ))),axis=1)
    return (train_x,train_y)

# %%
def get_formated_test(model_name):
    test_x=[]
    test_y=pickle.load(open( os.path.join(phage_init.data_dir,"test_Y.p"), "rb" ))
    if model_name == 'di':
        test_x=pickle.load(open( os.path.join(phage_init.data_dir,"di_test.p"), "rb" ))
    elif model_name == 'di_p':
        test_x=numpy.concatenate((pickle.load(open( os.path.join(phage_init.data_dir,"di_test.p"), "rb" )),pickle.load(open( os.path.join(phage_init.data_dir,"tt_test.p"), "rb" ))),axis=1)
    elif model_name == 'tri':
        test_x=pickle.load(open( os.path.join(phage_init.data_dir,"tri_test.p"), "rb" ))
    elif model_name == 'tri_p':
        test_x=numpy.concatenate((pickle.load(open( os.path.join(phage_init.data_dir,"tri_test.p"), "rb" )),pickle.load(open( os.path.join(phage_init.data_dir,"tt_test.p"), "rb" ))),axis=1)
    elif model_name == 'di_sc':
        test_x=pickle.load(open( os.path.join(phage_init.data_dir,"di_sc_test.p"), "rb" ))
    elif model_name == 'di_sc_p':
        test_x=numpy.concatenate((pickle.load(open( os.path.join(phage_init.data_dir,"di_sc_test.p"), "rb" )),pickle.load(open( os.path.join(phage_init.data_dir,"tt_test.p"), "rb" ))),axis=1)
    elif model_name == 'tri_sc':
        test_x=pickle.load(open( os.path.join(phage_init.data_dir,"tri_sc_test.p"), "rb" ))
    elif model_name == 'tri_sc_p':
        test_x=numpy.concatenate((pickle.load(open( os.path.join(phage_init.data_dir,"tri_sc_test.p"), "rb" )),pickle.load(open( os.path.join(phage_init.data_dir,"tt_test.p"), "rb" ))),axis=1)
    elif model_name == 'tetra_sc':
        test_x=pickle.load(open( os.path.join(phage_init.data_dir,"tetra_sc_test.p"), "rb" ))
    elif model_name == 'tetra_sc_p':
        test_x=numpy.concatenate((pickle.load(open( os.path.join(phage_init.data_dir,"tetra_sc_test.p"), "rb" )),pickle.load(open( os.path.join(phage_init.data_dir,"tt_test.p"), "rb" ))),axis=1)
    elif model_name == 'all':
        test_x=test_x=numpy.concatenate((pickle.load(open( os.path.join(phage_init.data_dir,"di_test.p"), "rb" )),pickle.load(open( os.path.join(phage_init.data_dir,"tri_test.p"), "rb" )),pickle.load(open( os.path.join(phage_init.data_dir,"di_sc_test.p"), "rb" )),pickle.load(open( os.path.join(phage_init.data_dir,"tri_sc_test.p"), "rb" )),pickle.load(open( os.path.join(phage_init.data_dir,"tetra_sc_test.p"), "rb" )),pickle.load(open( os.path.join(phage_init.data_dir,"tt_test.p"), "rb" ))),axis=1)
    return (test_x,test_y)


# %%
def get_train_id():
    return train_id

def get_test_id():
    return test_id

