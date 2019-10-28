# %%
import os
import sys
import phage_init
import numpy

# %%
import pickle
import itertools
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
    elif model_name == 'tetra_sc_tri_p':
        train_x=numpy.concatenate((pickle.load(open( os.path.join(phage_init.data_dir,"tetra_sc_train.p"), "rb" )),pickle.load(open( os.path.join(phage_init.data_dir,"tri_train.p"), "rb" )),pickle.load(open( os.path.join(phage_init.data_dir,"tt_train.p"), "rb" ))),axis=1)
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
    elif model_name == 'tetra_sc_tri_p':
        test_x=numpy.concatenate((pickle.load(open( os.path.join(phage_init.data_dir,"tetra_sc_test.p"), "rb" )),pickle.load(open( os.path.join(phage_init.data_dir,"tri_test.p"), "rb" )),pickle.load(open( os.path.join(phage_init.data_dir,"tt_test.p"), "rb" ))),axis=1)
    elif model_name == 'all':
        test_x=numpy.concatenate((pickle.load(open( os.path.join(phage_init.data_dir,"di_test.p"), "rb" )),pickle.load(open( os.path.join(phage_init.data_dir,"tri_test.p"), "rb" )),pickle.load(open( os.path.join(phage_init.data_dir,"di_sc_test.p"), "rb" )),pickle.load(open( os.path.join(phage_init.data_dir,"tri_sc_test.p"), "rb" )),pickle.load(open( os.path.join(phage_init.data_dir,"tetra_sc_test.p"), "rb" )),pickle.load(open( os.path.join(phage_init.data_dir,"tt_test.p"), "rb" ))),axis=1)
    return (test_x,test_y)


# %%
def get_train_id():
    train_id=pickle.load(open( os.path.join(phage_init.data_dir,"train_id.p"), "rb" ))
    return train_id

def get_test_id():
    test_id=pickle.load(open( os.path.join(phage_init.data_dir,"test_id.p"), "rb" ))
    return test_id

def get_feature_names(model_name):
    AA=["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]
    SC=["1","2","3","4","5","6","7"]
    di_pep = [''.join(i) for i in itertools.product(AA, repeat = 2)]
    tri_pep = [''.join(i) for i in itertools.product(AA, repeat = 3)]
    di_sc = [''.join(i) for i in itertools.product(SC, repeat = 2)]
    tri_sc = [''.join(i) for i in itertools.product(SC, repeat = 3)]
    tetra_sc = [''.join(i) for i in itertools.product(SC, repeat = 4)]
    extra=["IP","In_index","lenght","aromaticity","molar_extinction_1","molar_extrinction_2","gravy","weight"]
    f_index=[]
    if model_name == 'di':
        f_index=di_pep
    elif model_name == 'di_p':
        f_index=numpy.concatenate((di_pep,extra))
    elif model_name == 'tri':
        f_idex=tri_pep
    elif model_name == 'tri_p':
        f_index=numpy.concatenate((tri_pep,extra))
    elif model_name == 'di_sc':
        f_index=di_sc
    elif model_name == 'di_sc_p':
        f_index=numpy.concatenate((di_sc,extra))
    elif model_name == 'tri_sc':
        f_index=tri_sc
    elif model_name == 'tri_sc_p':
        f_index=numpy.concatenate((tri_sc,extra))
    elif model_name == 'tetra_sc':
        f_index=tetra_sc
    elif model_name == 'tetra_sc_p':
        f_index=numpy.concatenate((tetra_sc,extra))
    elif model_name == 'tetra_sc_tri_p':
        f_index=numpy.concatenate((tetra_sc,tri_pep,extra))
    elif model_name == 'all':
        f_index=numpy.concatenate((di_pep,tri_pep,di_sc,tri_sc,tetra_sc,extra))
    return f_index
