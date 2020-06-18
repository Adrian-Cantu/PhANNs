# %%
#import keras and numpy
import numpy
import pickle
import os


# %%
def get_train(model_name,model_number,class_arr,group_arr):
    kk=numpy.array(range(10))
    my_list=[pickle.load(open( os.path.join('06_features',str(x+1) +'_'+ model_name +'.p'),'rb')) for x in kk[kk!=model_number] ]
    my_cat_arr=numpy.concatenate(my_list,axis=0)
    del my_list
    Y_index  = class_arr[(group_arr!=model_number) & (group_arr!=10)]
    return (my_cat_arr,Y_index)

# %%
def get_validation(model_name,model_number,class_arr,group_arr):
    my_arr=pickle.load(open( os.path.join('06_features',str(model_number+1)+'_' + model_name +'.p'),'rb'))
    Y_index  = class_arr[(group_arr==model_number) & (group_arr!=10)]
    return (my_arr,Y_index)


# %%
def get_test(model_name,class_arr,group_arr):
    my_arr=pickle.load(open( os.path.join('06_features','11_' + model_name +'.p'),'rb'))
    Y_index  = class_arr[(group_arr==10)]
    return (my_arr,Y_index)
