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
#import keras and numpy
import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import pickle
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report
import os
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import load_model

# %%
import get_arr

# %%
import pandas as pd
d = {'model': [], 'class': [],'score_type':[],'value':[]}
df = pd.DataFrame(data=d)
df_val=pd.DataFrame(data=d)
df_acc=pd.DataFrame(data=d)

# %%
#di_n=400
#tri_n=8000
#di_sc_n=49
#tri_sc_n=343
#tetra_sc_n=2401
#g_tetra_inf_n=arr_2.shape[1]
#p_n=8
#di_end=di_n
#tri_end=di_end+tri_n
#di_sc_end=tri_end+di_sc_n
#tri_sc_end=di_sc_end+tri_sc_n
#tetra_sc_end=tri_sc_end+tetra_sc_n

# %%
#di_range=numpy.r_[:di_end]
#tri_range=numpy.r_[di_end:tri_end]
#di_sc_range=numpy.r_[tri_end:di_sc_end]
#tri_sc_range=numpy.r_[di_sc_end:tri_sc_end]
#tetra_sc_range=numpy.r_[tri_sc_end:tetra_sc_end]
#p_range=numpy.r_[tetra_sc_end:tetra_sc_end+p_n]
#di_sc_p_range=numpy.r_[di_sc_range,p_range]
#tri_sc_p_range=numpy.r_[tri_sc_range,p_range]
#tetra_sc_p_range=numpy.r_[tetra_sc_range,p_range]
#di_p_range=numpy.r_[di_range,p_range]
#tri_p_range=numpy.r_[tri_range,p_range]
#tetra_sc_tri_p_range=numpy.r_[tetra_sc_range,tri_range,p_range]
#all_range=numpy.r_[:tetra_sc_end+p_n]

# %%
#all_ranges=[di_sc_range,di_sc_p_range,tri_sc_range,tri_sc_p_range,tetra_sc_range,tetra_sc_p_range,di_range,di_p_range,tri_range,tri_p_range,tetra_sc_tri_p_range,all_range]

# %%
#all_models=['di_sc','di_sc_p','tri_sc','tri_sc_p','tetra_sc','tetra_sc_p','di','di_p','tri','tri_p','tetra_sc_tri_p','all']
all_models=['di_sc','tetra_sc_tri_p']
#all_models=['di_sc','di_sc_p']


# %%
#arr_z=pickle.load(open( os.path.join('data',"all_data.p"), "rb" ))
data_dir='06_features'
group_arr=pickle.load(open( os.path.join(data_dir,"group_arr.p"), "rb" ))
class_arr=pickle.load(open( os.path.join(data_dir,"class_arr.p"), "rb" ))


# %%
def add_to_df(df,test_Y_index, test_Y_predicted,model_name):
    labels_names=["Major capsid","Baseplate","Major tail","Minor tail","Portal","Tail fiber",
             "Tail shaft","Collar","Head-Tail joining","Others"]
    labels_dataframe=["Major capsid","Baseplate","Major tail","Minor tail","Portal","Tail fiber",
                 "Tail shaft","Collar","Head-Tail joining","Others","weighted avg"]
    report=classification_report(test_Y_index, test_Y_predicted, target_names=labels_names,output_dict=True )
    for label in labels_dataframe:
        #data_row=[report[label][i] for i in ['precision','recall',"f1-score"]]
        #data_row.insert(0,label)
        #data_row.insert(0,model_name)
        score_type='precision'
        data_row=[model_name,label,score_type,report[label][score_type]]
        df=df.append(pd.Series(data_row,index=df.columns),sort=False,ignore_index=True)
        score_type='recall'
        data_row=[model_name,label,score_type,report[label][score_type]]
        df=df.append(pd.Series(data_row,index=df.columns),sort=False,ignore_index=True)
        score_type='f1-score'
        data_row=[model_name,label,score_type,report[label][score_type]]
        df=df.append(pd.Series(data_row,index=df.columns),sort=False,ignore_index=True)
    return df


# %%
from matplotlib import pyplot
num_of_class=10
def train_kfold(model_name,df,df_val,df_acc,class_arr,group_arr):
    for model_number in range(10):
        print("Doing cross validation on "+model_name)
        (train_X,train_Y_index)=get_arr.get_train(model_name,model_number,class_arr,group_arr)
        (test_X,  test_Y_index)=get_arr.get_validation(model_name,model_number,class_arr,group_arr)
        #train_Y_index = class_arr[(group_arr!=model_number) & (group_arr!=10)]
        #test_Y_index  = class_arr[(group_arr==model_number) & (group_arr!=10)]
        f_num=train_X.shape[1]
        print('\tModel '+ str(model_number+1))
        train_Y = numpy.eye(num_of_class)[train_Y_index]
        test_Y  = numpy.eye(num_of_class)[test_Y_index]
        print(test_X.shape)
        print(test_Y.shape)
        es = EarlyStopping(monitor='loss', mode='min', verbose=2, patience=5, min_delta=0.02 )
        mc = ModelCheckpoint(os.path.join('07_models',model_name+'_val_'+"{:02d}".format(model_number)+'.h5'),
                        monitor='val_loss', mode='min', save_best_only=True, verbose=1)
        mc2 = ModelCheckpoint(os.path.join('07_models',model_name+'_acc_'+"{:02d}".format(model_number)+'.h5'),
                        monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
        kk=compute_class_weight('balanced',range(num_of_class),train_Y_index)
        train_weights=dict(zip(range(num_of_class),kk))
        model = Sequential()
        opt=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False)
        model.add(Dense(f_num, input_dim=f_num, kernel_initializer='random_uniform',activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(200,activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(200,activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(num_of_class,activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        print(train_weights)
        history = model.fit(train_X, train_Y, validation_data=(test_X, test_Y) , epochs=120, 
                  batch_size=5000, verbose=2,class_weight=train_weights,callbacks=[es,mc,mc2])
        
        
        test_Y_predicted = model.predict_classes(test_X)
        df=add_to_df(df,test_Y_index, test_Y_predicted,model_name)
        model.save( os.path.join('07_models',model_name+'_'+"{:02d}".format(model_number)+'.h5'))

        model_val=load_model(os.path.join('07_models',model_name+'_val_'+"{:02d}".format(model_number)+'.h5'))
        test_Y_predicted = model_val.predict_classes(test_X)
        df_val=add_to_df(df_val,test_Y_index, test_Y_predicted,model_name)
        
        model_acc=load_model(os.path.join('07_models',model_name+'_acc_'+"{:02d}".format(model_number)+'.h5'))
        test_Y_predicted = model_acc.predict_classes(test_X)
        df_acc=add_to_df(df_acc,test_Y_index, test_Y_predicted,model_name)
        
#        scores = model.evaluate(train_X[test], test_YY, verbose=0)
#        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#        cvscores.append(scores[1] * 100)
#        model.save( os.path.join(phage_init.model_dir,model_name+'_'+"{:02d}".format(model_number)+'.h5') )
#        pickle.dump(train_X[test], open( os.path.join(phage_init.kfold_dir,"{:02d}".format(model_number)+'_test_X.p' ), "wb"))
#        pickle.dump(test_YY, open( os.path.join(phage_init.kfold_dir,"{:02d}".format(model_number)+'_test_Y.p' ), "wb"))
        
        K.clear_session()
        
        
        #_, train_acc = model.evaluate(train_X, train_Y, verbose=0)
        #_, test_acc = model.evaluate(test_X, test_Y, verbose=0)
        #print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# plot training history
#        pyplot.plot(history.history['loss'], label='train')
#        pyplot.plot(history.history['val_loss'], label='test')
#        pyplot.plot(history.history['val_accuracy'], label='acc')
#        pyplot.legend()
#        pyplot.show()
        del model_acc
        del model_val
        del model
        del train_X
        del test_X
    return (df,df_val,df_acc)
#    return 1

# %%
for model_name in all_models:
    (df,df_val,df_acc)=train_kfold(model_name,df,df_val,df_acc,class_arr,group_arr)

# %%
pickle.dump(df,  open( os.path.join('07_models',"all_results_df.p") , "wb" ), protocol=4 )
pickle.dump(df_val,  open( os.path.join('07_models',"all_results_df_val.p") , "wb" ), protocol=4 )
pickle.dump(df_acc,  open( os.path.join('07_models',"all_results_df_acc.p") , "wb" ), protocol=4 )
