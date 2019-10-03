# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
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
import pickle
import random

# %%
#import ann_data
test_X=pickle.load(open( os.path.join(phage_init.data_dir,"test_X.p"), "rb" ))
test_Y_index=pickle.load(open( os.path.join(phage_init.data_dir,"test_Y_index.p"), "rb" ))

# %%
import pandas as pd
#d = {'model': [], 'class': [],'precision':[],'recall':[],'f1-score':[]}
d = {'model': [], 'class': [],'score_type':[],'value':[]}
df = pd.DataFrame(data=d)
#F = open('all_models_table.txt','w') 

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

# %%
model = load_model( os.path.join(phage_init.model_dir,'tri_p_single.h5') )


# %%
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

# %%
rc = test_X.shape

num_trials = 4
column_count = rc[1]
#column_count = 3

seeded_random = False

print("Loading predicted model:")
test_Y_predicted = model.predict_classes(test_X,verbose=1)
test_X_copy = test_X.copy()



for nt in range(num_trials):
    #create an array containing the difference in errors between the normal set and permuted set
    error_data = numpy.ndarray(shape=(1,column_count), dtype = int)
    print("")
    for column in range(column_count):
        for index in range(rc[0]):
            if seeded_random == True:
                random.seed((column+1)*(index+1)*(nt+1))
            swap = random.randint(0,rc[0]-1)
            test_X_copy[index,column], test_X_copy[swap,column] = test_X_copy[swap,column], test_X_copy[index,column]

        if(column % 10 == 0):
            pcomplete = round((float(column)/column_count),4)*100
            print("Trial " + str(nt) + ": " + str(pcomplete) + "%")    
        test_Y_varied = model.predict_classes(test_X_copy,verbose=0)


        predicted_shape = test_Y_predicted.shape
        numerrors_i = 0
        for index in range(predicted_shape[0]):
            if test_Y_index[index,] != test_Y_predicted[index,]:
                numerrors_i+=1

        predicted_shape = test_Y_varied.shape
        numerrors_v = 0
        for index in range(predicted_shape[0]):
            if test_Y_index[index,] != test_Y_varied[index,]:
                numerrors_v+=1


        #positive difference -> more errors after permute
        #negative difference -> less errors after permute
        error_diff = numerrors_v - numerrors_i

        error_data[0,column] = error_diff 
        
        for replace in range(rc[0]):
            test_X_copy[replace,column] = test_X[replace,column]
        
        
    
        
        
    curr = os.getcwd()
    fls = os.listdir(curr + "/Trials/")
    numfiles = len(fls)-1
    filename = "error" + str(numfiles) + ".p"
    pickle.dump(error_data, open(os.path.join(phage_init.error_dir, filename), "wb"), protocol = 4)
        


# %%
error_data
error_data.shape
numpy.array_equal(test_X,test_X_copy)

# %%
error_average = numpy.ndarray(shape = (column_count))
numpy.mean(error_data.copy(), axis = 0, dtype = float, out = error_average)
error_average

# %%
print("Mean # of errors: " + str(error_average.mean(dtype = float)))
print("Standard deviation: " + str(error_average.std(dtype = float)))
print("Most # of errors created: " + str(error_average.max()) + " @ index " + str(error_average.argmax()))
print("Most # of errors reduced: " + str(error_average.min()) + " @ index " + str(error_average.argmin()))

# %%
sorted_indices = error_average.argsort()

#top_x contains an array with the indices of the sequences with the top x number of errors
x_wanted = 50
top_x_reverse = (sorted_indices.copy())[-x_wanted:]
top_x = top_x_reverse[::-1]

#bottom_x contains an array with the indices of the sequences with the top x number of errors
x_wanted = 50
bottom_x = (sorted_indices.copy())[:x_wanted]

print(top_x)
print(bottom_x)


# %%
sorted_errors = numpy.sort((error_average.copy()))
print(sorted_errors)
sorted_reverse = sorted_errors[::-1]
print(sorted_reverse)

# %%
import matplotlib.pyplot as plt
import seaborn as sns
sns.lineplot(numpy.arange(2000), sorted_reverse[0:2000])

# %%
