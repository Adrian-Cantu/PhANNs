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
import pickle
import pandas as pd
import os
import numpy
import subprocess

# %%
mean_final=pickle.load(open( os.path.join('06_features',"mean_final.p"), "rb" ))
std_final=pickle.load(open( os.path.join('06_features',"std_final.p"), "rb" ))

# %%
di_n=400
tri_n=8000
di_sc_n=49
tri_sc_n=343
tetra_sc_n=2401
#g_tetra_inf_n=arr_2.shape[1]
p_n=8
di_end=di_n
tri_end=di_end+tri_n
di_sc_end=tri_end+di_sc_n
tri_sc_end=di_sc_end+tri_sc_n
tetra_sc_end=tri_sc_end+tetra_sc_n

# %%
di_range=numpy.r_[:di_end]
tri_range=numpy.r_[di_end:tri_end]
di_sc_range=numpy.r_[tri_end:di_sc_end]
tri_sc_range=numpy.r_[di_sc_end:tri_sc_end]
tetra_sc_range=numpy.r_[tri_sc_end:tetra_sc_end]
p_range=numpy.r_[tetra_sc_end:tetra_sc_end+p_n]
di_sc_p_range=numpy.r_[di_sc_range,p_range]
tri_sc_p_range=numpy.r_[tri_sc_range,p_range]
tetra_sc_p_range=numpy.r_[tetra_sc_range,p_range]
di_p_range=numpy.r_[di_range,p_range]
tri_p_range=numpy.r_[tri_range,p_range]
tetra_sc_tri_p_range=numpy.r_[tetra_sc_range,tri_range,p_range]
all_range=numpy.r_[:tetra_sc_end+p_n]

# %%
pickle.dump(mean_final[tetra_sc_tri_p_range], open( os.path.join('..','web_server','deca_model','mean_part.p'), "wb" ) )
pickle.dump( std_final[tetra_sc_tri_p_range], open( os.path.join('..','web_server','deca_model', 'std_part.p'), "wb" ) )

# %%
model='val_'     #can be '', 'acc_' or 'val_'
for num in range(10):
    command = "cp 07_models/tetra_sc_tri_p_{}{:02d}.h5 ../web_server/deca_model/tetra_sc_tri_p_{:02d}.h5".format(model,num,num) 
    print(command)
    subprocess.run(command,shell=True,check=True, text=True)

# %%
command = "cp test_set_stats_val_loss.csv ../web_server/test_set_stats.csv"
print(command)
__=subprocess.run(command,shell=True,check=True, text=True)
