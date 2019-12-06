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
#import keras and numpy
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Dropout
from keras.optimizers import Adam
from keras import backend as K
import pickle
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report

# %%
import numpy
import itertools
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy import stats
from Bio.Alphabet import IUPAC

# %%


with open(os.path.join(phage_init.data_dir,"informative_kmer_re.txt")) as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content] 



# %%
all_fasta=(os.path.join(phage_init.fasta_dir,"major_capsid_all_clustered.fasta"),os.path.join(phage_init.fasta_dir,"minor_capsid_all_clustered.fasta"),
           os.path.join(phage_init.fasta_dir,"baseplate_all_clustered.fasta"),os.path.join(phage_init.fasta_dir,"major_tail_all_clustered.fasta"),
           os.path.join(phage_init.fasta_dir,"minor_tail_all_clustered.fasta"),os.path.join(phage_init.fasta_dir,"portal_all_clustered.fasta"),
           os.path.join(phage_init.fasta_dir,"tail_fiber_all_clustered.fasta"),os.path.join(phage_init.fasta_dir,"shaft_all_clustered.fasta"),
           os.path.join(phage_init.fasta_dir,"collar_all_clustered.fasta"),os.path.join(phage_init.fasta_dir,"HTJ_all_clustered.fasta"),
           os.path.join(phage_init.fasta_dir,"others_all_clustered.fasta"))

# %%
import re
import pandas as pd
def extract_all_re(fasta_list,re_list):
    d = {'seq_description': [], 'seq_id': [], "sec_code":[]}
    sec_code=0
    df = pd.DataFrame(data=d)
    total_fasta=0
    for file in fasta_list:
        for record in SeqIO.parse(file, "fasta"):
            total_fasta+=1
    prot_class=0;
    arr = numpy.empty((total_fasta,len(re_list)), dtype=numpy.int)
    id_arr = numpy.empty((total_fasta), dtype=numpy.int)
    class_arr = numpy.empty((total_fasta), dtype=numpy.int)
    this_prot=0
    for file in fasta_list:
        print('####################' + file)
        for record in SeqIO.parse(file, "fasta"):
            ll=len(record.seq)
            re_match_count=[len(re.findall(x,str(record.seq))) for x in re_list]



            #arr = numpy.append(arr,cat_n , axis=0)
            #class_arr = numpy.append(class_arr,prot_class)
            #id_arr = numpy.append(id_arr,sec_code)
            arr[sec_code,:]=re_match_count
            class_arr[sec_code]=prot_class
            id_arr[sec_code]=sec_code
            
            data_row=[record.description,record.id,int(sec_code)]
            df=df.append(pd.Series(data_row,index=df.columns),sort=False,ignore_index=True)
            sec_code+=1
            this_prot+=1
            if (this_prot%50==0):
                print("processing sequence # " + str(this_prot),end="\r")
            
        prot_class+=1
        this_prot=0
    return (arr,class_arr,id_arr,df)
    


# %%
one_fasta=[os.path.join(phage_init.fasta_dir,"minor_capsid_all_clustered.fasta")]
#print(one_fasta)
#(arr,class_arr,id_arr,df)=extract_all_re(one_fasta,content)
(arr,class_arr,id_arr,df)=extract_all_re(all_fasta,content)

# %%
numpy.count_nonzero(arr)

# %%
import pickle
pickle.dump(arr, open( os.path.join(phage_init.data_dir,"re_raw_arr.p"), "wb" ),protocol=4 )
pickle.dump(class_arr, open( os.path.join(phage_init.data_dir,"re_raw_class_arr.p"), "wb" ),protocol=4 )
pickle.dump(id_arr, open( os.path.join(phage_init.data_dir,"re_raw_id_arr.p"), "wb" ),protocol=4 )
pickle.dump(df, open( os.path.join(phage_init.data_dir,"re_raw_df.p"), "wb" ),protocol=4 )

# %%
nb_classes = 11
one_hot_targets = numpy.eye(nb_classes)[class_arr]

# %%
final = numpy.concatenate((id_arr.reshape((id_arr.shape[0],1)), arr, one_hot_targets), axis=1)

# %%
del arr

# %%
numpy.random.shuffle(final)

# %%
pickle.dump(final, open( os.path.join(phage_init.data_dir,"re_all_final.p"), "wb" ),protocol=4 )

# %%
tt=200000  
f_num=final.shape[1]-11
train_id=final[0:tt,0]
train_X_total=final[0:tt,1:f_num]
train_Y_total=final[0:tt,f_num:]
test_id=final[tt:,0]
test_X_total=final[tt:,1:f_num]
test_Y_total=final[tt:,f_num:]

# %%
pickle.dump(train_X_total, open( os.path.join(phage_init.data_dir,"re_train_X.p"), "wb" ),protocol=4 )
pickle.dump(test_X_total , open( os.path.join(phage_init.data_dir,"re_test_X.p" ), "wb" ),protocol=4 )
pickle.dump(test_Y_total,open( os.path.join(phage_init.data_dir,"re_test_Y.p"), "wb" ),protocol=4 )
pickle.dump(train_Y_total,open( os.path.join(phage_init.data_dir,"re_train_Y.p"), "wb" ),protocol=4 )
pickle.dump(test_id,open( os.path.join(phage_init.data_dir,"re_test_id.p"), "wb" ),protocol=4 )
pickle.dump(train_id,open( os.path.join(phage_init.data_dir,"re_train_id.p"), "wb" ),protocol=4 )
