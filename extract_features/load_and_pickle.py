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

# %% [markdown]
# ## load\_and\_pickle
# This scripts read the fasta files of the 11 classes, extract all features, and saves them in pickle files.
# raw files produced:
# 1. raw_arr.p - the 11201 features of each sequence.
# 2. raw_class_arr.p - the class of each sequence coded as an integer
# 3. raw_id_arr.p - the id (internal, not the one in the fasta file) of each sequence
# 4. raw_df.p - a datafram that contains the internal id, fasta id, and description of each sequence
#
# After z-scoring and shuffling the script produces:
# 1. mean_final.p - mean of each feature
# 2. std_final.p - standar deciation of each feature
# 3. zscore_all_final.p - A shuffled matrix with id, features z-score, and one-hot encoded class for each sequences. read using ann_data.py library 
#
#

# %%
import os
import sys
sys.path.append("..")
import phage_init

# %%
import numpy
import itertools
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy import stats
from Bio.Alphabet import IUPAC
#from itertools import permutations
import pandas as pd

# %%
all_fasta=(os.path.join(phage_init.fasta_dir,"major_capsid_all_clustered.fasta"),os.path.join(phage_init.fasta_dir,"minor_capsid_all_clustered.fasta"),
           os.path.join(phage_init.fasta_dir,"baseplate_all_clustered.fasta"),os.path.join(phage_init.fasta_dir,"major_tail_all_clustered.fasta"),
           os.path.join(phage_init.fasta_dir,"minor_tail_all_clustered.fasta"),os.path.join(phage_init.fasta_dir,"portal_all_clustered.fasta"),
           os.path.join(phage_init.fasta_dir,"tail_fiber_all_clustered.fasta"),os.path.join(phage_init.fasta_dir,"shaft_all_clustered.fasta"),
           os.path.join(phage_init.fasta_dir,"collar_all_clustered.fasta"),os.path.join(phage_init.fasta_dir,"HTJ_all_clustered.fasta"),
           os.path.join(phage_init.fasta_dir,"others_all_clustered.fasta"))


# %%
#all_fasta=(os.path.join(phage_init.fasta_dir,"minor_capsid_all_clustered.fasta"),os.path.join(phage_init.fasta_dir,"collar_all_clustered.fasta"))

# %%

def extract_all(fasta_list):
    d = {'seq_description': [], 'seq_id': [], "sec_code":[]}
    sec_code=0
    df = pd.DataFrame(data=d)
    total_fasta=0
    for file in fasta_list:
        for record in SeqIO.parse(file, "fasta"):
            total_fasta+=1
    
    AA=["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]
    SC=["1","2","3","4","5","6","7"]
    di_pep = [''.join(i) for i in itertools.product(AA, repeat = 2)]
    tri_pep = [''.join(i) for i in itertools.product(AA, repeat = 3)]
    di_sc = [''.join(i) for i in itertools.product(SC, repeat = 2)]
    tri_sc = [''.join(i) for i in itertools.product(SC, repeat = 3)]
    tetra_sc = [''.join(i) for i in itertools.product(SC, repeat = 4)]
    prot_class=0;
    myseq="AILMVNQSTGPCHKRDEFWY"
    trantab2=myseq.maketrans("AILMVNQSTGPCHKRDEFWY","11111222233455566777")
    arr = numpy.empty((total_fasta,11201), dtype=numpy.float)
    class_arr = numpy.empty((total_fasta), dtype=numpy.int)
    id_arr = numpy.empty((total_fasta), dtype=numpy.int)
    this_prot=0
    for file in fasta_list:
        print('####################' + file)
        for record in SeqIO.parse(file, "fasta"):
            ll=len(record.seq)
            seqq=record.seq.__str__().upper()
            seqqq=seqq.replace('X','A').replace('J','L').replace('*','A').replace('Z','E').replace('B','D')
            X = ProteinAnalysis(seqqq)
            tt= [X.isoelectric_point(), X.instability_index(),ll,X.aromaticity(),
                X.molar_extinction_coefficient()[0],X.molar_extinction_coefficient()[1],
                X.gravy(),X.molecular_weight()]
            tt_n = numpy.asarray(tt,dtype=numpy.float)
            myseq=seqq.translate(trantab2)
            
            di_pep_count=[seqq.count(i)/(ll-1) for i in di_pep]
            di_pep_count_n = numpy.asarray(di_pep_count,dtype=numpy.float)
            
            tri_pep_count=[seqq.count(i)/(ll-2) for i in tri_pep]
            tri_pep_count_n = numpy.asarray(tri_pep_count,dtype=numpy.float)
            
            di_sc_count=[myseq.count(i)/(ll-1) for i in di_sc]
            di_sc_count_n = numpy.asarray(di_sc_count,dtype=numpy.float)
    
            tri_sc_count=[myseq.count(i)/(ll-2) for i in tri_sc]
            tri_sc_count_n = numpy.asarray(tri_sc_count,dtype=numpy.float)
            
            tetra_sc_count=[myseq.count(i)/(ll-3) for i in tetra_sc]
            tetra_sc_count_n = numpy.asarray(tetra_sc_count,dtype=numpy.float)
    
            cat_n= numpy.concatenate((di_pep_count_n,tri_pep_count_n,di_sc_count_n,tri_sc_count_n,tetra_sc_count,tt_n))
            #print(cat_n.shape)
            cat_n = cat_n.reshape((1,cat_n.shape[0]))

            #arr = numpy.append(arr,cat_n , axis=0)
            #class_arr = numpy.append(class_arr,prot_class)
            #id_arr = numpy.append(id_arr,sec_code)
            arr[sec_code,:]=cat_n
            class_arr[sec_code]=prot_class
            id_arr[sec_code]=sec_code
            
            data_row=[record.description,record.id,int(sec_code)]
            df=df.append(pd.Series(data_row,index=df.columns),sort=False,ignore_index=True)
            sec_code+=1
            this_prot+=1
            if (this_prot%500==0):
                print("processing sequence # " + str(this_prot))
            
        prot_class+=1
        this_prot=0
    return (arr,class_arr,id_arr,df)
    


# %%
#0     - 400   (400)  di
#400   - 8400  (8000) tri
#8400  - 4449  (49)   di_sc
#4449  - 8792  (343)  tri_sc
#8792  - 11193 (2401) tetra_sc
#11193 - 11201 (8)    p

# %%
(arr,class_arr,id_arr,df)=extract_all(all_fasta)
print(arr.shape)

# %%

# %%
#df   class_arr,id_arr,df
import pickle
pickle.dump(arr, open( os.path.join(phage_init.data_dir,"raw_arr.p"), "wb" ),protocol=4 )
pickle.dump(class_arr, open( os.path.join(phage_init.data_dir,"raw_class_arr.p"), "wb" ),protocol=4 )
pickle.dump(id_arr, open( os.path.join(phage_init.data_dir,"raw_id_arr.p"), "wb" ),protocol=4 )
pickle.dump(df, open( os.path.join(phage_init.data_dir,"raw_df.p"), "wb" ),protocol=4 )

# %%
print(arr.shape)

# %%
arr_z=numpy.apply_along_axis(stats.zscore,0,arr)
mean_arr=numpy.apply_along_axis(numpy.mean,0,arr)
std_arr=numpy.apply_along_axis(numpy.std,0,arr)
pickle.dump(mean_arr,  open( os.path.join(phage_init.data_dir,"mean_final.p") , "wb" ), protocol=4 )
pickle.dump(std_arr,  open( os.path.join(phage_init.data_dir,"std_final.p") , "wb" ), protocol=4)


# %%
del arr
del mean_arr
del std_arr

# %%
nb_classes = 11
one_hot_targets = numpy.eye(nb_classes)[class_arr]

# %%
#cat_n = cat_n.reshape((1,cat_n.shape[0]))
print(id_arr.reshape((id_arr.shape[0],1)).shape)
print(arr_z.shape)
print(one_hot_targets.shape)
final = numpy.concatenate((id_arr.reshape((id_arr.shape[0],1)), arr_z, one_hot_targets), axis=1)

# %%
del arr_z

# %%
numpy.random.shuffle(final)

# %%
import pickle

pickle.dump(final, open( os.path.join(phage_init.data_dir,"zscore_all_final.p"), "wb" ),protocol=4 )



# %%
#import pickle
#final=pickle.load(open( os.path.join(phage_init.data_dir,"zscore_all_final.p"), "rb" ))
