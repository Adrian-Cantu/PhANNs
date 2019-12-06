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
# ## get\_size\_distribution
# this script gets the size distribution of sequences
#
#

# %%
import os
import sys
sys.path.append("..")
import phage_init

# %%
import numpy
#import itertools
#from Bio.Seq import Seq
#from Bio.Alphabet import IUPAC
from Bio import SeqIO
#from Bio.SeqUtils.ProtParam import ProteinAnalysis
#from scipy import stats
#from Bio.Alphabet import IUPAC
#from itertools import permutations
import pandas as pd
#d = {'lenght': [], 'class': [],'sec_id':[],'sec_code':[]}
#df = pd.DataFrame(data=d)

# %%
all_fasta=(os.path.join(phage_init.fasta_dir,"major_capsid_all_clustered.fasta"),os.path.join(phage_init.fasta_dir,"minor_capsid_all_clustered.fasta"),
           os.path.join(phage_init.fasta_dir,"baseplate_all_clustered.fasta"),os.path.join(phage_init.fasta_dir,"major_tail_all_clustered.fasta"),
           os.path.join(phage_init.fasta_dir,"minor_tail_all_clustered.fasta"),os.path.join(phage_init.fasta_dir,"portal_all_clustered.fasta"),
           os.path.join(phage_init.fasta_dir,"tail_fiber_all_clustered.fasta"),os.path.join(phage_init.fasta_dir,"shaft_all_clustered.fasta"),
           os.path.join(phage_init.fasta_dir,"collar_all_clustered.fasta"),os.path.join(phage_init.fasta_dir,"HTJ_all_clustered.fasta"),
           os.path.join(phage_init.fasta_dir,"others_all_clustered.fasta"))


# %%

def get_size_df(fasta_list):
    #d = {'seq_description': [], 'seq_id': [], "sec_code":[]}
    d = {'length': [], 'class': [],'sec_id':[],'sec_code':[]}
    sec_code=0
    df = pd.DataFrame(data=d)
    prot_class=0
    this_prot=0
    total_fasta=0
    for file in fasta_list:
        for record in SeqIO.parse(file, "fasta"):
            total_fasta+=1
    lenghth_list = numpy.empty((total_fasta), dtype=numpy.int)
    class_list = numpy.empty((total_fasta), dtype=numpy.int)
    sec_code_list = numpy.empty((total_fasta), dtype=numpy.int)
    sec_id_list=numpy.empty((total_fasta),dtype='U30')
            
    
    for file in fasta_list:
        print('####################' + file)
        for record in SeqIO.parse(file, "fasta"):
            ll=len(record.seq)
            #data_row=[ll,prot_class,record.id,sec_code]
            #df=df.append(pd.Series(data_row,index=df.columns),sort=False,ignore_index=True)
            lenghth_list[sec_code]=ll
            class_list[sec_code]=prot_class
            sec_code_list[sec_code]=sec_code
            sec_id_list[sec_code]=record.id
            sec_code+=1
        prot_class+=1
        this_prot=0
    df['length']=lenghth_list
    df['class']=class_list
    df['sec_id']=sec_id_list
    df['sec_code']=sec_code_list
    return df
    


# %%
#0     - 400   (400)  di
#400   - 8400  (8000) tri
#8400  - 4449  (49)   di_sc
#4449  - 8792  (343)  tri_sc
#8792  - 11193 (2401) tetra_sc
#11193 - 11201 (8)    p

# %%
df=get_size_df(all_fasta)

# %%
numpy.percentile(df['length'],[80,85,90,95,99])


# %%
import matplotlib.pyplot as plt
plt.hist(df['length'],bins=100)
plt.show()
