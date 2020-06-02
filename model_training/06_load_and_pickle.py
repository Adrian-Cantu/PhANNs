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
import os
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
import pickle

# %%
files=['major_capsid.fasta',
       'minor_capsid.fasta',
       'baseplate.fasta',
       'major_tail.fasta',
       'minor_tail.fasta',
       'portal.fasta',
       'tail_fiber.fasta',
       'shaft.fasta',
       'collar.fasta',
       'HTJ.fasta',
       'others.fasta'
]

# %%
class_label=['major_capsid',
       'minor_capsid',
       'baseplate',
       'major_tail',
       'minor_tail',
       'portal',
       'tail_fiber',
       'shaft',
       'collar',
       'HTJ',
       'others'
]


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
    group_arr = numpy.empty((total_fasta), dtype=numpy.int)
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
            class_arr[sec_code]=prot_class%11
            group_arr[sec_code]=prot_class//11
            id_arr[sec_code]=sec_code
            
            data_row=[record.description,record.id,int(sec_code)]
    #        df=df.append(pd.Series(data_row,index=df.columns),sort=False,ignore_index=True)
            sec_code+=1
            this_prot+=1
            if (this_prot%500==0):
                print("processing sequence # " + str(this_prot),end="\r")
            
        prot_class+=1
        this_prot=0
    return (arr,class_arr,group_arr,id_arr,df)



# %%
ll=[]
for num in range(11):
    ll+=[os.path.join('05_2_expanded_clusters',str(num+1)+'_'+xx) for xx in files]
    #(arr,class_arr,id_arr,df)=extract_all(fasta_list)
    
    #print(ll)

# %%
(arr,class_arr,group_arr,id_arr,df)=extract_all(ll)

# %%
sum((class_arr==0) & (group_arr==0))

# %%
arr_z=numpy.apply_along_axis(stats.zscore,0,arr)
mean_arr=numpy.apply_along_axis(numpy.mean,0,arr)
std_arr=numpy.apply_along_axis(numpy.std,0,arr)

# %%
pickle.dump(mean_arr,  open( os.path.join('data',"mean_final.p") , "wb" ), protocol=4 )
pickle.dump(std_arr,  open( os.path.join('data',"std_final.p") , "wb" ), protocol=4)

pickle.dump(class_arr,  open( os.path.join('data',"class_arr.p") , "wb" ), protocol=4)
pickle.dump(group_arr,  open( os.path.join('data',"group_arr.p") , "wb" ), protocol=4)

# %%
del arr

# %%
pickle.dump(arr_z,  open( os.path.join('data',"all_data.p") , "wb" ), protocol=4)

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
all_ranges=[di_sc_range,di_sc_p_range,tri_sc_range,tri_sc_p_range,tetra_sc_range,tetra_sc_p_range,di_range,di_p_range,tri_range,tri_p_range,tetra_sc_tri_p_range,all_range]
all_models=['di_sc','di_sc_p','tri_sc','tri_sc_p','tetra_sc','tetra_sc_p','di','di_p','tri','tri_p','tetra_sc_tri_p','all']
model_ranges=dict(zip(all_models,all_ranges))

# %%
for group_number in range(11):
    for model_name in all_models:
        print(os.path.join('data',str(group_number+1)+'_'+model_name+".p"))
        meh=arr_z[(group_arr==group_number),]
        mehh=meh[:,model_ranges[model_name]]
        pickle.dump(mehh,  open( os.path.join('data',str(group_number+1)+'_'+model_name+".p") , "wb" ), protocol=4 )
        del mehh
        del meh

# %%
#for group in range(2):
#    for p_class in range(11):
#        pickle.dump(arr_z[(class_arr==p_class) & (group_arr==group),], open( os.path.join('data',str(group+1)+'_'+class_label[p_class]+".p"), "wb" ),protocol=4 )
