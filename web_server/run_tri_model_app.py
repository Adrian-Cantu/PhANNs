
# coding: utf-8

# In[1]:

import sys
import numpy
import itertools
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy import stats
from Bio.Alphabet import generic_dna, generic_protein
import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Dropout
from keras.models import load_model
from flask import Markup
import ntpath
import os
import pandas as pd
from rq import get_current_job
job = get_current_job()
print(job.get_id())
class ann_result:
    infile=''
    html_table=''
#    job=''

    def __init__(self, filename):
        self.infile=filename
#        self.job = get_current_job()

    def prot_check(self, sequence):
        return set(sequence.upper()).issubset("ACDEFGHIJKLMNPQRSTVWY*")

    def print_table(self):
    #from itertools import permutations
        AA=["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]
        tri_pep = [''.join(i) for i in itertools.product(AA, repeat = 3)]
        total_fasta=0
        sec_code=0
        for record in SeqIO.parse(self.infile, "fasta"):
            if self.prot_check(str(record.seq)):
                total_fasta+=1
        job.meta['total']=total_fasta
        arr = numpy.empty((total_fasta,8008), dtype=numpy.float)
        names = numpy.empty((total_fasta,1),  dtype=object)
        names_dic=dict()
        record_current=0
        print(job.get_id())
        for record in SeqIO.parse(self.infile, "fasta"):
            record_current += 1
            job.meta['current']=record_current
            job.save_meta()
            ll=len(record.seq)
            seq_name=''
            if not self.prot_check(str(record.seq)):
                print("Warning: " + record.id + " is not a valid protein sequence")
                continue
            if record.id in names_dic:
                seq_name= record.id + '_' + str(names_dic[record.id])
                names_dic[record.id]=names_dic[record.id]+1
            else:
                seq_name= record.id
                #seq_name=record_current
                names_dic[record.id]=1
            #seq_name=record_current
            #print(str(record.seq))
            X = ProteinAnalysis(record.seq.__str__().upper().replace('X','A').replace('J','L').replace('*',''))
            #print(record.id)
            #print(record.seq.__str__())
            tt= [X.isoelectric_point(), X.instability_index(),ll,X.aromaticity(),
                 X.molar_extinction_coefficient()[0],X.molar_extinction_coefficient()[1],
                 X.gravy(),X.molecular_weight()]
            tt_n = numpy.asarray(tt,dtype=numpy.float)

            tri_pep_count=[record.seq.count(i)/(ll-2) for i in tri_pep]
            tri_pep_count_n = numpy.asarray(tri_pep_count,dtype=numpy.float)
    
            cat_n= numpy.append(tri_pep_count_n,tt_n)
            cat_n = cat_n.reshape((1,cat_n.shape[0]))


            arr[sec_code,:]=cat_n
            names[sec_code,0]=seq_name
            sec_code += 1

        job.meta['running']=1
        job.save_meta()
        
        mean_arr_tmp=pickle.load(open( "tri_p_model/mean_final.p", "rb" ) )
        std_arr_tmp=pickle.load(open( "tri_p_model/std_final.p", "rb" ) )
        mean_arr=numpy.concatenate((mean_arr_tmp[400:8400],mean_arr_tmp[8792:]),axis=0)
        std_arr=numpy.concatenate((std_arr_tmp[400:8400],std_arr_tmp[8792:]),axis=0)


        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                arr[i,j]=(arr[i,j]-mean_arr[j])/std_arr[j]
                #mean_arr[j-1]=1

        model = load_model('tri_p_model/tri_p.h5')


        arr_pred=model.predict(arr)

        col_names=["Major capsid","Minor capsid","Baseplate",
                   "Major tail","Minor tail","Portal",
                    "Tail fiber","Tail shaft","Collar",
                       "HTJ","Other"]

        table1=pd.DataFrame(data=arr_pred,
                    index=names[:,0],
                    columns=col_names,
                    dtype=numpy.float64
                    )
        print(table1.dtypes)
        pd.options.display.float_format = '{:.2f}'.format
        table1.astype(float).to_csv("csv_saves/"+ os.path.splitext(ntpath.basename(self.infile))[0] + '.csv',float_format = "%.2f")
        html_style=table1.style.set_uuid("table_1").set_table_styles([{'selector':'table', 'props': [('border', '1px solid black'),('border-collapse','collapse'),('width','100%')]},{'selector':'th', 'props': [('border', '1px solid black'),('padding', '15px')]},{'selector':'td', 'props': [('border', '1px solid black'),('padding', '15px')]}]).format("{:.2f}").highlight_max(axis=1)
        self.html_table=html_style.render()
        table_code_raw= Markup(self.html_table)
        pickle.dump(table_code_raw,open('saves/' + ntpath.basename(self.infile),"wb"))
        
        
        arr_class=arr_pred.argmax(axis=1)
        sec_code=0
        major_capsid_sequences = []
        minor_capsid_sequences = []
        baseplate_sequences = []
        major_tail_sequences = []
        minor_tail_sequences = []
        portal_sequences = []
        tail_fiber_sequences = []
        tail_shaft_sequences = []
        collar_sequences = []
        htj_sequences = []
        other_sequences = []
        for record in SeqIO.parse(self.infile, "fasta"):
            seq_name=''
            if not self.prot_check(str(record.seq)):
                continue
            if arr_class[sec_code]==0:
                major_capsid_sequences.append(record)
            elif arr_class[sec_code]==1:
                minor_capsid_sequences.append(record)
            elif arr_class[sec_code]==2:
                baseplate_sequences.append(record)
            elif arr_class[sec_code]==3:
                major_tail_sequences.append(record)
            elif arr_class[sec_code]==4:
                minor_tail_sequences.append(record)
            elif arr_class[sec_code]==5:
                portal_sequences.append(record)
            elif arr_class[sec_code]==6:
                tail_fiber_sequences.append(record)
            elif arr_class[sec_code]==7:
                tail_shaft_sequences.append(record)
            elif arr_class[sec_code]==8:
                collar_sequences.append(record)
            elif arr_class[sec_code]==9:
                htj_sequences.append(record)
            elif arr_class[sec_code]==10:
                other_sequences.append(record)

            sec_code += 1
        SeqIO.write(major_capsid_sequences, "csv_saves/major_capsid_"+ ntpath.basename(self.infile) , "fasta")
        SeqIO.write(minor_capsid_sequences, "csv_saves/minor_capsid" + "_"+ ntpath.basename(self.infile) , "fasta")
        SeqIO.write(baseplate_sequences, "csv_saves/baseplate" + "_"+ ntpath.basename(self.infile) , "fasta")
        SeqIO.write(major_tail_sequences, "csv_saves/major_tail" + "_"+ ntpath.basename(self.infile) , "fasta")
        SeqIO.write(minor_tail_sequences, "csv_saves/minor_tail" + "_"+ ntpath.basename(self.infile) , "fasta")
        SeqIO.write(portal_sequences, "csv_saves/portal" + "_"+ ntpath.basename(self.infile) , "fasta")
        SeqIO.write(tail_fiber_sequences, "csv_saves/tail_fiber" + "_"+ ntpath.basename(self.infile) , "fasta")
        SeqIO.write(tail_shaft_sequences, "csv_saves/tail_shaft" + "_"+ ntpath.basename(self.infile) , "fasta")
        SeqIO.write(collar_sequences, "csv_saves/collar" + "_"+ ntpath.basename(self.infile) , "fasta")
        SeqIO.write(htj_sequences, "csv_saves/htj" + "_"+ ntpath.basename(self.infile) , "fasta")
        SeqIO.write(other_sequences, "csv_saves/other" + "_"+ ntpath.basename(self.infile) , "fasta")


def entrypoint(filename):
    #open_file = 'A45_phage_orfs.txt'
    #kk=ann_result(open_file)
    kk=ann_result(filename)
    table=kk.print_table()
#    print(table)



if __name__ == "__main__":
    if len(sys.argv) > 1:
        open_file = sys.argv[1]
    else:
        open_file = 'A45_phage_orfs.txt'
    kk=ann_result(open_file)
    table=kk.print_table()
    print(table)



