import numpy
import itertools
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy import stats
from Bio.Alphabet import IUPAC
import tensorflow as tf
import pandas as pd
import os
import ntpath
from flask import Markup
import pickle

import ann_config

class ann_result:

    infile=''
    html_table=''
    
    def __init__(self, filename):
        self.infile=filename
    
    def prot_check(self, sequence):
        return set(sequence.upper()).issubset("ACDEFGHIJKLMNPQRSTVWY*")

    def extract(self):
        AA=["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]
        SC=["1","2","3","4","5","6","7"]
        tri_pep = [''.join(i) for i in itertools.product(AA, repeat = 3)]
        myseq="AILMVNQSTGPCHKRDEFWY"
        trantab2=myseq.maketrans("AILMVNQSTGPCHKRDEFWY","11111222233455566777")
        tetra_sc = [''.join(i) for i in itertools.product(SC, repeat = 4)]
        total_fasta=0
        sec_code=0
        record_current=0
        for record in SeqIO.parse(self.infile, "fasta"):
            if self.prot_check(str(record.seq)):
                total_fasta+=1
        arr = numpy.empty((total_fasta,10409), dtype=numpy.float)
        names = numpy.empty((total_fasta,1),  dtype=object)
        names_dic=dict()
        for record in SeqIO.parse(self.infile, "fasta"):
            record_current += 1
            #job.meta['current']=record_current
            #job.save_meta()
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
                names_dic[record.id]=1
            seqq=record.seq.__str__().upper()
            seqqq=seqq.replace('X','A').replace('J','L').replace('*','A').replace('Z','E').replace('B','D')
           # X = ProteinAnalysis(record.seq.__str__().upper().replace('X','A').replace('J','L').replace('*',''))
            X = ProteinAnalysis(seqqq)
            myseq=seqq.translate(trantab2)
            tt= [X.isoelectric_point(), X.instability_index(),ll,X.aromaticity(),
                 X.molar_extinction_coefficient()[0],X.molar_extinction_coefficient()[1],
                 X.gravy(),X.molecular_weight()]
            tt_n = numpy.asarray(tt,dtype=numpy.float)

            tri_pep_count=[seqq.count(i)/(ll-2) for i in tri_pep]
            tri_pep_count_n = numpy.asarray(tri_pep_count,dtype=numpy.float)
            
            tetra_sc_count=[myseq.count(i)/(ll-3) for i in tetra_sc]
            tetra_sc_count_n = numpy.asarray(tetra_sc_count,dtype=numpy.float)
    
            cat_n= numpy.concatenate((tetra_sc_count_n,tri_pep_count_n,tt_n))
            cat_n = cat_n.reshape((1,cat_n.shape[0]))


            arr[sec_code,:]=cat_n
            names[sec_code,0]=seq_name
            sec_code += 1
        return (names,arr)

    def extract_n(self):
        (names,arr)=self.extract()
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if ann_config.std_arr[j]==0:
                    pass
                else:
                    arr[i,j]=(arr[i,j]-ann_config.mean_arr[j])/ann_config.std_arr[j]
        return(names,arr)
    
    def predict_score(self):
        (names,arr)=self.extract_n()
        yhats = [model.predict(arr) for model in ann_config.models]
        yhats_v=numpy.array(yhats)
        predicted_Y=numpy.sum(yhats_v, axis=0)
        col_names=["Major capsid","Minor capsid","Baseplate",
            "Major tail","Minor tail","Portal",
            "Tail fiber","Tail shaft","Collar",
            "HTJ","Other"]

        table1=pd.DataFrame(data=arr_pred,
            index=names[:,0],
            columns=col_names,
            dtype=numpy.float64
            )
        pd.options.display.float_format = '{:.2f}'.format
        table1.astype(float).to_csv("csv_saves/"+ os.path.splitext(ntpath.basename(self.infile))[0] + '.csv',float_format = "%.2f")
        html_style=table1.style.set_uuid("table_1").set_table_styles([{'selector':'table', 'props': [('border', '1px solid black'),('border-collapse','collapse'),('width','100%')]},{'selector':'th', 'props': [('border', '1px solid black'),('padding', '15px')]},{'selector':'td', 'props': [('border', '1px solid black'),('padding', '15px')]}]).format("{:.2f}").highlight_max(axis=1)
        self.html_table=html_style.render()
        table_code_raw= Markup(self.html_table)
        pickle.dump(table_code_raw,open('saves/' + ntpath.basename(self.infile),"wb"))
        return (names,predicted_Y)
    
    def predict_score_test(self):
        #global ann_config.graph
        with ann_config.graph.as_default():
            (names,arr)=self.extract_n()
            yhats_v=ann_config.models.predict(arr)
            predicted_Y=numpy.sum(yhats_v, axis=0)
            col_names=["Major capsid","Minor capsid","Baseplate",
            "Major tail","Minor tail","Portal",
            "Tail fiber","Tail shaft","Collar",
            "HTJ","Other"]

            table1=pd.DataFrame(data=yhats_v,
            index=names[:,0],
            columns=col_names,
            dtype=numpy.float64
            )
            pd.options.display.float_format = '{:.2f}'.format
            table1.astype(float).to_csv("csv_saves/"+ os.path.splitext(ntpath.basename(self.infile))[0] + '.csv',float_format = "%.2f")
            html_style=table1.style.set_uuid("table_1").set_table_styles([{'selector':'table', 'props': [('border', '1px solid black'),('border-collapse','collapse'),('width','100%')]},{'selector':'th', 'props': [('border', '1px solid black'),('padding', '15px')]},{'selector':'td', 'props': [('border', '1px solid black'),('padding', '15px')]}]).format("{:.2f}").highlight_max(axis=1)
            self.html_table=html_style.render()
            table_code_raw= Markup(self.html_table)
            pickle.dump(table_code_raw,open('saves/' + ntpath.basename(self.infile),"wb"))
            return (names,predicted_Y)
    
    


