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
from flask_socketio import emit, SocketIO

import ann_config

class ann_result:

    infile=''
    html_table=''
    g_total_fasta=''
    g_all_fasta=''
    g_sid=''
    g_socketio = SocketIO()
    g_is_socket = 0
    
    def __init__(self, filename,sid_n=8888,socketio=SocketIO()):
        self.infile=filename
        self.g_sid=sid_n
        self.g_socketio=socketio
        total_fasta=0
        all_fasta=0
        for record in SeqIO.parse(self.infile, "fasta"):
            all_fasta+=1
            if self.prot_check(str(record.seq)):
                total_fasta+=1
        self.g_total_fasta=total_fasta
        self.g_all_fasta=all_fasta
        try:
            self.g_socketio.emit('set bar', {'data': '0'},room=self.g_sid)
        except AttributeError:
            pass
        else:
            self.g_is_socket = 1

#    def __init__(self, filename):
#        self.infile=filename
#        self.g_is_socket = 0
#        total_fasta=0
#        all_fasta=0
#        for record in SeqIO.parse(self.infile, "fasta"):
#            all_fasta+=1
#            if self.prot_check(str(record.seq)):
#                total_fasta+=1
#        self.g_total_fasta=total_fasta
#        self.g_all_fasta=all_fasta
    
    def prot_check(self, sequence):
        return set(sequence.upper()).issubset("ACDEFGHIJKLMNPQRSTVWY*")

    def extract(self):
        AA=["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]
        SC=["1","2","3","4","5","6","7"]
        tri_pep = [''.join(i) for i in itertools.product(AA, repeat = 3)]
        myseq="AILMVNQSTGPCHKRDEFWY"
        trantab2=myseq.maketrans("AILMVNQSTGPCHKRDEFWY","11111222233455566777")
        tetra_sc = [''.join(i) for i in itertools.product(SC, repeat = 4)]
        total_fasta=self.g_total_fasta
        sec_code=0
        record_current=0
        arr = numpy.empty((total_fasta,10409), dtype=numpy.float)
        names = numpy.empty((total_fasta,1),  dtype=object)
        names_dic=dict()
        for record in SeqIO.parse(self.infile, "fasta"):
            data=(record_current/total_fasta) * 100
            if (self.g_is_socket==1):
                self.g_socketio.emit('set bar', {'data': data},room=self.g_sid)
            else:
                print('extracting features of seq ' + str(record_current+1) + ' of ' + str(total_fasta),end='\r')
            #yield "event: update\ndata:" + str(data) + "\n\n"
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
        if (self.g_is_socket==1):
            self.g_socketio.emit('set bar', {'data': 100},room=self.g_sid)
            self.g_socketio.emit('done features',1,room=self.g_sid)
        return (names,arr)

    def extract_n(self):
#        if not self.g_names:
        (names,arr)=self.extract()
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if ann_config.std_arr[j]==0:
                    pass
                else:
                    arr[i,j]=(arr[i,j]-ann_config.mean_arr[j])/ann_config.std_arr[j]
        return (names,arr)
    
    
    def predict_score(self):
        #global ann_config.graph
        #with ann_config.graph.as_default():
            (names,arr)=self.extract_n()
            yhats = [model.predict(arr) for model in ann_config.models]
            yhats_v=numpy.array(yhats)
            predicted_Y=numpy.sum(yhats_v, axis=0)
            #predicted_Y=numpy.sum(yhats_v, axis=0)
            col_names=["Major capsid","Minor capsid","Baseplate",
            "Major tail","Minor tail","Portal",
            "Tail fiber","Tail shaft","Collar",
            "HTJ","Other"]

            table1=pd.DataFrame(data=predicted_Y,
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
            self.generate_fasta(predicted_Y)
            return (names,predicted_Y)

    
    def predict_score_test(self):
        #global ann_config.graph
        #with ann_config.graph.as_default():
            (names,arr)=self.extract_n()
            yhats_v=ann_config.models.predict(arr)
            #predicted_Y=numpy.sum(yhats_v, axis=0)
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
            self.generate_fasta(yhats_v)
            return (names,yhats_v)
        
    def predict_score_single_run(self):
            (names,arr)=self.extract_n()
            yhats = [model.predict(arr) for model in ann_config.models]
            yhats_v=numpy.array(yhats)
            predicted_Y=numpy.sum(yhats_v, axis=0)
            #print(arr)
            #predicted_Y=ann_config.models.predict(arr, verbose=1)
            col_names=["Major capsid","Minor capsid","Baseplate",
            "Major tail","Minor tail","Portal",
            "Tail fiber","Tail shaft","Collar",
            "HTJ","Other"]

            table1=pd.DataFrame(data=predicted_Y,
            index=names[:,0],
            columns=col_names,
            dtype=numpy.float64
            )
            pd.options.display.float_format = '{:.2f}'.format
            table1.astype(float).to_csv(os.path.splitext(ntpath.basename(self.infile))[0] + '.csv',float_format = "%.4f")
            html_style=table1.style.set_uuid("table_1").set_table_styles([{'selector':'table', 'props': [('border', '1px solid black'),('border-collapse','collapse'),('width','100%')]},{'selector':'th', 'props': [('border', '1px solid black'),('padding', '15px')]},{'selector':'td', 'props': [('border', '1px solid black'),('padding', '15px')]}]).format("{:.2f}").highlight_max(axis=1)
            self.html_table=html_style.render()
            #table_code_raw= Markup(self.html_table)
            #pickle.dump(table_code_raw,open('saves/' + ntpath.basename(self.infile),"wb"))
            #self.generate_fasta(predicted_Y)
            return (names,table1)

    def generate_fasta(self, predicted_Y):
        arr_class=predicted_Y.argmax(axis=1)
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
    


