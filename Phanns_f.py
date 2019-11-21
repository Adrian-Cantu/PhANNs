import numpy
import itertools
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy import stats
from Bio.Alphabet import IUPAC

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


