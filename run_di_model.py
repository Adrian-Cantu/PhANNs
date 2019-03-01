
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
from Bio.Alphabet import IUPAC

import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Dropout
from keras.models import load_model

import pandas as pd
class ann_result:
	infile=''
	html_table=''

	def __init__(self, filename):
		self.infile=filename


	def print_table(self):
	#from itertools import permutations
		AA=["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]
		dipep = [''.join(i) for i in itertools.product(AA, repeat = 2)]
		arr = numpy.empty((0,407), dtype=numpy.float)
		names = numpy.empty((0,1),  dtype=object)
		names_dic=dict()
		for record in SeqIO.parse(self.infile, "fasta"):
			ll=len(record.seq)
			seq_name=''
			if record.id in names_dic:
				seq_name= record.id + '_' + str(names_dic[record.id])
				names_dic[record.id]=names_dic[record.id]+1
			else:
				seq_name= record.id
				names_dic[record.id]=1
			#print(str(record.seq))
			X = ProteinAnalysis(record.seq.__str__().replace('X','A').replace('J','L').replace('*',''))
			tt= [X.isoelectric_point(), X.instability_index(),ll,X.aromaticity(),
     			X.molar_extinction_coefficient()[0],X.molar_extinction_coefficient()[1],
     			X.gravy()]
			tt_n = numpy.asarray(tt,dtype=numpy.float)

			dipep_count=[record.seq.count(i)/ll for i in dipep]
			dipep_count_n = numpy.asarray(dipep_count,dtype=numpy.float)
    
			cat_n= numpy.append(dipep_count_n,tt_n)
			cat_n = cat_n.reshape((1,cat_n.shape[0]))

			arr = numpy.append(arr,cat_n , axis=0)
			names = numpy.append(names,seq_name)

		mean_arr=pickle.load(open( "dipep_new_mean.p", "rb" ) )
		std_arr=pickle.load(open( "dipep_new_std.p", "rb" ) )

		for i in range(arr.shape[0]):
			for j in range(arr.shape[1]):
				arr[i,j]=(arr[i,j]-mean_arr[j])/std_arr[j]

		model = load_model('di_new_model.h5')


		arr_pred=model.predict(arr)

		col_names=["Major capsid","Minor capsid","Baseplate",
        	   	"Major tail","Minor tail","Portal",
         	   	"Tail fiber","Tail shaft","Collar",
                   	"HTJ"]
		table1=pd.DataFrame(data=arr_pred,
                	index=names,
                	columns=col_names
                	)

		html_style=table1.style.set_table_styles([{'selector':'table', 'props': [('border', '1px solid black'),('border-collapse','collapse'),('width','100%')]},{'selector':'th', 'props': [('border', '1px solid black'),('padding', '15px')]},{'selector':'td', 'props': [('border', '1px solid black'),('padding', '15px')]}]).format("{:.2f}").highlight_max(axis=1)
		self.html_table=html_style.render()
		return(self.html_table)

if __name__ == "__main__":
	if len(sys.argv) > 1:
		open_file = sys.argv[1]
	else:
		open_file = 'A45_phage_orfs.txt'
	kk=ann_result(open_file)
	table=kk.print_table()
	print(table)



