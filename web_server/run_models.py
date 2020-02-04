"""
Run one of the models against a fasta file and generate tab separated output.
This command is meant to be run on the command line (e.g. as part of a pipeline), if you would
like a graphical interface, please see the README.md file to view the results as a web server
"""

import os
import sys
import argparse

from roblib import bcolors


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

import pandas as pd
class ann_result:
	infile=''
	result_table=''

	def __init__(self, filename):
		self.infile=filename

	def prot_check(self, sequence):
		return set(sequence.upper()).issubset("ACDEFGHIJKLMNPQRSTVWY*")

	def print_table(self):


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', help='fasta file of protein sequences')
    parser.add_argument('-v', help='print the current version and exit', action='store_true')
    args = parser.parse_args()

    if args.v:
        with open(os.path.join("..", 'VERSION')) as version_file:
            version = version_file.read().strip()
            print("{} version {}".format(sys.argv[0], version))
            sys.exit(0)

