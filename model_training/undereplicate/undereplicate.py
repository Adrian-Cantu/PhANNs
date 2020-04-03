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
from Bio import SeqIO
import os

# %%
#records = list(SeqIO.parse(file, "fasta"))
#for record in SeqIO.parse(file, "fasta"):

# %%
files=['basplate.fasta',
'collar.fasta',
'HTJ.fasta',
'major_capsid.fasta',
'major_tail.fasta',
'minor_capsid.fasta',
'minor_tail.fasta',
'other.fasta',
'portal.fasta',
'shaft.fasta',
'tail_fiber.fasta']

# %%
#file='hitmix.fasta'

# %%
for fasta in files:
    records = list(SeqIO.parse(os.path.join('dereplicate30',fasta), "fasta"))
    for f in range(11):
        SeqIO.write(records[f::11], os.path.join('parts', str(f+1) + '_' + fasta) , "fasta")

# %%
for fasta in files:
    print(fasta)
    f = open(os.path.join('dereplicate30',fasta+".clstr"), "r")
    pp=True;
    derep_dict=dict()
    for line in f:
        line = line.rstrip()    # remove ALL whitespaces on the right side, including '\n'
    # do something with line
        if line[0] == '>':
            if pp:
                seq_list=[]
                pp=False
                continue
            derep_dict[main_id]=seq_list
            seq_list=[]
            continue
        if line[0] == '*':
            line=line.lstrip('*')
            main_id=line
        seq_list.append(line)
    #print(line)
    derep_dict[main_id]=seq_list
    f.close()
    record_dict = SeqIO.to_dict(SeqIO.parse(os.path.join('clean',fasta), "fasta"), key_function = lambda rec : rec.description)
    for num in range(11):
        print(num)
        record_list_to_print=[]
        for record in SeqIO.parse(os.path.join('parts', str(num+1) + '_' + fasta), "fasta"):
            #print(record.description)
            record_list_to_print+=derep_dict[record.description]
        SeqIO.write([record_dict[xx] for xx in record_list_to_print], os.path.join('expanded', str(num+1) + '_' + fasta) , "fasta")

# %%
