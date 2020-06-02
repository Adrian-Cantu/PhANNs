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
import sys
sys.path.append("..")
import phage_init
import subprocess
from Bio import SeqIO

# %%
#records = list(SeqIO.parse(file, "fasta"))
#for record in SeqIO.parse(file, "fasta"):

# %%
fasta_list=[
'minor_capsid.fasta',
'tail_fiber.fasta',
'major_tail.fasta',
'portal.fasta',
'minor_tail.fasta',
'baseplate.fasta',
'collar.fasta',
'shaft.fasta',
'major_capsid.fasta',
'HTJ.fasta',
'others.fasta'
]

# %% [markdown]
# ## warning
# this step uses a modified version of cd-hit available here

# %%
for fasta in fasta_list:
    command = command='./cd-hit -i ' +  os.path.join('03_curated_fasta',fasta) + ' -o ' + os.path.join('05_1_cluster_split',fasta) + ' -M 0 -T 0 -c  0.4 -n 2'''
    print(command)
    subprocess.run(command,shell=True,check=True, text=True)

# %%
for fasta in fasta_list:
    records = list(SeqIO.parse(os.path.join('05_1_cluster_split',fasta), "fasta"))
    for f in range(11):
        SeqIO.write(records[f::11], os.path.join('05_1_cluster_split', str(f+1) + '_' + fasta) , "fasta")

# %%
for fasta in fasta_list:
    print(fasta)
    f = open(os.path.join('05_1_cluster_split',fasta+".clstr"), "r")
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
    record_dict = SeqIO.to_dict(SeqIO.parse(os.path.join('03_curated_fasta',fasta), "fasta"), key_function = lambda rec : rec.description)
    for num in range(11):
        print(num)
        record_list_to_print=[]
        for record in SeqIO.parse(os.path.join('05_1_cluster_split', str(num+1) + '_' + fasta), "fasta"):
            #print(record.description)
            record_list_to_print+=derep_dict[record.description]
        SeqIO.write([record_dict[xx] for xx in record_list_to_print], os.path.join('05_2_expanded_clusters', str(num+1) + '_tmp_' + fasta) , "fasta")

# %%
for fasta in fasta_list:
    for num in range(11):
        command = command='./cd-hit -i ' + os.path.join('05_2_expanded_clusters', str(num+1) + '_tmp_' + fasta) + ' -o ' + os.path.join('05_2_expanded_clusters', str(num+1) + '_' + fasta) + ' -M 0 -T 0 -c 1 -n 5'''
        print(command)
        subprocess.run(command,shell=True,check=True, text=True)
