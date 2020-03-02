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

# %%
records = list(SeqIO.parse(file, "fasta"))
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
file='hitmix.fasta'

# %%
for f in range(11):
    print(f+1)
    SeqIO.write(records[f::11], str(f+1) + '_' + file , "fasta")

# %%
f = open("hitmix.fasta.clstr", "r")
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
f.close()

# %%
derep_dict

# %%
record_dict = SeqIO.to_dict(SeqIO.parse("minor_capsid_all_clustered.fasta", "fasta"), key_function = lambda rec : rec.description)

# %%
record_dict['GAV38282.1 phage minor capsid protein 2 [Streptomyces acidiscabies]']

# %%
derep_dict[records[4].description]

# %%

ll=[record_dict[xx] for xx in derep_dict['WP_031017713.1 phage minor capsid protein 2 [Streptomyces sp. NRRL WC-3795]']]

# %%
SeqIO.write(ll, "test.fasta" , "fasta")

# %%
record_list_to_print=[]
for record in SeqIO.parse('6_hitmix.fasta', "fasta"):
    record_list_to_print+=derep_dict[record.description]
SeqIO.write([record_dict[xx] for xx in record_list_to_print], "expanded/test.fasta" , "fasta")

# %%
