# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.1
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

# %%
import ann_data

# %%
import pickle
from Bio import SeqIO

# %%
test_id=ann_data.get_test_id()
id_df=pickle.load(open( os.path.join(phage_init.data_dir,"raw_df.p"), "rb" ))

# %%
(test_X,test_Y)=ann_data.get_formated_test("di")
test_Y_index = test_Y.argmax(axis=1)

# %%
major_tail_id=test_id[test_Y_index == 3]
major_tail_id.shape

# %%
major_tail_df=id_df.loc[id_df['sec_code'].isin(major_tail_id)]

# %%
major_tail_seq_id=major_tail_df['seq_id']

# %%
major_tail_list=major_tail_seq_id.tolist()

# %%
write_records = [] 
for record in SeqIO.parse(os.path.join(phage_init.fasta_dir,"major_tail_all_clustered.fasta"), "fasta"):
    if record.id in major_tail_list:
        print(record.id)
        write_records.append(record)
SeqIO.write(write_records, 'major_tail.fasta', "fasta")
