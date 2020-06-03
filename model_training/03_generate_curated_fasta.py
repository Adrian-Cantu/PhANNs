# -*- coding: utf-8 -*-
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

# %% [markdown]
# ## generate_curated_fasta
# This script generates curated fastas from the base fasta and the list file. The "generate_curating_list" script has more details on the manual curation.

# %%
import os
import sys
sys.path.append("..")
import phage_init
import subprocess

# %%
fasta_list=[
'01_fasta/minor_capsid.fasta',
'01_fasta/tail_fiber.fasta',
'01_fasta/major_tail.fasta',
'01_fasta/portal.fasta',
'01_fasta/minor_tail.fasta',
'01_fasta/baseplate.fasta',
'01_fasta/collar.fasta',
'01_fasta/shaft.fasta',
'01_fasta/major_capsid.fasta',
'01_fasta/HTJ.fasta'
]

# %%
for fasta in fasta_list:
    command = 'perl list2fasta.pl ' +fasta+ ' > 03_curated_fasta/' + os.path.basename(fasta)
    print(command)
    subprocess.run(command,shell=True,check=True, text=True)

# %%
command = 'perl list2fasta.pl 01_fasta/others.fasta > 03_curated_fasta/others_tmp.fasta'
print(command)
__=subprocess.run(command,shell=True,check=True, text=True)
