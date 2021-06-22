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
# ## generate_curating_lists
# This script generates the list of terms ( .list) for each fasta file. Each line in a list file contains a fasta fasta header and a number that indicates how many sequences have that header. Manual curation involves reading the list file and adding ‘+’ to start of each line with relevant terms. The sequences with those headers will be added to the database in the next step. Note that ‘+’ must be the first character in the line, any other character is ignore. WARNING, running this script will overwrite the default list files included with the repository.

# %%
import os
import sys
sys.path.append("..")
import phage_init
import subprocess

# %%
fasta_list=[
'01_fasta/others.fasta',
#'01_fasta/minor_capsid.fasta',
'01_fasta/tail_fiber.fasta',
'01_fasta/major_tail.fasta',
'01_fasta/portal.fasta',
'01_fasta/minor_tail.fasta',
'01_fasta/baseplate.fasta',
'01_fasta/collar.fasta',
'01_fasta/shaft.fasta',
#'01_fasta/major_capsid.fasta',
'01_fasta/capsid.fasta',
'01_fasta/HTJ.fasta'
]

# %%
for fasta in fasta_list:
    command = 'grep ">" ' + fasta + ' | cut -f1 -d' + "'['" + ' | cut -f1 --complement -d" " | sort | uniq -c | sort -n -r > ' +  os.path.splitext(fasta)[0] + '.list'
    print(command)
    subprocess.run(command,shell=True,check=True, text=True)
