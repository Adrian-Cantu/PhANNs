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
# ## clean_others
# This script remove any sequence in the “others” database that clusters at 60% with any sequence in the structural database. 

# %%
import os
import sys
sys.path.append("..")
import phage_init
import subprocess

# %%
fasta_list=[
#'03_curated_fasta/others_indexed.fasta',
'03_curated_fasta/minor_capsid.fasta',
'03_curated_fasta/tail_fiber.fasta',
'03_curated_fasta/major_tail.fasta',
'03_curated_fasta/portal.fasta',
'03_curated_fasta/minor_tail.fasta',
'03_curated_fasta/baseplate.fasta',
'03_curated_fasta/collar.fasta',
'03_curated_fasta/shaft.fasta',
'03_curated_fasta/major_capsid.fasta',
'03_curated_fasta/HTJ.fasta'
]

# %%
command = '''cat 03_curated_fasta/others_tmp.fasta | perl -lpe 'BEGIN{$i=1} if (/^>/) { print STDERR ">$i\_pat_\\t$_"; s/^>.*$/>$i\_pat_/;$i++;} ' 2> 03_curated_fasta/others.index | perl -pe 'chomp unless (/^>/)' | perl -lpe 's/(?<=.)>/\\n>/' | paste - - | tr '\\t' '\\n' > 03_curated_fasta/others_indexed.fasta'''
print(command)
__=subprocess.run(command,shell=True,check=True, text=True)

# %%
command = 'cat ' + ' '.join(fasta_list) + ''' | sed '/^$/d' | perl -lpe 'BEGIN{$i=1} if (/^>/) {s/^>.*$/>$i\#vi#/;$i++;}'  > 03_curated_fasta/structural_indexed.fasta'''
print(command)
__=subprocess.run(command,shell=True,check=True, text=True)

# %%
command =  '''cat 03_curated_fasta/others_indexed.fasta  03_curated_fasta/structural_indexed.fasta | sed '/^$/d' > 03_curated_fasta/other_plus_structural.fasta'''
print(command)
__=subprocess.run(command,shell=True,check=True, text=True)

# %%
command='''cd-hit -i 03_curated_fasta/other_plus_structural.fasta -o 03_curated_fasta/hitmix.fasta -M 0 -T 0 -c  0.6 -n 3 > errlog'''
print(command)
__=subprocess.run(command,shell=True,check=True, text=True)

# %%
command='''perl get_others_id_argv.pl 03_curated_fasta/hitmix.fasta.clstr 03_curated_fasta/others.index 03_curated_fasta/others_indexed.fasta > 03_curated_fasta/others.fasta'''
print(command)
__=subprocess.run(command,shell=True,check=True, text=True)
