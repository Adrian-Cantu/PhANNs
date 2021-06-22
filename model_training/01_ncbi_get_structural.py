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
# ## ncbi_get_structural
#
# This script connects to NCBI to download all non curated, non de-replicated sequences. Code is included for reference. unless you want to rebuild the database or add a new class, there is no need to run this.

# %%
import os
import sys
sys.path.append("..")
import phage_init

# %%
from Bio import Entrez
from urllib.error import HTTPError
Entrez.email = ""
#Entrez.api_key = ""

# %% [markdown]
# ## get_search and get_full_search
# Construct search terms for each of the classes

# %%


def get_search(term, extra=''):
    handle = Entrez.esearch(db="protein", term='('+ term +'[Title]) AND phage[Title] NOT hypothetical[Title] ' +
                        'NOT putative[Title] AND 50:1000000[SLEN] NOT putitive[Title] ' +
                        'NOT probable[Title] NOT possible[Title] NOT unknown[Title] ' + extra,
                        idtype="acc",usehistory="y")
    #,retmax=2000
    search_results = Entrez.read(handle)
    handle.close()
    return search_results



# %%


def get_full_search(term, extra=''):
    handle = Entrez.esearch(db="protein", term=term +' AND phage[Title] NOT hypothetical[Title] ' +
                        'NOT putative[Title] AND 50:1000000[SLEN] NOT putitive[Title] ' +
                        'NOT probable[Title] NOT possible[Title] NOT unknown[Title]' + extra,
                        idtype="acc",usehistory="y")
    search_results = Entrez.read(handle)
    handle.close()
    return search_results



# %% [markdown]
# ## get_sequences
#
# Download sequences to fasta files. Sequence retrieval is prone to fail depending on the load of the servers. You can restart the download from a particular batch (use the last batch mentioned before any error). A new file with the part number in its tilte will be created. Concatenate all parts to get the final fasta.
#

# %%

def get_sequences(search_results, out="out.fasta", batch_size = 100, start_batch=0):
    webenv = search_results["WebEnv"]
    query_key = search_results["QueryKey"]
    count= int(search_results["Count"])
    
    
    file_name = out
    if os.path.isfile(file_name):
        expand = 1
        while True:
            expand += 1
            new_file_name = file_name.split(".fasta")[0] + str(expand) + ".fasta"
            if os.path.isfile(new_file_name):
                continue
            else:
                file_name = new_file_name
                break
    out_handle = open(file_name, "w")
    for start in range(start_batch*batch_size, count, batch_size):
        end = min(count, start+batch_size)
        print("Going to download record %i to %i of %i (batch %i)" % (start+1, end, count, start/batch_size))
        attempt = 0
        while attempt < 3:
            attempt += 1
            try:
                fetch_handle = Entrez.efetch(db="protein",
                                         rettype="fasta", retmode="text",
                                         retstart=start, retmax=batch_size,
                                         webenv=webenv, query_key=query_key,
                                         idtype="acc")
            except HTTPError as err:
                if 500 <= err.code <= 599:
                    print("Received error from server %s" % err)
                    print("Attempt %i of 3" % attempt)
                    time.sleep(15)
                else:
                    raise
        data = fetch_handle.read()
        fetch_handle.close()
        out_handle.write(data)
    out_handle.close()
    print("Done")



# %% [markdown]
# ## Downloading sequences

# %%
search_results = get_search('capsid')

# %%
get_sequences(search_results,out='01_fasta/major_capsid.fasta',start_batch=573)

# %%
#search_results = get_search('minor capsid')

# %%
#get_sequences(search_results,out='01_fasta/minor_capsid.fasta',start_batch=0)

# %%
search_results = get_search('baseplate')

# %%
get_sequences(search_results,out='01_fasta/baseplate.fasta',start_batch=0)

# %%
search_results = get_search('major tail')

# %%
get_sequences(search_results,out='01_fasta/major_tail.fasta',start_batch=846)

# %%
search_results = get_search('minor tail')

# %%
get_sequences(search_results,out='01_fasta/minor_tail.fasta',start_batch=291)

# %%
search_results = get_search('portal')

# %%
get_sequences(search_results,out='01_fasta/portal.fasta',start_batch=1871)

# %%
search_results = get_search('tail fiber')

# %%
get_sequences(search_results,out='01_fasta/tail_fiber.fasta',start_batch=141)

# %%
search_results = get_search('collar')

# %%
get_sequences(search_results,out='01_fasta/collar.fasta',start_batch=0)

# %%
search_results = get_full_search('tail[Title] AND (shaft[Title] OR sheath[Title])')

# %%
get_sequences(search_results,out='01_fasta/shaft.fasta',start_batch=368)

# %%
search_results = get_search('head-tail')

# %%
get_sequences(search_results,out='01_fasta/HTJ.fasta',start_batch=693)

# %%
