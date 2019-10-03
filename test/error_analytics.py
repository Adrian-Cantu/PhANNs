# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
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
import pickle
import random
import pandas as pd
import numpy

# %%
error1 = pickle.load(open( os.path.join(phage_init.error_dir,"error1.p"), "rb" ))
error2 = pickle.load(open( os.path.join(phage_init.error_dir,"error2.p"), "rb" ))
error3 = pickle.load(open( os.path.join(phage_init.error_dir,"error3.p"), "rb" ))
error4 = pickle.load(open( os.path.join(phage_init.error_dir,"error4.p"), "rb" ))

error_data = numpy.ndarray(shape = (4,8008), dtype = int)
error_data[:0] = error1
error_data[:1] = error2
error_data[:2] = error3
error_data[:3] = error4

#pickle.dump(error_data, open(os.path.join(phage_init.error_dir, "alltrials.p"), "wb"), protocol = 4

# %%
error_average = numpy.ndarray(shape = (8008))
numpy.mean(error_data.copy(), axis = 0, dtype = float, out = error_average)
error_average

# %%
print("Mean # of errors: " + str(error_average.mean(dtype = float)))
print("Standard deviation: " + str(error_average.std(dtype = float)))
print("Most # of errors created: " + str(error_average.max()) + " @ index " + str(error_average.argmax()))
print("Most # of errors reduced: " + str(error_average.min()) + " @ index " + str(error_average.argmin()))

# %%
sorted_indices = error_average.argsort()

#top_x contains an array with the indices of the sequences with the top x number of errors
x_wanted = 20
top_x_reverse = (sorted_indices.copy())[-x_wanted:]
top_x = top_x_reverse[::-1]

#bottom_x contains an array with the indices of the sequences with the top x number of errors
x_wanted = 20
bottom_x = (sorted_indices.copy())[:x_wanted]

print(top_x)
print(bottom_x)

# %%
sorted_errors = numpy.sort((error_average.copy()))
print(sorted_errors)
sorted_reverse = sorted_errors[::-1]
print(sorted_reverse)

# %%
import matplotlib.pyplot as plt
import seaborn as sns
sns.lineplot(numpy.arange(500), sorted_reverse[0:500])

# %%
