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
import pickle
import pandas as pd
import os


# %%
stop_number=0
stop_time_list=['val_loss','val_acc','train_loss']
stop_time=stop_time_list[stop_number]

# %%

# %%
df_dir={'val_loss':'all_results_df_val.p','val_acc':'all_results_df_acc.p','train_loss':'all_results_df.p'}
df=pickle.load(open( os.path.join('07_models',df_dir[stop_time]),'rb'))


# %%
#df=pd.concat([df1,df2])

# %%
#pickle.dump(df,  open( os.path.join('data',"all_results_df.p") , "wb" ), protocol=4)

# %%
all_models=['di_sc','di_sc_p','tri_sc','tri_sc_p','tetra_sc','tetra_sc_p','di','di_p','tri','tri_p','tetra_sc_tri_p','all']

# %%
custom_dict = {'di_sc':0,'di_sc_p':1,'tri_sc':2,'tri_sc_p':3,'tetra_sc':4,'tetra_sc_p':5,'di':6,'di_p':7,
               'tri':8,'tri_p':9,'tetra_sc_tri_p':10,'all':11}

# %%
labels_names=["Major capsid","Minor capsid","Baseplate","Major tail","Minor tail","Portal","Tail fiber",
             "Tail shaft","Collar","Head-Tail joining","Others"]

# %%
from collections import Counter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import matplotlib as mpl
import seaborn as sns
sns.set(style="whitegrid")

# %%
avg_df=df[df['class'] == 'weighted avg']
f1_df=df[df['score_type'] == 'f1-score']

# %%
fig, ax = plt.subplots()
ax.yaxis.grid(True)
#fig.set_size_inches(18, 15)
fig.set_size_inches(8, 6)
    
ax.tick_params(axis='y',labelsize=24)
ax.tick_params(axis='x',labelsize=24, rotation=80)
#ax.set_title('Model metrics : ' + df , fontsize=30,va='bottom')
    #ax.set_title('')
sns.barplot(ax=ax,y="value", x="model", hue="score_type", data=avg_df)
ax.set_ylabel('')    
ax.set_xlabel('')
l = ax.legend()
plt.setp(ax.get_legend().get_texts(), fontsize='18') # for legend text
#print(dir(l))
ax.set(ylim=(0.4, 1))
    
#ax.set_xticklabels(['di','di_p','tri','tri_p','di_sc','di_sc_p','tri_sc','tri_sc_p','all'])
plt.show()
#fig.savefig('08_figures/avg_score_master.png',bbox_inches="tight")
fig.savefig(os.path.join('08_figures','avg_score_master',stop_time))

# %%
import seaborn as sns
import numpy
colors=["#69ef7b", "#b70d61", "#60e9dc", "#473c85", "#b4d170", "#104b6d", "#b4dbe7", "#1c5f1e", "#fd92fa", "#36a620", "#a834bf", "#fda547"]
customPalette = sns.color_palette(colors)

# %%
fig2, ax2 = plt.subplots()
fig2.set_size_inches(18, 15)
#fig2.set_size_inches(8, 6)
sns.set(style="whitegrid")
ax2.tick_params(axis='y',labelsize=30)
ax2.tick_params(axis='x',labelsize=35,rotation=80)
#ax2.set_title('Per model f1-score', fontsize=40,va='bottom')
#sns.barplot(ax=ax2,y="value", x="model", hue="class", data=f1_df, palette="Paired")
sns.barplot(ax=ax2,y="value", x="model", hue="class", data=f1_df, palette=colors)
#sns.barplot(ax=ax2,y="value", x="model", hue="class", data=f1_df)
ax2.set_ylabel('')    
ax2.set_xlabel('')
l = ax2.legend()
plt.setp(ax2.get_legend().get_texts(), fontsize='27') # for legend text
#print(dir(l))
ax2.set(ylim=(0, 1))
ax2.set(xlim=(-0.5, 12.2))
plt.yticks(numpy.arange(0, 1.1, 0.1))
#ax2.set_xticklabels(['di','di_p','tri','tri_p','di_sc','di_sc_p','tri_sc','tri_sc_p','all'])
plt.show()
fig2.savefig(os.path.join('08_figures','f1_score_master_per_model',stop_time))
#fig2.savefig('08_figures/f1_score_master_per_model.png',bbox_inches="tight")

# %%
fig3, ax3 = plt.subplots()
fig3.set_size_inches(18, 15)
sns.set(style="whitegrid")
ax3.tick_params(axis='y',labelsize=30)
ax3.tick_params(axis='x',labelsize=35,rotation=80)
#ax3.set_title('Per class f1-score', fontsize=40,va='bottom')
#sns.barplot(ax=ax3,y="value", x="class", hue="model", data=f1_df)
sns.barplot(ax=ax3,y="value", x="class", hue="model", data=f1_df,palette=colors)
ax3.set_ylabel('')    
ax3.set_xlabel('')
l = ax3.legend()
plt.setp(ax3.get_legend().get_texts(), fontsize='27') # for legend text
#print(dir(l))
#plt.xticks(tick_marks, labels_names, rotation=90)
ax3.set(ylim=(0, 1))
ax3.set(xlim=(-0.5, 12.2))
plt.yticks(numpy.arange(0, 1.1, 0.1))
plt.show()
fig3.savefig(os.path.join('08_figures','f1_score_master_per_class',stop_time))
#fig3.savefig('08_figures/f1_score_master_per_class.png',bbox_inches="tight")

# %%
for x in all_models:
    kk=df.loc[(df['model']==x) & (df['class']=='weighted avg') & (df['score_type'] == 'f1-score')].describe()['value']['mean']
    print(x +' -> ' + str(kk))

# %%
group_arr=pickle.load(open( os.path.join('06_features',"group_arr.p"), "rb" ))
class_arr=pickle.load(open( os.path.join('06_features',"class_arr.p"), "rb" ))
import get_arr
from tensorflow.keras.models import load_model

# %%
#stop using gpu to not run into memoru issues
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# %%
model_dir={'val_loss':'val_','val_acc':'acc_','train_loss':''}
n_members = 10
models = list()
#yhats = numpy.empty((test_X.shape[0],10,11), dtype=numpy.float)
for model_number in range(n_members):
    # load model.
        print('loading ...' + os.path.join('07_models',"tetra_sc_tri_p_{}{:02d}.h5".format(model_dir[stop_time],model_number)))
        model =  load_model( os.path.join('07_models',"tetra_sc_tri_p_{}{:02d}.h5".format(model_dir[stop_time],model_number)) )
    # store in memory
        models.append(model)
    #row=model.predict(test_X)
    #yhats[model_number,:]=row
    #K.clear_session()

# %%
(test_X,test_Y)=get_arr.get_test("tetra_sc_tri_p",class_arr,group_arr)

# %%
yhats = [model.predict(test_X,verbose=2) for model in models]

# %%
import numpy

# %%
yhats_v=numpy.array(yhats)
predicted_Y=numpy.sum(yhats_v, axis=0)
predicted_Y_index = numpy.argmax(predicted_Y, axis=1)

# %%
pickle.dump(predicted_Y_index, open( os.path.join('08_figures','CM',"CM_predicted_test_Y_index_{}.p".format(stop_time)), "wb" ) )
pickle.dump(predicted_Y, open( os.path.join('08_figures','CM',"CM_predicted_test_Y_{}.p".format(stop_time)), "wb" ) )

#pickle.dump(test_Y, open( os.path.join('08_figures','CM',"CM_test_Y.p"), "wb" ) )

# %%
#predicted_Y_index=pickle.load(open( os.path.join('08_figures',"CM_predicted_test_Y.p"), "rb"))
#test_Y=pickle.load(open( os.path.join('08_figures',"CM_test_Y.p"), "rb" ))

# %%
from sklearn.metrics import classification_report
labels_names=["Major capsid","Minor capsid","Baseplate","Major tail","Minor tail","Portal","Tail fiber",
             "Tail shaft","Collar","Head-Tail joining","Others"]
print(classification_report(test_Y, predicted_Y_index, target_names=labels_names ))

# %%
zz=Counter(test_Y)
sample_w=[zz[i] for i in range(0,11,1)]
CM=confusion_matrix(test_Y, predicted_Y_index)
CM_n=CM/numpy.array(sample_w)[:,None]
scale_up=1.4

# %%
plt.viridis()
#plt.grid(b=None)
plt.figure(figsize=[6.4*scale_up, 4.8*scale_up])
plt.imshow(CM_n, interpolation='nearest')
#plt.title('CM : ')
#plt.colorbar()
plt.grid(b=None)
tick_marks = numpy.arange(len(labels_names))
plt.xticks(tick_marks, labels_names, rotation=90,fontsize='15')
plt.yticks(tick_marks, labels_names,fontsize='15')
fmt = '.2f'
for i, j in itertools.product(range(CM_n.shape[0]), range(CM_n.shape[1])):
    plt.text(j, i, format(CM_n[i, j], fmt),horizontalalignment="center",verticalalignment='center',
    color="white" if CM_n[i, j] < 0.50 else "black")
plt.ylabel('True Class',fontsize='20')
plt.xlabel('Predicted Class',fontsize='20')
plt.clim(0,1)
#plt.savefig('08_figures/tetra_sc_tri_p_CM.png',bbox_inches="tight")
plt.show()
fig.savefig(os.path.join('08_figures','CM',stop_time))

# %%
kk=classification_report(test_Y, predicted_Y_index, target_names=labels_names, output_dict=True )


# %%
kk
