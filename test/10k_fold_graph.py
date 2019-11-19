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
#import keras and numpy
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Dropout
from keras.optimizers import Adam
from keras import backend as K
import pickle
from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

# %%
#stop using gpu to not run into memoru issues
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# %%
#this list the devices, just making sure there is a GPU present, you might be fine with no GPU
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# %%
import pandas as pd
d = {'model': [], 'class': [],'score_type':[],'value':[]}
df = pd.DataFrame(data=d)


# %%
def add_to_df(df,test_Y_index, test_Y_predicted,model_name):
    labels_names=["Major capsid","Minor capsid","Baseplate","Major tail","Minor tail","Portal","Tail fiber",
             "Tail shaft","Collar","Head-Tail joining","Others"]
    labels_dataframe=["Major capsid","Minor capsid","Baseplate","Major tail","Minor tail","Portal","Tail fiber",
                 "Tail shaft","Collar","Head-Tail joining","Others","weighted avg"]
    report=classification_report(test_Y_index, test_Y_predicted, target_names=labels_names,output_dict=True )
    for label in labels_dataframe:
        #data_row=[report[label][i] for i in ['precision','recall',"f1-score"]]
        #data_row.insert(0,label)
        #data_row.insert(0,model_name)
        score_type='precision'
        data_row=[model_name,label,score_type,report[label][score_type]]
        df=df.append(pd.Series(data_row,index=df.columns),sort=False,ignore_index=True)
        score_type='recall'
        data_row=[model_name,label,score_type,report[label][score_type]]
        df=df.append(pd.Series(data_row,index=df.columns),sort=False,ignore_index=True)
        score_type='f1-score'
        data_row=[model_name,label,score_type,report[label][score_type]]
        df=df.append(pd.Series(data_row,index=df.columns),sort=False,ignore_index=True)
    return df


# %%
df = pd.DataFrame(data=d)

# %%
all_models=['di','di_p','tri','tri_p','di_sc','di_sc_p','tri_sc','tri_sc_p','tetra_sc','tetra_sc_p','tetra_sc_tri_p','all']

# %%
for this_model in all_models:
    (test_X,test_Y)=ann_data.get_formated_test(this_model)
    test_Y_index = test_Y.argmax(axis=1)
    print('Runing model '+this_model)
    for model_number in range(1, 11):
        #print("\tsplit  "+ "{:02d}".format(model_number),end=' ', flush=True)
        print("\tsplit  "+ "{:02d}".format(model_number))
        model = load_model( os.path.join(phage_init.model_dir,this_model+'_'+"{:02d}".format(model_number)+'.h5') )
        test_Y_predicted = model.predict_classes(test_X)
        #print(classification_report(test_Y_index, test_Y_predicted, target_names=labels_names ))
        df=add_to_df(df,test_Y_index, test_Y_predicted,this_model)
        #print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        #F.write(this_model)
        #F.write(classification_report(test_Y_index, test_Y_predicted, target_names=labels_names ))
        del model
        K.clear_session()

# %%
pickle.dump(df, open( os.path.join(phage_init.data_dir,"kfold_df.p"), "wb" ) )
#df=pickle.load(open( os.path.join(phage_init.data_dir,"kfold_df.p"), "rb" ))

# %%
#pickle.dump(df, open( os.path.join(phage_init.data_dir,"kfold_df_p.p"), "wb" ) )
#df=pickle.load(open( os.path.join(phage_init.data_dir,"kfold_df_p.p"), "rb" ))

# %%
import seaborn as sns
import matplotlib.pyplot as plt


# %%
custom_dict = {'di_sc':0,'di_sc_p':1,'tri_sc':2,'tri_sc_p':3,'tetra_sc':4,'tetra_sc_p':5,'di':6,'di_p':7,
               'tri':8,'tri_p':9,'tetra_sc_tri_p':10,'all':11}


f1_df=df[df['score_type'] == 'f1-score']
f1_df['model'] = pd.Categorical(
    f1_df['model'], categories=sorted(custom_dict, key=custom_dict.get), ordered=True
)

avg_df=df[df['class'] == 'weighted avg']
avg_df['model'] = pd.Categorical(
    avg_df['model'], categories=sorted(custom_dict, key=custom_dict.get), ordered=True
)

# %%

# %%
fig, ax = plt.subplots()

#fig.set_size_inches(18, 15)
fig.set_size_inches(8, 6)
sns.set(style="whitegrid")
ax.tick_params(axis='y',labelsize=24)
ax.tick_params(axis='x',labelsize=24, rotation=80)
#ax.set_title('Weighted average model metrics', fontsize=30,va='bottom')
ax.set_title('')
sns.barplot(ax=ax,y="value", x="model", hue="score_type", data=avg_df)
ax.set_ylabel('')    
ax.set_xlabel('')
l = ax.legend()
plt.setp(ax.get_legend().get_texts(), fontsize='18') # for legend text
#print(dir(l))
ax.set(ylim=(0.4, 1))
#ax.set_xticklabels(['di','di_p','tri','tri_p','di_sc','di_sc_p','tri_sc','tri_sc_p','all'])
plt.show()
fig.savefig('avg_score_master.png',bbox_inches="tight")

# %%
#ax.properties()['children']
#ax.lines[0].get_data()

kk= [(x.get_x(),x.get_height()) for x in ax.patches]
kkk =numpy.array(kk,dtype=[('x', float), ('y', float)])
kkk

# %%
kkkk=numpy.sort(kkk,order='x')
for xx in zip(kkkk[0::3],all_models):
    print(xx[0],xx[1])

# %%
#colors = ["#C46210", "#2E5894","#9C2542","#A57164","#58427C","#85754E","#319177","#9C7C38","#8D4E85","#8FD400","#D98695","#0081AB"]
#colors=["#69ef7b", "#b70d61", "#60e9dc", "#473c85", "#b4d170", "#104b6d", "#b4dbe7", "#1c5f1e", "#fd92fa", "#36a620", "#a834bf", "#fda547"]
#colors=["#fda547", "#691b9e", "#8dba22", "#c551dc", "#0b5313", "#38b5fc", "#752e4f", "#3fe34b", "#f24325", "#94e4ca", "#1f4196", "#ffacec"]
colors=["#fda547", "#2c4a5e", "#83e9a3", "#cc156d", "#1d8a20", "#8711ac", "#80de1a", "#ef66f0", "#687f39", "#30b6f3", "#743502", "#adbea6"]
#colors = ["#C46210", "#2E5894"]
customPalette = sns.color_palette(colors)
#sns.set_palette(colors)

# %%
sns.palplot(customPalette)

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
fig2.savefig('f1_score_master_per_model.png',bbox_inches="tight")

# %%
handles,labels = ax2.get_legend_handles_labels()
fige, axe = plt.subplots()
axe.legend(handles, labels, loc='center')
axe.xaxis.set_visible(False)
axe.yaxis.set_visible(False)
for v in axe.spines.values():
    v.set_visible(False)
plt.show()

# %%
fig3, ax3 = plt.subplots()
fig3.set_size_inches(18, 15)
sns.set(style="whitegrid")
ax3.tick_params(axis='y',labelsize=30)
ax3.tick_params(axis='x',labelsize=35,rotation=80)
#ax3.set_title('Per class f1-score', fontsize=40,va='bottom')
sns.barplot(ax=ax3,y="value", x="class", hue="model", data=f1_df)
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
fig3.savefig('f1_score_master_per_class.png',bbox_inches="tight")

# %%
df.to_csv("sup_data_1.tsv", sep='\t', encoding='utf-8')

# %%
fig4, ax4 = plt.subplots()
fig4.set_size_inches(18, 15)
sns.set(style="whitegrid")
ax4.tick_params(axis='y',labelsize=18)
ax4.tick_params(axis='x',labelsize=18,rotation=80)
ax4.set_title('Per class f1-score', fontsize=40,va='bottom')
sns.boxplot(ax=ax4,y="value", x="class", hue="model", data=f1_df, linewidth=2)
ax4.set_ylabel('')    
ax4.set_xlabel('')
l = ax4.legend()
plt.setp(ax4.get_legend().get_texts(), fontsize='22') # for legend text
#print(dir(l))
#plt.xticks(tick_marks, labels_names, rotation=90)
ax4.set(ylim=(0, 1))
ax4.set(xlim=(-0.5, 11.5))
#ax2.set_xticklabels(['di','di_p','tri','tri_p','di_sc','di_sc_p','tri_sc','tri_sc_p','all'])
plt.axvline(0.5, linestyle = '--', color = 'g')
plt.axvline(1.5, linestyle = '--', color = 'g')
plt.axvline(2.5, linestyle = '--', color = 'g')
plt.axvline(3.5, linestyle = '--', color = 'g')
plt.axvline(4.5, linestyle = '--', color = 'g')
plt.axvline(5.5, linestyle = '--', color = 'g')
plt.axvline(6.5, linestyle = '--', color = 'g')
plt.axvline(7.5, linestyle = '--', color = 'g')
plt.axvline(8.5, linestyle = '--', color = 'g')
plt.axvline(9.5, linestyle = '--', color = 'g')
plt.axvline(10.5, linestyle = '--', color = 'g')
plt.show()
fig4.savefig('f1_score_master_per_class_boxplot.png',bbox_inches="tight")

# %%
fig5, ax5 = plt.subplots()

#fig.set_size_inches(18, 15)
fig5.set_size_inches(8, 6)
sns.set(style="whitegrid")
ax5.tick_params(axis='y',labelsize=24)
ax5.tick_params(axis='x',labelsize=24, rotation=80)
#ax.set_title('Weighted average model metrics', fontsize=30,va='bottom')
ax5.set_title('')
sns.barplot(ax=ax5,y="value", x="model", hue="score_type", data=avg_df)
ax5.set_ylabel('')    
ax5.set_xlabel('')
l = ax5.legend()
plt.setp(ax5.get_legend().get_texts(), fontsize='18') # for legend text
#print(dir(l))
ax5.set(ylim=(0.4, 1))
#ax.set_xticklabels(['di','di_p','tri','tri_p','di_sc','di_sc_p','tri_sc','tri_sc_p','all'])
plt.show()

# %%
fig3, ax3 = plt.subplots()
fig3.set_size_inches(10, 10)
sns.set(style="whitegrid")
ax3.tick_params(axis='y',labelsize=20)
ax3.tick_params(axis='x',labelsize=25,rotation=80)
#ax3.set_title('Per class f1-score', fontsize=40,va='bottom')
sns.barplot(ax=ax3,y="value", x="class", data=f1_df)
ax3.set_ylabel('')    
ax3.set_xlabel('')
l = ax3.legend()
plt.setp(ax3.get_legend().get_texts(), fontsize='22') # for legend text
#print(dir(l))
#plt.xticks(tick_marks, labels_names, rotation=90)
ax3.set(ylim=(0, 1))
ax3.set(xlim=(-0.5, 12.2))
plt.yticks(numpy.arange(0, 1.1, 0.1))
plt.show()

# %%
 K.clear_session()
