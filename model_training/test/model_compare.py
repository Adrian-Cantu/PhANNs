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

# %%
import ann_data

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# %%
#import keras and numpy
import numpy
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.layers import LSTM
#from tensorflow.keras.layers import Activation
#from tensorflow.keras.layers import Dropout
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras import backend as K
import pickle
#from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
#from sklearn.model_selection import StratifiedKFold

# %%
df90=pickle.load(open( "../data_v3.3/kfold_df.p", "rb" ))
df80=pickle.load(open( "../data80/kfold_df.p", "rb" ))
df70=pickle.load(open( "../data70/kfold_df.p", "rb" ))
df50=pickle.load(open( "../data50/kfold_df.p", "rb" ))
df50_s=pickle.load(open( "../data50_sub/kfold_df.p", "rb" ))
df_log=pickle.load(open( "../logistic/kfold_df.p", "rb" ))

# %%
my_df_dic= {'90':df90,'80':df80,'70':df70,'50':df50,'50_s':df50_s,'logistic':df_log}
df_n=['90','80','70','50','50_s','logistic']          

# %%
all_models=['di_sc','di_sc_p','tri_sc','tri_sc_p','tetra_sc','tetra_sc_p','di','di_p','tri','tri_p','tetra_sc_tri_p','all']

# %%
custom_dict = {'di_sc':0,'di_sc_p':1,'tri_sc':2,'tri_sc_p':3,'tetra_sc':4,'tetra_sc_p':5,'di':6,'di_p':7,
               'tri':8,'tri_p':9,'tetra_sc_tri_p':10,'all':11}

# %%
test_p_90=pickle.load(open( "../model_v3.3/CM_predicted_test_Y.p", "rb")) 
test_p_80=pickle.load(open( "../model80/CM_predicted_test_Y.p", "rb" ))
test_p_70=pickle.load(open( "../model70/CM_predicted_test_Y.p", "rb" ))
test_p_50=pickle.load(open( "../model50/CM_predicted_test_Y.p", "rb" ))
test_p_50_s=pickle.load(open( "../model50_sub/CM_predicted_test_Y.p", "rb" ))
test_p_ll=pickle.load(open( "../logistic/CM_predicted_test_Y.p", "rb" ))

# %%
test_90=pickle.load(open( "../model_v3.3/CM_test_Y.p", "rb")) 
test_80=pickle.load(open( "../model80/CM_test_Y.p", "rb" ))
test_70=pickle.load(open( "../model70/CM_test_Y.p", "rb" ))
test_50=pickle.load(open( "../model50/CM_test_Y.p", "rb" ))
test_50_s=pickle.load(open( "../model50_sub/CM_test_Y.p", "rb" ))
test_ll=pickle.load(open( "../logistic/CM_test_Y.p", "rb" ))

# %%
labels_names=["Major capsid","Minor capsid","Baseplate","Major tail","Minor tail","Portal","Tail fiber",
             "Tail shaft","Collar","Head-Tail joining","Others"]

# %%
test_p_dic= {'90':test_p_90,'80':test_p_80,'70':test_p_70,'50':test_p_50,'50_s':test_p_50_s,'logistic':test_p_ll}
test_dic= {'90':test_90,'80':test_80,'70':test_70,'50':test_50,'50_s':test_50_s,'logistic':test_ll}

# %%
from collections import Counter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import matplotlib as mpl

# %%
for df in df_n:
    #print(classification_report(test_dic[df], test_p_dic[df], target_names=labels_names ))
    zz=Counter(test_dic[df])
    sample_w=[zz[i] for i in range(0,11,1)]
    CM=confusion_matrix(test_dic[df], test_p_dic[df])
    CM_n=CM/numpy.array(sample_w)[:,None]
    scale_up=1.4
    plt.figure(figsize=[6.4*scale_up, 4.8*scale_up])
    plt.imshow(CM_n, interpolation='nearest')
    plt.title('CM : ' + df)
    plt.colorbar()
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
    plt.savefig('tetra_sc_tri_p_CM_'+df+'.png',bbox_inches="tight")
    plt.show()

# %%
for df in df_n:
    print("\ntetra_sc_tri_p : "+df+"\n")
    print(classification_report(test_dic[df], test_p_dic[df], target_names=labels_names ))

# %%
sns.set(style="whitegrid")
for df in df_n:
    avg_df=my_df_dic[df][my_df_dic[df]['class'] == 'weighted avg']
    f1_df=my_df_dic[df][my_df_dic[df]['score_type'] == 'f1-score']
    fig, ax = plt.subplots()
    ax.yaxis.grid(True)
#fig.set_size_inches(18, 15)
    fig.set_size_inches(8, 6)
    
    ax.tick_params(axis='y',labelsize=24)
    ax.tick_params(axis='x',labelsize=24, rotation=80)
    ax.set_title('Model metrics : ' + df , fontsize=30,va='bottom')
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
    fig.savefig('avg_score_master_'+df+'.png',bbox_inches="tight")

# %%
import pandas as pd
d = {'model': [],'derep':[],'mean':[]}
df_all_f1 = pd.DataFrame(data=d)

# %%
df_log.loc[(df_log['model']=="di_sc") & (df_log['class']=='weighted avg') & (df_log['score_type'] == 'f1-score')].describe()

# %%
for df in ['90','80','70','50','logistic']  :
    print(df)
    for x in all_models:
        kk=my_df_dic[df].loc[(my_df_dic[df]['model']==x) & (my_df_dic[df]['class']=='weighted avg') & (my_df_dic[df]['score_type'] == 'f1-score')].describe()['value']['mean']
        print(x.rjust(15,' ') +" -> " + "{:.3f}".format(kk))
        #df_all_f1['model']=x
        #df_all_f1['derep']=df
        #df_all_f1['mean']=kk
        data_row=[x,df,kk]
        df_all_f1=df_all_f1.append(pd.Series(data_row,index=df_all_f1.columns),sort=False,ignore_index=True)


# %%
dd = {'model': [],'dereplication level':[],'f1-score':[]}
df_si_f1 = pd.DataFrame(data=dd)
for df in ['90','80','70','50','logistic']  :
    print(df)
    for x in all_models:
        kk=my_df_dic[df].loc[(my_df_dic[df]['model']==x) & (my_df_dic[df]['class']=='weighted avg') & (my_df_dic[df]['score_type'] == 'f1-score')]
        for index, row in kk.iterrows():
            #df_all_f1['model']=x
            #df_all_f1['derep']=df
            #df_all_f1['mean']=kk
            data_row=[x,df+'%',row['value']]
            df_si_f1=df_si_f1.append(pd.Series(data_row,index=df_si_f1.columns),sort=False,ignore_index=True)

# %%
df_si_f1

# %%
colors=["#69ef7b", "#b70d61", "#60e9dc", "#473c85", "#b4d170", "#104b6d", "#b4dbe7", "#1c5f1e", "#fd92fa", "#36a620", "#a834bf", "#fda547"]
customPalette = sns.color_palette(colors)

# %%
fig2, ax2 = plt.subplots()
fig2.set_size_inches(7, 7)
sns.pointplot(x='derep',y='mean',hue='model', ax=ax2, data=df_si_f1, palette=colors,order=['90','80','70','50']  )
ax2.tick_params(axis='y',labelsize=30)
ax2.tick_params(axis='x',labelsize=35,rotation=80)
ax2.set_ylabel('')    
ax2.set_xlabel('')
#plt.yticks(numpy.arange(0, 1.1, 0.1))
ax2.set(xlim=(-0.5, 4.7))
fig2.savefig('f1vsderep.png',bbox_inches="tight")

# %%
fig3, ax3 = plt.subplots()
fig3.set_size_inches(7, 7)
sns.boxplot(x='model',y='f1-score',hue='dereplication level', ax=ax3, hue_order=['90%','80%','70%','50%','logistic%'],data=df_si_f1, palette=colors )
ax3.tick_params(axis='y',labelsize=15)
ax3.tick_params(axis='x',labelsize=20,rotation=90)
plt.axvline( 0.5, linestyle = '--', color = 'g')
plt.axvline( 1.5, linestyle = '--', color = 'g')
plt.axvline( 2.5, linestyle = '--', color = 'g')
plt.axvline( 3.5, linestyle = '--', color = 'g')
plt.axvline( 4.5, linestyle = '--', color = 'g')
plt.axvline( 5.5, linestyle = '--', color = 'g')
plt.axvline( 6.5, linestyle = '--', color = 'g')
plt.axvline( 7.5, linestyle = '--', color = 'g')
plt.axvline( 8.5, linestyle = '--', color = 'g')
plt.axvline( 9.5, linestyle = '--', color = 'g')
plt.axvline(10.5, linestyle = '--', color = 'g')
ax3.set_ylabel('10-fold Cross-Validatio f1-score',fontsize='20')    
ax3.set_xlabel('')
ax3.set_yticks(numpy.arange(0, 1.1, 0.025),minor=False)
ax3.set(ylim=(0.4, 1))
plt.setp(ax3.get_legend().get_title(), fontsize='20',text='Dereplication')
plt.setp(ax3.get_legend().get_texts(), fontsize='20')
fig3.savefig('f1vsderep.png',bbox_inches="tight")
plt.show()
