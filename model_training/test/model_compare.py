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

import numpy
import pickle
from sklearn.metrics import classification_report

# %%
df90=pickle.load(open( "../data_v3.3/kfold_df.p", "rb" ))
df80=pickle.load(open( "../data80/kfold_df.p", "rb" ))
df70=pickle.load(open( "../data70/kfold_df.p", "rb" ))
df50=pickle.load(open( "../data50/kfold_df.p", "rb" ))
df50_s=pickle.load(open( "../data50_sub/kfold_df.p", "rb" ))
#df_log=pickle.load(open( "../logistic/kfold_df.p", "rb" ))
#df_log=pickle.load(open( "../undereplicate/log_kfold_df.p", "rb" ))
df_log=pickle.load(open( "../undereplicate/log_kfold_fast_df.p", "rb" ))
df_un=pickle.load(open( "../undereplicate/data/all_results_df.p", "rb" ))
df_un_val=pickle.load(open( "../undereplicate/data/val_all_results_df.p", "rb" ))
df_un_acc=pickle.load(open( "../undereplicate/data/acc_all_results_df.p", "rb" ))
df40=pickle.load(open( "../undereplicate/data30/val_30_all_results_df.p", "rb" ))

# %%
my_df_dic= {'90':df90,'80':df80,'70':df70,'50':df50,'50_s':df50_s,'40':df40,'logistic':df_log, 'undereplicate':df_un,
           'undereplicate_val':df_un_val,'undereplicate_acc':df_un_acc}
df_n=['90','80','70','50','50_s','40','logistic','undereplicate','undereplicate_val','undereplicate_acc']          

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
test_p_un=pickle.load(open( "../undereplicate/data/CM_predicted_test_Y.p", "rb" ))
test_p_un_val=pickle.load(open( "../undereplicate/data/val_CM_predicted_test_Y.p", "rb" ))
test_p_un_acc=pickle.load(open( "../undereplicate/data/acc_CM_predicted_test_Y.p", "rb" ))

# %%
test_90=pickle.load(open( "../model_v3.3/CM_test_Y.p", "rb")) 
test_80=pickle.load(open( "../model80/CM_test_Y.p", "rb" ))
test_70=pickle.load(open( "../model70/CM_test_Y.p", "rb" ))
test_50=pickle.load(open( "../model50/CM_test_Y.p", "rb" ))
test_50_s=pickle.load(open( "../model50_sub/CM_test_Y.p", "rb" ))
test_ll=pickle.load(open( "../logistic/CM_test_Y.p", "rb" ))
test_un=pickle.load(open( "../undereplicate/data/CM_test_Y.p", "rb" ))

# %%
labels_names=["Major capsid","Minor capsid","Baseplate","Major tail","Minor tail","Portal","Tail fiber",
             "Tail shaft","Collar","Head-Tail joining","Others"]

# %%
test_p_dic= {'90':test_p_90,'80':test_p_80,'70':test_p_70,'50':test_p_50,'50_s':test_p_50_s,'logistic':test_p_ll,
             'undereplicate':test_p_un,'undereplicate_val':test_p_un_val,'undereplicate_acc':test_p_un_acc}
test_dic= {'90':test_90,'80':test_80,'70':test_70,'50':test_50,'50_s':test_50_s,'logistic':test_ll,
           'undereplicate':test_un,'undereplicate_val':test_un,'undereplicate_acc':test_un}

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
#for df in ['undereplicate_val','40']:
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
print('average average f1-score')
for df in df_n :
    print(df)
    for x in all_models:
        kk=my_df_dic[df].loc[(my_df_dic[df]['model']==x) & (my_df_dic[df]['class']=='weighted avg') & (my_df_dic[df]['score_type'] == 'f1-score')].describe()['value']['mean']
        print(x.rjust(15,' ') +" -> " + "{:.3f}".format(kk))
        data_row=[x,df,kk]
        df_all_f1=df_all_f1.append(pd.Series(data_row,index=df_all_f1.columns),sort=False,ignore_index=True)


# %%
dd = {'model': [],'Model':[],'F1-score':[]}
df_si_f1 = pd.DataFrame(data=dd)
legend_dic={'40':'1d-10d ANN ensemble','logistic':'logistic regression','undereplicate_val':'1D-10D ANN ensemble'}
for df in ['40','logistic','undereplicate_val']  :
#for df in df_n:
#    print(df)
    for x in all_models:
        kk=my_df_dic[df].loc[(my_df_dic[df]['model']==x) & (my_df_dic[df]['class']=='weighted avg') & (my_df_dic[df]['score_type'] == 'f1-score')]
        for index, row in kk.iterrows():
            data_row=[x,legend_dic[df],row['value']]
            df_si_f1=df_si_f1.append(pd.Series(data_row,index=df_si_f1.columns),sort=False,ignore_index=True)

# %%
colors=["#69ef7b", "#b70d61", "#60e9dc", "#473c85", "#b4d170", "#104b6d", "#b4dbe7", "#1c5f1e", "#fd92fa", "#36a620", "#a834bf", "#fda547"]
customPalette = sns.color_palette(colors)

# %%
fig3, ax3 = plt.subplots()
fig3.set_size_inches(13, 7)
sns.boxplot(x='model',y='F1-score',hue='Model', ax=ax3, 
#            hue_order=['90%','80%','70%','50%','logistic%','undereplicate%',
#                       'undereplicate_val%','undereplicate_acc%']
            hue_order=[legend_dic['logistic'],legend_dic['40'],legend_dic['undereplicate_val']]
                        ,data=df_si_f1, palette=colors )
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
ax3.set_ylabel('10-fold Cross-Validation F1-score',fontsize='20')    
ax3.set_xlabel('')
ax3.set_yticks(numpy.arange(0, 1.1, 0.025),minor=False)
ax3.set(ylim=(0.4, 1))
#plt.setp(ax3.get_legend().get_title(), fontsize='20',text='Dereplication')
plt.setp(ax3.get_legend().get_title(),fontsize='0',text='')
plt.setp(ax3.get_legend().get_texts(), fontsize='20')
fig3.savefig('derep_per_model.png',bbox_inches="tight")
plt.show()

# %%
xx='tetra_sc_tri_p'
ddd = {'class': [],'Model':[],'F1-score':[]}
df_class_f1 = pd.DataFrame(data=ddd)
for df in ['40','logistic','undereplicate_val']  :
    kk=my_df_dic[df].loc[(my_df_dic[df]['model']==xx) & (my_df_dic[df]['class']!='weighted avg') & (my_df_dic[df]['score_type'] == 'f1-score')]
    for index, row in kk.iterrows():
        data_row=[row['class'],legend_dic[df],row['value']]
        df_class_f1=df_class_f1.append(pd.Series(data_row,index=df_class_f1.columns),sort=False,ignore_index=True)

# %%
fig4, ax4 = plt.subplots()
fig4.set_size_inches(13, 7)
sns.boxplot(x='class',y='F1-score',hue='Model', ax=ax4, 
#            hue_order=['90%','80%','70%','50%','logistic%','undereplicate%',
#                       'undereplicate_val%','undereplicate_acc%']
            hue_order=[legend_dic['logistic'],legend_dic['40'],legend_dic['undereplicate_val']]
                        ,data=df_class_f1, palette=colors )
ax4.tick_params(axis='y',labelsize=15)
ax4.tick_params(axis='x',labelsize=20,rotation=90)
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
#plt.axvline(10.5, linestyle = '--', color = 'g')
ax4.set_ylabel('10-fold Cross-Validation F1-score',fontsize='20')    
ax4.set_xlabel('')
ax4.set_yticks(numpy.arange(0, 1.1, 0.1),minor=False)
ax4.set(ylim=(0, 1))
#plt.setp(ax3.get_legend().get_title(), fontsize='20',text='Dereplication')
plt.setp(ax4.get_legend().get_title(),fontsize='0',text='')
plt.setp(ax4.get_legend().get_texts(), fontsize='20')
fig4.savefig('derep_per_class.png',bbox_inches="tight")
plt.show()
