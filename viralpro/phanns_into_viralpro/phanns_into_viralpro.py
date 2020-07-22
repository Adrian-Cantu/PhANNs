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
import pandas as pd
import numpy
import seaborn as sns
import pickle

# %%
capsid_scores=pd.read_csv('all_phanns_into_viralpro_capsids.csv',index_col=0,header=None,names=['id','score'],sep='\s+' )

# %%
tail_scores=pd.read_csv('all_phanns_into_viralpro_tails.csv',index_col=0,header=None,names=['id','score'],sep='\s+' )

# %%
class_index=['major_capsid','baseplate',
               'major_tail','minor_tail','portal',
               'tail_fiber','shaft','collar',
               'HTJ','other']

# %%
kk=capsid_scores[['major_capsid' in x for x in capsid_scores.index.values]]

# %%
pos = sum(kk['score'] > 0)
total = kk.shape[0]

# %%
for prot_class in class_index:
    kk=capsid_scores[[prot_class in x for x in capsid_scores.index.values]]
    pos = sum(kk['score'] > 0)
    total = kk.shape[0]
    prop=pos/total
    print(prot_class)
    print(prop)
    print()

# %%
for prot_class in class_index:
    kk=tail_scores[[prot_class in x for x in capsid_scores.index.values]]
    pos = sum(kk['score'] > 0)
    total = kk.shape[0]
    prop=pos/total
    print(prot_class)
    print(prop)
    print()

# %%
pos_tail=['baseplate','major_tail','minor_tail','tail_fiber','shaft','collar']
neg_tail=['major_capsid','portal','HTJ','other']

pos_capsid=['major_capsid']
neg_capsid=['baseplate','major_tail','minor_tail','portal','tail_fiber','shaft','collar','HTJ','other']


# %%
def class_scores(tt,dataframe,poss_class,hue,df):
    is_real=[any([prot_class in x for prot_class in poss_class]) for x in dataframe.index.values]
    is_predicted =dataframe['score']>=tt
    TP=sum(numpy.logical_and(is_real,is_predicted))
    FN=sum(numpy.logical_and(is_real,numpy.logical_not(is_predicted)))
    TN=sum(numpy.logical_and(numpy.logical_not(is_real),numpy.logical_not(is_predicted)))
    FP=sum(numpy.logical_and(numpy.logical_not(is_real),is_predicted))
    #is_real_negative=len(dataframe.index.values)-sum(is_real)
    #FP=is_real_negative-TN
    num_pred=TP+FP
    if not num_pred:
        precision=0
    else:
        precision=TP/num_pred
    recall=TP/(TP+FN)
    specificity=TN/(TN+FP)
    false_positive_rate=FP/(FP+TN)
    fscore=(2*TP)/(2*TP+FP+FN)
    accuracy=(TP+TN)/(TP+TN+FP+FN)
    data_row=[hue,precision,recall,fscore,specificity,false_positive_rate,accuracy,tt]
    df=df.append(pd.Series(data_row,index=df.columns),sort=False,ignore_index=True)
    return df


# %%
d = {'class':[],'precision': [], 'recall': [],'f1-score':[],'specificity':[],
     'false_positive_rate':[],'accuracy':[],'threshold':[]}
df = pd.DataFrame(data=d)
df2 = pd.DataFrame(data=d)
score_range=numpy.arange(min(capsid_scores['score']),max(capsid_scores['score'])+0.001,0.1)
print('capsid')
for tt in score_range:
    df=class_scores(tt,capsid_scores,pos_capsid,'capsid',df)
    print(tt,end="\r")
score_range=numpy.arange(min(tail_scores['score']),max(tail_scores['score'])+0.001,0.1)
print()
print('tails')
for tt in score_range:
    df=class_scores(tt,tail_scores,pos_tail,'tail',df)
    print(tt,end="\r")
df2=class_scores(0,capsid_scores,pos_capsid,'capsid',df2)
df2=class_scores(0,tail_scores,pos_tail,'tail',df2)

# %%
import matplotlib.pyplot as plt
import matplotlib as mpl


# %%

#colors=["#69ef7b", "#b70d61", "#60e9dc", "#473c85", "#b4d170", "#104b6d", "#b4dbe7", "#1c5f1e", "#fd92fa", "#36a620", "#a834bf", "#fda547"]
colors=["#69ef7b", "#b70d61"]

customPalette = sns.color_palette(colors)
size_d={'capsid':3,'tail':3}

# %%
fig, ax = plt.subplots()
fig.set_size_inches(18, 15)
sns.set(style="whitegrid")
sns.lineplot(ax=ax,x='false_positive_rate',y='recall',data=df,hue='class',
             palette=colors,style='class',ci=None,size='class',sizes={'capsid':3,'tail':3})
plt.title('ROC curve', fontsize=27)
ax.tick_params(axis='y',labelsize=20)
ax.tick_params(axis='x',labelsize=20)

#for legobj in plt.legend().legendHandles:
#    legobj.set_linewidth(10.0)
#    legobj.set_linestyle('-')
#plt.setp(ax.get_legend().get_texts(), fontsize='27') # for legend text
plt.ylabel('Recall',fontsize='20')
plt.xlabel('False positive rate',fontsize='20')
ax.set(xlim=(0, 1))
ax.set(ylim=(0, 1))
handles, labels = ax.get_legend_handles_labels()
ax.set_aspect('equal')
plt.legend(handles[1:],labels[1:],handlelength=2,fontsize=27,markerfirst=False,handletextpad=0.1,
           loc='upper right',bbox_to_anchor=(1.34, 1))

# %%
print("ROC area under curve")
for class_name in ['capsid','tail']:
    tmp_df=df[df['class']==class_name]
    print(class_name)
    print(numpy.around(numpy.trapz(numpy.flip(tmp_df['recall']),x=numpy.flip(tmp_df['false_positive_rate'])),decimals=3))

# %%
df2[df2['threshold']==0]

# %%
df[[numpy.isclose(x,0,atol=0.1) for x in df['threshold']]]
