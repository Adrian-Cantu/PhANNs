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
false_capsid=pd.read_csv('vp_FALSE_CAPSID.csv',index_col=0)
true_capsid=pd.read_csv('vp_TRUE_CAPSID.csv',index_col=0)
false_tail=pd.read_csv('vp_FALSE_TAIL.csv',index_col=0)
true_tail=pd.read_csv('vp_TRUE_TAIL.csv',index_col=0)

# %%

pos_tail=['Major tail','Minor tail','Tail fiber','Tail shaft']
neg_tail=['Major capsid','Portal','HTJ','Other']

pos_capsid=['Major capsid']
neg_capsid=['Major Tail','Minor tail','Portal','Tail fiber','Tail shaft','HTJ','Other']


# %%
def class_scores(dataframe_t,dataframe_f,poss_class,df):
    class_dic={'Major capsid' : 'major_capsid','Minor capsid':'minor_capsid','Baseplate':'baseplate',
               'Major tail':'major_tail','Minor tail':'minor_tail','Portal':'portal',
               'Tail fiber':'tail_fiber','Tail shaft':'shaft','Collar':'collar',
               'HTJ':'HTJ','Other':'other'}
    TP=sum([any([x==y for y in poss_class]) for x in dataframe_t.idxmax(1)])
    FN=dataframe_t.shape[0]-TP
    FP=sum([any([x==y for y in poss_class]) for x in dataframe_f.idxmax(1)])
    TN=dataframe_f.shape[0]-FP

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
    data_row=[precision,recall,fscore,specificity,false_positive_rate,accuracy]
    #data_row=[hue,precision,recall,fscore,specificity,false_positive_rate,accuracy,tt]
    df=df.append(pd.Series(data_row,index=df.columns),sort=False,ignore_index=True)
    return df


# %%
def class_scores2(prot_class,dataframe_t,dataframe_f,poss_class,tt,df):
    TP=sum(list(dataframe_t[poss_class].max(axis=1)>tt))
    FN=dataframe_t.shape[0]-TP
    FP=sum(list(dataframe_f[poss_class].max(axis=1)>tt))
    TN=dataframe_f.shape[0]-FP

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
    data_row=[prot_class,precision,recall,fscore,specificity,false_positive_rate,accuracy,tt]
    #data_row=[hue,precision,recall,fscore,specificity,false_positive_rate,accuracy,tt]
    df=df.append(pd.Series(data_row,index=df.columns),sort=False,ignore_index=True)
    return df


# %%
d = {'class':[],'precision': [], 'recall': [],'f1-score':[],'specificity':[],
     'false_positive_rate':[],'accuracy':[],'threshold':[]}
df = pd.DataFrame(data=d)
score_range=numpy.arange(0,10.0001,0.1)
kk1=[any([x==y for y in pos_capsid]) for x in true_capsid.idxmax(1)]
kk2=[any([x==y for y in pos_capsid]) for x in false_capsid.idxmax(1)]
part_true_capsid=true_capsid[kk1]
part_false_capsid=false_capsid[kk2]
for tt in score_range:
    tt=numpy.around(tt,decimals=2)
    print(tt,end="\r")
    df=class_scores2('capsid',part_true_capsid,part_false_capsid,pos_capsid,tt,df)
print()
kk3=[any([x==y for y in pos_tail]) for x in true_tail.idxmax(1)]
kk4=[any([x==y for y in pos_tail]) for x in false_tail.idxmax(1)]
part_true_tail=true_tail[kk3]
part_false_tail=false_tail[kk4]
for tt in score_range:
    tt=numpy.around(tt,decimals=2)
    print(tt,end="\r")
    df=class_scores2('tail',part_true_tail,part_false_tail,pos_tail,tt,df)

# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
#colors=["#69ef7b", "#b70d61", "#60e9dc", "#473c85", "#b4d170", "#104b6d", "#b4dbe7", "#1c5f1e", "#fd92fa", "#36a620", "#a834bf", "#fda547"]
colors=["#69ef7b", "#b70d61"]

customPalette = sns.color_palette(colors)
size_d={'capsid':3,'tail':3}

# %%
fig, ax = plt.subplots()
fig.set_size_inches(18, 15)
#sns.set(style="whitegrid")
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
dd = {'precision': [], 'recall': [],'f1-score':[],'specificity':[],
     'false_positive_rate':[],'accuracy':[]}
df2 = pd.DataFrame(data=dd)
class_scores(part_true_capsid,part_false_capsid,pos_capsid,df2)

# %%
df3 = pd.DataFrame(data=dd)
class_scores(part_true_tail,part_false_tail,pos_tail,df3)
