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

# %%

# %%
test_predictions=pd.read_csv('cat_tag_train_loss.csv',index_col=0)


# %%
test_predictions

# %%
class_list=['Major capsid', 'Baseplate', 'Major tail','Minor tail',
             'Portal','Tail fiber','Tail shaft','Collar','HTJ','Other']


# %%
def class_scores(tt,dataframe,prot_class,df):
    class_dic={'Major capsid' : 'major_capsid','Minor capsid':'minor_capsid','Baseplate':'baseplate',
               'Major tail':'major_tail','Minor tail':'minor_tail','Portal':'portal',
               'Tail fiber':'tail_fiber','Tail shaft':'shaft','Collar':'collar',
               'HTJ':'HTJ','Other':'other'}
    is_real=[ class_dic[prot_class] in x for x in dataframe.index.values]
    is_predicted=[x>=tt for x in dataframe[prot_class]]
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
    data_row=[prot_class,precision,recall,fscore,specificity,false_positive_rate,accuracy,tt]
    df=df.append(pd.Series(data_row,index=df.columns),sort=False,ignore_index=True)
    return df

    

# %%
d = {'class':[],'precision': [], 'recall': [],'f1-score':[],'specificity':[],
     'false_positive_rate':[],'accuracy':[],'threshold':[]}
df = pd.DataFrame(data=d)
score_range=numpy.arange(0,10.0001,0.1)
for class_name in class_list:
    #get only the entries where the specific class is predicted
    df_part=test_predictions[test_predictions.idxmax(axis=1)==class_name]
    print(class_name)
    for tt in score_range:
        tt=numpy.around(tt,decimals=2)
        print(tt,end="\r")
        df=class_scores(tt,df_part,class_name,df)
    print()

# %%
for class_name in class_list:
    df_part=test_predictions[test_predictions.idxmax(axis=1)==class_name]
    print(class_name)
    print(df_part.shape[0])

# %%
#with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#    print(kk_df[kk_df['class']=='Minor capsid'])
#kk_df.dtypes


# %%
import matplotlib.pyplot as plt
import matplotlib as mpl

# %%

#colors=["#69ef7b", "#b70d61", "#60e9dc", "#473c85", "#b4d170", "#104b6d", "#b4dbe7", "#1c5f1e", "#fd92fa", "#36a620", "#a834bf", "#fda547"]
colors=["#69ef7b", "#b70d61", "#60e9dc", "#473c85", "#b4d170", "#104b6d", "#b4dbe7", "#1c5f1e", "#fd92fa", "#36a620"]

customPalette = sns.color_palette(colors)

# %%
l_size=3
dash=[[2.8, 1.2] ,[7.8, 1.2, 1.8, 1.2], [1.1, 1.1],'' ,[2.8, 1.2] ,[7.8, 1.2, 1.8, 1.2], [1.1, 1.1],'' ,[2.8, 1.2] ,[7.8, 1.2, 1.8, 1.2],'']
l_size_l=[l_size,l_size,l_size,l_size,l_size,l_size,l_size,l_size,l_size,l_size,l_size]
dash_d=dict(zip(class_list,dash))
size_d=dict(zip(class_list,l_size_l))
sns.set(style="whitegrid")

# %%
fig, ax = plt.subplots()
fig.set_size_inches(18, 15)

sns.lineplot(ax=ax,x='threshold',y='f1-score',data=df,hue='class',
             palette=colors,style='class',ci=None,size='class',sizes=size_d
            ,dashes=dash_d)
#plt.title('CM : ')
ax.tick_params(axis='y',labelsize=20)
ax.tick_params(axis='x',labelsize=20)
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[1:],labels[1:],handlelength=2,fontsize=27,markerfirst=False,handletextpad=0.1,
           loc='upper right',bbox_to_anchor=(1.34, 1)) # for legend text
plt.ylabel('F1-score',fontsize='20')
plt.xlabel('Threshold',fontsize='20')
ax.set(xlim=(0,10))
ax.set(ylim=(0,1))
ax.set_aspect(10)
#fig.savefig('F1_vs_tt',bbox_inches="tight")

# %%
fig, ax = plt.subplots()
fig.set_size_inches(18, 15)
sns.set(style="whitegrid")
sns.lineplot(ax=ax,x='false_positive_rate',y='recall',data=df,hue='class',
             palette=colors,style='class',ci=None,size='class',sizes=size_d
            ,dashes=dash_d)
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
           loc='upper right',bbox_to_anchor=(1.34, 1)) # for legend tex
#fig.savefig('ROC_curves',bbox_inches="tight")


# %%
print("ROC area under curve")
for class_name in class_list:
    tmp_df=df[df['class']==class_name]
    print(class_name)
    print(numpy.around(numpy.trapz(numpy.flip(tmp_df['recall']),x=numpy.flip(tmp_df['false_positive_rate'])),decimals=3))


# %%
fig, ax = plt.subplots()
fig.set_size_inches(18, 15)
sns.set(style="whitegrid")
sns.lineplot(ax=ax,x='threshold',y='accuracy',data=df,hue='class',
             palette=colors,style='class',ci=None,size='class',sizes=size_d
            ,dashes=dash_d)
ax.tick_params(axis='y',labelsize=20)
ax.tick_params(axis='x',labelsize=20)
plt.setp(ax.get_legend().get_texts(), fontsize='27') # for legend text
plt.ylabel('Accuracy',fontsize='20')
plt.xlabel('Threshold',fontsize='20')
plt.legend(handles[1:],labels[1:],handlelength=2,fontsize=27,markerfirst=False,handletextpad=0.1,
           loc='upper right',bbox_to_anchor=(1.34, 1)) # for legend tex
ax.set(xlim=(0, 10))
ax.set(ylim=(0, 1))
ax.set_aspect(10)
#fig.savefig('acc_vs_tt',bbox_inches="tight")

# %%
fig, ax = plt.subplots()
fig.set_size_inches(18, 15)
sns.set(style="whitegrid")
sns.lineplot(ax=ax,x='threshold',y='precision',data=df,hue='class',
             palette=colors,style='class',ci=None,size='class',sizes=size_d
            ,dashes=dash_d,hue_order=class_list)
ax.tick_params(axis='y',labelsize=20)
ax.tick_params(axis='x',labelsize=20)
plt.setp(ax.get_legend().get_texts(), fontsize='27') # for legend text
plt.ylabel('Precision',fontsize='20')
plt.xlabel('Threshold',fontsize='20')
plt.legend(handles[1:],labels[1:],handlelength=2,fontsize=27,markerfirst=False,handletextpad=0.1,
           loc='upper right',bbox_to_anchor=(1.34, 1)) # for legend tex
ax.set(xlim=(0, 10))
ax.set(ylim=(0, 1))
ax.set_aspect(10)

# %%
float(df[(df['threshold']==7) & (df['class']=='Major capsid')]['precision'])



# %%
df.to_csv('test_set_stats.csv')


# %%
def print_full(x):
    pd.set_option('display.max_rows', len(x))
    display(x)
    pd.reset_option('display.max_rows')


# %%
print_full(df)
