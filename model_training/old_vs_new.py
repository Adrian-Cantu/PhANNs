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
test_predictions_new=pd.read_csv('cat_tag_val_loss.csv',index_col=0)
test_predictions_old=pd.read_csv('cat_tag_old.csv',index_col=0)

# %%
test_predictions_old

# %%
class_list_new=['Major capsid', 'Baseplate', 'Major tail','Minor tail',
             'Portal','Tail fiber','Tail shaft','Collar','HTJ','Other']
class_list_old=['Major capsid','Minor capsid', 'Baseplate', 'Major tail','Minor tail',
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

# %%
df_new = pd.DataFrame(data=d)
score_range=numpy.arange(0,10.0001,0.1)
for class_name in class_list_new:
    #get only the entries where the specific class is predicted
    df_part=test_predictions_new[test_predictions_new.idxmax(axis=1)==class_name]
    print(class_name)
    for tt in score_range:
        tt=numpy.around(tt,decimals=2)
        print(tt,end="\r")
        df_new=class_scores(tt,df_part,class_name,df_new)
    print()

# %%
df_old = pd.DataFrame(data=d)
score_range=numpy.arange(0,10.0001,0.1)
for class_name in class_list_old:
    #get only the entries where the specific class is predicted
    df_part=test_predictions_old[test_predictions_old.idxmax(axis=1)==class_name]
    print(class_name)
    for tt in score_range:
        tt=numpy.around(tt,decimals=2)
        print(tt,end="\r")
        df_old=class_scores(tt,df_part,class_name,df_old)
    print()

# %%
colors_old=["#69ef7b", "#b70d61", "#60e9dc", "#473c85", "#b4d170", "#104b6d", "#b4dbe7", "#1c5f1e", "#fd92fa", "#36a620", "#a834bf"]
colors_new=["#69ef7b", "#60e9dc", "#473c85", "#b4d170", "#104b6d", "#b4dbe7", "#1c5f1e", "#fd92fa", "#36a620", "#a834bf"]

#customPalette = sns.color_palette(colors)

# %%
l_size=3
dash_old=[[2.8, 1.2] ,[7.8, 1.2, 1.8, 1.2], [1.1, 1.1],'' ,[2.8, 1.2] ,[7.8, 1.2, 1.8, 1.2], [1.1, 1.1],'' ,[2.8, 1.2] ,[7.8, 1.2, 1.8, 1.2],'']
l_size_l_old=[l_size,l_size,l_size,l_size,l_size,l_size,l_size,l_size,l_size,l_size,l_size]
dash_d_old=dict(zip(class_list_old,dash_old))
size_d_old=dict(zip(class_list_old,l_size_l_old))

# %%
dash_new=[[2.8, 1.2] , [1.1, 1.1],'' ,[2.8, 1.2] ,[7.8, 1.2, 1.8, 1.2], [1.1, 1.1],'' ,[2.8, 1.2] ,[7.8, 1.2, 1.8, 1.2],'']
l_size_l_new=[l_size,l_size,l_size,l_size,l_size,l_size,l_size,l_size,l_size,l_size]
dash_d_new=dict(zip(class_list_new,dash_new))
size_d_new=dict(zip(class_list_new,l_size_l_new))

# %%
sns.set(style="whitegrid")

# %%
import matplotlib.pyplot as plt
import matplotlib as mpl

# %%
fig1, ax1 = plt.subplots()
fig1.set_size_inches(18, 15)
sns.set(style="whitegrid")
sns.lineplot(ax=ax1,x='false_positive_rate',y='recall',data=df_old,hue='class',
             palette=colors_old,style='class',ci=None,size='class',sizes=size_d_old
            ,dashes=dash_d_old)
#plt.title('ROC curve', fontsize=27)
ax1.tick_params(axis='y',labelsize=20)
ax1.tick_params(axis='x',labelsize=20)

#for legobj in plt.legend().legendHandles:
#    legobj.set_linewidth(10.0)
#    legobj.set_linestyle('-')
#plt.setp(ax.get_legend().get_texts(), fontsize='27') # for legend text
plt.ylabel('Recall',fontsize='20')
plt.xlabel('False positive rate',fontsize='20')
ax1.set(xlim=(0, 1))
ax1.set(ylim=(0, 1))
handles_old, labels_old = ax1.get_legend_handles_labels()
ax1.set_aspect('equal')
plt.legend(handles_old[1:],labels_old[1:],handlelength=2,fontsize=27,markerfirst=False,handletextpad=0.1,
           loc='upper right',bbox_to_anchor=(1.34, 1)) # for legend tex
#fig.savefig('ROC_curves',bbox_inches="tight")

# %%
fig2, ax2 = plt.subplots()
fig2.set_size_inches(18, 15)
sns.set(style="whitegrid")
sns.lineplot(ax=ax2,x='false_positive_rate',y='recall',data=df_new,hue='class',
             palette=colors_new,style='class',ci=None,size='class',sizes=size_d_new
            ,dashes=dash_d_new)
#plt.title('ROC curve', fontsize=27)
ax2.tick_params(axis='y',labelsize=20)
ax2.tick_params(axis='x',labelsize=20)

#for legobj in plt.legend().legendHandles:
#    legobj.set_linewidth(10.0)
#    legobj.set_linestyle('-')
#plt.setp(ax.get_legend().get_texts(), fontsize='27') # for legend text
plt.ylabel('Recall',fontsize='20')
plt.xlabel('False positive rate',fontsize='20')
ax2.set(xlim=(0, 1))
ax2.set(ylim=(0, 1))
handles, labels = ax2.get_legend_handles_labels()
ax2.set_aspect('equal')
plt.legend(handles[1:],labels[1:],handlelength=2,fontsize=27,markerfirst=False,handletextpad=0.1,
           loc='upper right',bbox_to_anchor=(1.34, 1)) # for legend tex
#fig.savefig('ROC_curves',bbox_inches="tight")

# %%
fig = plt.figure(figsize=(25, 25))
grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
roc_old  = fig.add_subplot(grid[0,0])
roc_new  = fig.add_subplot(grid[0,1])
conf_old = fig.add_subplot(grid[1,0])
conf_new = fig.add_subplot(grid[1,1])
#------------
sns.lineplot(ax=roc_old,x='false_positive_rate',y='recall',data=df_old,hue='class',
             palette=colors_old,style='class',ci=None,size='class',sizes=size_d_old
            ,dashes=dash_d_old,legend=False)
roc_old.set(xlim=(0, 1),ylim=(0, 1))
roc_old.annotate("A)", xy=(-0.1, 1.05), xycoords="axes fraction")
#-----------------------
sns.lineplot(ax=roc_new,x='false_positive_rate',y='recall',data=df_new,hue='class',
             palette=colors_new,style='class',ci=None,size='class',sizes=size_d_new
            ,dashes=dash_d_new,legend=False)
roc_new.set(xlim=(0, 1),ylim=(0, 1))
roc_new.annotate("B)", xy=(-0.1, 1.05), xycoords="axes fraction")
#------------
sns.lineplot(ax=conf_old,x='threshold',y='precision',data=df_old,hue='class',
             palette=colors_old,style='class',ci=None,size='class',sizes=size_d_old
            ,dashes=dash_d_old,hue_order=class_list_old,legend=False)
conf_old.set(xlim=(0, 10),ylim=(0, 1))
conf_old.set_xlabel('Score')
conf_old.set_ylabel('Confidence')
conf_old.annotate("C)", xy=(-0.1, 1.05), xycoords="axes fraction")
#------------
sns.lineplot(ax=conf_new,x='threshold',y='precision',data=df_new,hue='class',
             palette=colors_new,style='class',ci=None,size='class',sizes=size_d_new
            ,dashes=dash_d_new,hue_order=class_list_new,legend=False)
conf_new.set(xlim=(0, 10),ylim=(0, 1))
conf_new.set_xlabel('Score')
conf_new.set_ylabel('Confidence')
conf_new.annotate("d)", xy=(-0.1, 1.05), xycoords="axes fraction")
#------------
plt.legend(handles_old[1:],labels_old[1:],handlelength=2,fontsize=22,markerfirst=False,handletextpad=0.1,
           loc='upper right',bbox_to_anchor=(1.8, 2.2))
fig.savefig('new_vs_old',bbox_inches="tight")

# %%
import get_arr
from tensorflow.keras.models import load_model


# %%
def get_class_old(class_index):
        class_dic={'Major capsid' : 'major_capsid','Minor capsid':'minor_capsid','Baseplate':'baseplate',
               'Major tail':'major_tail','Minor tail':'minor_tail','Portal':'portal',
               'Tail fiber':'tail_fiber','Tail shaft':'shaft','Collar':'collar',
               'HTJ':'HTJ','Other':'other'}
        class_list_old=['Major capsid','Minor capsid', 'Baseplate', 'Major tail','Minor tail',
             'Portal','Tail fiber','Tail shaft','Collar','HTJ','Other']
        ret=0
        for prot_class in class_list_old:
            if class_dic[prot_class] in class_index:
                return ret
            else:
                ret=ret+1
        return ret


# %%
real_class_old=[get_class_old(x) for x in test_predictions_old.index.values]
predicted_class_old=numpy.argmax(test_predictions_old.to_numpy(), axis=1)

# %%
predicted_class_old

# %%
from sklearn.metrics import classification_report
labels_names=["Major capsid","Minor capsid","Baseplate","Major tail","Minor tail","Portal","Tail fiber",
             "Tail shaft","Collar","Head-Tail joining","Others"]
print(classification_report(real_class_old, predicted_class_old, target_names=labels_names ))
