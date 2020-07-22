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
#pos_tail=['baseplate','major_tail','minor_tail','tail_fiber','shaft','collar']
#neg_tail=['major_capsid','portal','HTJ','other']
#pos_capsid=['major_capsid']
#neg_capsid=['baseplate','major_tail','minor_tail','portal','tail_fiber','shaft','collar','HTJ','other']


# %%
pos_tail=['Baseplate','Major capsid','Minor tail','Tail fiber','Tail shaft','Collar']
neg_tail=['Major capsid','Portal','HTJ','Other']
pos_capsid=['Major capsid']
neg_capsid=['Baseplate','Major Tail','Minor tail','Portal','Tail fiber','Tail shaft','Collar','HTJ','Other']

# %%
class_dic={'Major capsid' : 'major_capsid','Minor capsid':'minor_capsid','Baseplate':'baseplate',
               'Major tail':'major_tail','Minor tail':'minor_tail','Portal':'portal',
               'Tail fiber':'tail_fiber','Tail shaft':'shaft','Collar':'collar',
               'HTJ':'HTJ','Other':'other'}

# %%
sum([any([class_dic[x]==y for y in pos_capsid]) for x in true_capsid.idxmax(1)])

# %%
sum(list(true_capsid[pos_tail].max(axis=1)>3))

# %%
true_capsid


# %%
def class_scores(dataframe_t,dataframe_f,poss_class,df):
    class_dic={'Major capsid' : 'major_capsid','Minor capsid':'minor_capsid','Baseplate':'baseplate',
               'Major tail':'major_tail','Minor tail':'minor_tail','Portal':'portal',
               'Tail fiber':'tail_fiber','Tail shaft':'shaft','Collar':'collar',
               'HTJ':'HTJ','Other':'other'}
    TP=sum([any([class_dic[x]==y for y in poss_class]) for x in dataframe_t.idxmax(1)])
    FN=dataframe_t.shape[0]-TP
    FP=sum([any([class_dic[x]==y for y in poss_class]) for x in dataframe_f.idxmax(1)])
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
    TP=sum(list(dataframe_t[pos_class].max(axis=1)>tt))
    FN=dataframe_t.shape[0]-TP
    FP=sum(list(dataframe_f[pos_class].max(axis=1)>tt))
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
d = {'class':[],'precision': [], 'recall': [],'f1-score':[],'specificity':[],
     'false_positive_rate':[],'accuracy':[],'threshold':[]}
df = pd.DataFrame(data=d)
score_range=numpy.arange(0,10.0001,0.1)
for tt in score_range:
    tt=numpy.around(tt,decimals=2)
    print(tt,end="\r")
    df=class_scores('capsid',true_capsid,false_capsid,pos_capsid,tt,df)
    print()

# %%
df = pd.DataFrame(data=d)
class_scores(true_capsid,false_capsid,pos_capsid,df)

# %%
class_scores(true_tail,false_tail,pos_tail,df)
