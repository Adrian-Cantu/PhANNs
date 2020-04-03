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
import numpy

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

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
all_models=['di_sc','di_sc_p','tri_sc','tri_sc_p','tetra_sc','tetra_sc_p','di','di_p','tri','tri_p','tetra_sc_tri_p','all']

# %%
#from collections import Counter
from sklearn.model_selection import StratifiedKFold
df = pd.DataFrame(data=d)
#def train_kfold(model_name,df):
for model_name in all_models:
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=43)
    cvscores = []
    model_number=0
    print("Doing cross validation on "+model_name)
    (train_X,train_Y)=ann_data.get_formated_train(model_name)
    train_Y_index = train_Y.argmax(axis=1)
    #f_num=train_X.shape[1]
    #rain_count=Counter(train_Y_index)
    #train_w_temp=[train_Y.shape[0]/train_count[i] for i in range(0,11,1)]
    #train_weights = dict(zip(range(0,11,1),train_w_temp) )
    for train, test in kfold.split(train_X, train_Y_index):
        model = LogisticRegression(class_weight='balanced')
        model_number=model_number+1
        print('\tModel '+ str(model_number))
        model.fit(train_X[train], train_Y_index[train])
        predicted_Y_index = model.predict(train_X[test])
        df=add_to_df(df,train_Y_index[test], predicted_Y_index ,model_name)
        del model


# %%
df

# %%
#pickle.dump(df, open( "kfold_df.p", "wb" ) )

# %%
avg_df=df[df['class'] == 'weighted avg']

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
ax.yaxis.grid(True)
#ax.set_xticklabels(['di','di_p','tri','tri_p','di_sc','di_sc_p','tri_sc','tri_sc_p','all'])
plt.show()
fig.savefig('logistic_avg_score_master.png',bbox_inches="tight")

# %%

# %%
(test_X,test_Y)=ann_data.get_formated_test("tetra_sc_tri_p")

# %%
(train_X,train_Y)=ann_data.get_formated_train("tetra_sc_tri_p")

# %%
test_Y_index = test_Y.argmax(axis=1)
train_Y_index = train_Y.argmax(axis=1)

# %%
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

# %%
model.fit(train_X, train_Y_index)

# %%
model.score(test_X, test_Y_index)

# %%
predicted_Y_index = model.predict(test_X)

# %%
pickle.dump(predicted_Y_index, open( "CM_predicted_test_Y.p", "wb" ) )
pickle.dump(test_Y_index, open( "CM_test_Y.p", "wb" ) )

# %%
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_Y_index, predicted_Y_index)
cm

# %%
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

# %%
from sklearn.metrics import classification_report
labels_names=["Major capsid","Minor capsid","Baseplate","Major tail","Minor tail","Portal","Tail fiber",
             "Tail shaft","Collar","Head-Tail joining","Others"]
print(classification_report(test_Y_index, predicted_Y_index, target_names=labels_names ))

# %%
from collections import Counter
zz=Counter(test_Y_index)
sample_w=[zz[i] for i in range(0,11,1)]
print(zz)
print(sample_w)
print()

# %%
sns.reset_defaults()

# %%
current_palette = sns.color_palette()
sns.palplot(current_palette)

# %%
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
sns.set(style='white')
plt.show()
CM=confusion_matrix(test_Y_index, predicted_Y_index)
CM_n=CM/numpy.array(sample_w)[:,None]
scale_up=1.4
plt.figure(figsize=[6.4*scale_up, 4.8*scale_up])
plt.imshow(CM_n, interpolation='nearest',cmap=plt.get_cmap('viridis'))
plt.title('CM : Logistic 80')
#plt.title('CM all')
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
plt.savefig('tetra_sc_tri_p_logistic_CM.png',bbox_inches="tight")
plt.show()
