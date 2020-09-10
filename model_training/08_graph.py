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
#ax.set_ylabel('')    
ax.set_xlabel('')
ax.set_ylabel('Score',fontsize='27') 
l = ax.legend()
plt.setp(ax.get_legend().get_texts(), fontsize='18') # for legend text
#print(dir(l))
ax.set(ylim=(0.4, 1))
    
#ax.set_xticklabels(['di','di_p','tri','tri_p','di_sc','di_sc_p','tri_sc','tri_sc_p','all'])
plt.show()
#fig.savefig('08_figures/avg_score_master.png',bbox_inches="tight")
fig.savefig(os.path.join('08_figures','avg_score_master',stop_time),bbox_inches="tight")

# %%
for kk in all_models:
    print(kk,end=' ')
    print(avg_df[(avg_df['model']==kk) & (avg_df['score_type']=='f1-score')]['value'].mean())

# %%
import seaborn as sns
import numpy
colors=["#69ef7b", "#b70d61", "#60e9dc", "#473c85", "#b4d170", "#104b6d", "#b4dbe7", "#1c5f1e", "#fd92fa", "#36a620", "#a834bf", "#fda547"]
customPalette = sns.color_palette(colors)

# %%
fig2, ax2 = plt.subplots()
fig2.set_size_inches(15, 15)

sns.set(style="whitegrid")
ax2.tick_params(axis='y',labelsize=30)
ax2.tick_params(axis='x',labelsize=35,rotation=80)
sns.barplot(ax=ax2,y="value", x="model", hue="class", data=f1_df, palette=colors)

ax2.set_ylabel('F1-score',fontsize='27')    
ax2.set_xlabel('')
l = ax2.legend()
plt.setp(ax2.get_legend().get_texts(), fontsize='27') # for legend text

handles2, labels2 = ax2.get_legend_handles_labels()

ax2.set(ylim=(0, 1))
ax2.set(xlim=(-0.5, 11.5))
plt.yticks(numpy.arange(0, 1.1, 0.1))
#ax2.set_xticklabels(['di','di_p','tri','tri_p','di_sc','di_sc_p','tri_sc','tri_sc_p','all'])
plt.legend(handles2,labels2,handlelength=2,fontsize=27,markerfirst=False,handletextpad=0.1,
           loc='upper right',bbox_to_anchor=(1.4, 1))
plt.show()
#fig2.savefig(os.path.join('08_figures','f1_score_master_per_model',stop_time),bbox_inches="tight")
#fig2.savefig('08_figures/f1_score_master_per_model.png',bbox_inches="tight")

# %%
#df[(df['score_type'] == 'f1-score') and ([any(x in ['di_sc','di_sc_p','tri_sc','tri_sc_p','tetra_sc','tetra_sc_p']) for x in df['model']] )]
#df[(df['score_type'] == 'f1-score') & (df['model']=='di_sc')]
df_sc_mod=df[(df['score_type'] == 'f1-score') & [x in ['di_sc','di_sc_p','tri_sc','tri_sc_p','tetra_sc','tetra_sc_p'] for x in df['model']]]

df_pp_mod=df[(df['score_type'] == 'f1-score') & [x not in ['di_sc','di_sc_p','tri_sc','tri_sc_p','tetra_sc','tetra_sc_p'] for x in df['model']]]

# %%
fig2, ax2 = plt.subplots()
fig2.set_size_inches(15, 15)

sns.set(style="whitegrid")
ax2.tick_params(axis='y',labelsize=30)
ax2.tick_params(axis='x',labelsize=35,rotation=80)
sns.barplot(ax=ax2,y="value", x="model", hue="class", data=df_sc_mod, palette=colors)

ax2.set_ylabel('F1-score',fontsize='27')    
ax2.set_xlabel('')
l = ax2.legend()
plt.setp(ax2.get_legend().get_texts(), fontsize='27') # for legend text

handles2, labels2 = ax2.get_legend_handles_labels()

ax2.set(ylim=(0, 1))
ax2.set(xlim=(-0.5, 5.5))
plt.yticks(numpy.arange(0, 1.1, 0.1))
#ax2.set_xticklabels(['di','di_p','tri','tri_p','di_sc','di_sc_p','tri_sc','tri_sc_p','all'])
plt.legend(handles2,labels2,handlelength=2,fontsize=27,markerfirst=False,handletextpad=0.1,
           loc='upper right',bbox_to_anchor=(1.4, 1))
plt.show()
fig2.savefig(os.path.join('08_figures','f1_score_master_per_model_sc'),bbox_inches="tight")
#fig2.savefig(os.path.join('08_figures','f1_score_master_per_model',stop_time),bbox_inches="tight")
#fig2.savefig('08_figures/f1_score_master_per_model.png',bbox_inches="tight")

# %%
fig2, ax2 = plt.subplots()
fig2.set_size_inches(15, 15)

sns.set(style="whitegrid")
ax2.tick_params(axis='y',labelsize=30)
ax2.tick_params(axis='x',labelsize=35,rotation=80)
sns.barplot(ax=ax2,y="value", x="model", hue="class", data=df_pp_mod, palette=colors)

ax2.set_ylabel('F1-score',fontsize='27')    
ax2.set_xlabel('')
l = ax2.legend()
plt.setp(ax2.get_legend().get_texts(), fontsize='27') # for legend text

handles2, labels2 = ax2.get_legend_handles_labels()

ax2.set(ylim=(0, 1))
#ax2.set(xlim=(-0.5, 5.5))
plt.yticks(numpy.arange(0, 1.1, 0.1))
#ax2.set_xticklabels(['di','di_p','tri','tri_p','di_sc','di_sc_p','tri_sc','tri_sc_p','all'])
plt.legend(handles2,labels2,handlelength=2,fontsize=27,markerfirst=False,handletextpad=0.1,
           loc='upper right',bbox_to_anchor=(1.4, 1))
plt.show()
fig2.savefig(os.path.join('08_figures','f1_score_master_per_model_pp'),bbox_inches="tight")
#fig2.savefig(os.path.join('08_figures','f1_score_master_per_model',stop_time),bbox_inches="tight")
#fig2.savefig('08_figures/f1_score_master_per_model.png',bbox_inches="tight")

# %%
fig2, ax2 = plt.subplots()
fig2.set_size_inches(15, 15)

sns.set(style="whitegrid")
ax2.tick_params(axis='y',labelsize=30)
ax2.tick_params(axis='x',labelsize=35,rotation=80)
sns.barplot(ax=ax2,y="value", x="class", hue="model", data=df_sc_mod, palette=colors)

ax2.set_ylabel('F1-score',fontsize='27')    
ax2.set_xlabel('')
l = ax2.legend()
plt.setp(ax2.get_legend().get_texts(), fontsize='27') # for legend text

handles2, labels2 = ax2.get_legend_handles_labels()

ax2.set(ylim=(0, 1))
#ax2.set(xlim=(-0.5, 5.5))
plt.yticks(numpy.arange(0, 1.1, 0.1))
#ax2.set_xticklabels(['di','di_p','tri','tri_p','di_sc','di_sc_p','tri_sc','tri_sc_p','all'])
plt.legend(handles2,labels2,handlelength=2,fontsize=27,markerfirst=False,handletextpad=0.1,
           loc='upper right',bbox_to_anchor=(1.33, 1))
fig2.savefig(os.path.join('08_figures','f1_score_master_per_class_sc'),bbox_inches="tight")

plt.show()

# %%
fig2, ax2 = plt.subplots()
fig2.set_size_inches(15, 15)

sns.set(style="whitegrid")
ax2.tick_params(axis='y',labelsize=30)
ax2.tick_params(axis='x',labelsize=35,rotation=80)
sns.barplot(ax=ax2,y="value", x="class", hue="model", data=df_pp_mod, palette=colors)

ax2.set_ylabel('F1-score',fontsize='27')    
ax2.set_xlabel('')
l = ax2.legend()
plt.setp(ax2.get_legend().get_texts(), fontsize='27') # for legend text

handles2, labels2 = ax2.get_legend_handles_labels()

ax2.set(ylim=(0, 1))
#ax2.set(xlim=(-0.5, 5.5))
plt.yticks(numpy.arange(0, 1.1, 0.1))
#ax2.set_xticklabels(['di','di_p','tri','tri_p','di_sc','di_sc_p','tri_sc','tri_sc_p','all'])
plt.legend(handles2,labels2,handlelength=2,fontsize=27,markerfirst=False,handletextpad=0.1,
           loc='upper right',bbox_to_anchor=(1.33, 1))
fig2.savefig(os.path.join('08_figures','f1_score_master_per_class_pp'),bbox_inches="tight")

plt.show()

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
fig3.savefig(os.path.join('08_figures','f1_score_master_per_class',stop_time),bbox_inches="tight")
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
pickle.dump(test_Y, open( os.path.join('08_figures','CM',"CM_test_Y_{}.p".format(stop_time)), "wb" ) )

# %%
#predicted_Y_index=pickle.load(open( os.path.join('08_figures','CM',"CM_predicted_test_Y_index_{}.p".format(stop_time)), "rb" ))
#predicted_Y=pickle.load(open( os.path.join('08_figures','CM',"CM_predicted_test_Y_{}.p".format(stop_time)), "rb" ) )


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
plt.savefig(os.path.join('08_figures','CM',stop_time),bbox_inches="tight")
plt.show()


# %%
kk=classification_report(test_Y, predicted_Y_index, target_names=labels_names, output_dict=True )


# %%
def class_scores(tt,scores,is_real,prot_class,df):
    class_dic={'Major capsid' : 'major_capsid','Minor capsid':'minor_capsid','Baseplate':'baseplate',
               'Major tail':'major_tail','Minor tail':'minor_tail','Portal':'portal',
               'Tail fiber':'tail_fiber','Tail shaft':'shaft','Collar':'collar',
               'HTJ':'HTJ','Other':'other'}
#    is_real=[ class_dic[prot_class] in x for x in dataframe.index.values]
    is_predicted=[x>=tt-0.05 for x in scores]
    TP=sum(numpy.logical_and(is_real,is_predicted))
    FN=sum(numpy.logical_and(is_real,numpy.logical_not(is_predicted)))
    TN=sum(numpy.logical_and(numpy.logical_not(is_real),numpy.logical_not(is_predicted)))
    FP=sum(numpy.logical_and(numpy.logical_not(is_real),is_predicted))
    #is_real_negative=len(dataframe.index.values)-sum(is_real)
    #FP=is_real_negative-TN
    if not (TP+TN+FP+FN):
        return df
    num_pred=TP+FP
    if not num_pred:
        precision=0
    else:
        precision=TP/num_pred
    num_rec=(TP+FN)
    if not num_rec:
        recall=0
    else:
        recall=TP/num_rec
    try:
        specificity=TN/(TN+FP)
    except ZeroDivisionError:
        specificity=0
    false_positive_rate=FP/(FP+TN)
    fscore=(2*TP)/(2*TP+FP+FN)
    accuracy=(TP+TN)/(TP+TN+FP+FN)
    data_row=[prot_class,precision,recall,fscore,specificity,false_positive_rate,accuracy,tt]
    df=df.append(pd.Series(data_row,index=df.columns),sort=False,ignore_index=True)
    return df


# %%
#df score takes all proteins classified as some class
d = {'class':[],'precision': [], 'recall': [],'f1-score':[],'specificity':[],
     'false_positive_rate':[],'accuracy':[],'threshold':[]}
df_tt = pd.DataFrame(data=d)
score_range=numpy.arange(0,10.0001,0.1)
class_list=['Major capsid', 'Minor capsid', 'Baseplate', 'Major tail','Minor tail',
             'Portal','Tail fiber','Tail shaft','Collar','HTJ','Other']
for num in range(11):
    #labels_names
    #get only the entries where the specific class is predicted
    #df_part=test_predictions[test_predictions.idxmax(axis=1)==class_name]
    print(class_list[num])
    for tt in score_range:
        tt=numpy.around(tt,decimals=2)
        print(tt,end="\r")
        df_tt=class_scores(tt,predicted_Y[predicted_Y_index==num,num],test_Y[predicted_Y_index==num]==num,class_list[num],df_tt)
    print()

# %%
test_set_p=predicted_Y[predicted_Y_index==num,num]
test_set_t=test_Y[predicted_Y_index==num]==num
print(test_set_t.shape)
print(test_set_p[test_set_p>5].shape)

# %%
# -0.05 to deal with float rounjding error
#df test score takes only the proteins classified with a score or better
df_test_score = pd.DataFrame(data=d)
score_range=numpy.arange(0,10.1,0.1)
class_list=['Major capsid', 'Minor capsid', 'Baseplate', 'Major tail','Minor tail',
             'Portal','Tail fiber','Tail shaft','Collar','HTJ','Other']
for num in range(11):
    #labels_names
    #get only the entries where the specific class is predicted
    #df_part=test_predictions[test_predictions.idxmax(axis=1)==class_name]
    print(class_list[num])
    test_set_p=predicted_Y[predicted_Y_index==num,num]
    test_set_t=test_Y[predicted_Y_index==num]==num
    for tt in score_range:
        
        #tt=numpy.around(tt,decimals=2)
        #try:
        #    kk1=max(numpy.around(test_set_p[test_set_p>=tt-0.05],decimals=1))
        #except:
        #    kk1='-'
        #try:
        #    kk2=max(test_set_p[test_set_p>=tt-0.05])
        #except:
        #    kk2='-'
        print(numpy.around(tt,decimals=2),end="\r")
        #print("{} {} {}".format(tt,kk1,kk2))
        df_test_score=class_scores(tt,numpy.around(test_set_p[test_set_p>=tt-0.05],decimals=1)
                                                   ,test_set_t[test_set_p>=tt-0.05]
                                                   ,class_list[num],df_test_score)
    print()

# %%
class_list=['Major capsid', 'Minor capsid', 'Baseplate', 'Major tail','Minor tail',
             'Portal','Tail fiber','Tail shaft','Collar','HTJ','Other']
for prot_class in class_list:
    kk100=df_test_score[(df_test_score['threshold']==10) & (df_test_score['class']==prot_class)]
    kk99=df_test_score[(df_test_score['threshold']==9.9) & (df_test_score['class']==prot_class)]
    if kk100.empty:
        print(prot_class)
        data_row=[prot_class,float(kk99['precision']),0,0,0,0,0,10]
        df_test_score=df_test_score.append(pd.Series(data_row,index=df_test_score.columns),sort=False,ignore_index=True)
#with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
#    print(df_test_score[(df_test_score['threshold']>4) & (df_test_score['class']=='Baseplate')])

# %%
with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
    print(df_test_score[(df_test_score['threshold']>4) & (df_test_score['class']=='Minor capsid')])

# %%
kk99=df_test_score[(df_test_score['threshold']==9.9) & (df_test_score['class']==prot_class)]
float(kk99['precision'])

# %%
df_test_score.to_csv('test_set_stats.csv')

# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
colors=["#69ef7b", "#b70d61", "#60e9dc", "#473c85", "#b4d170", "#104b6d", "#b4dbe7", "#1c5f1e", "#fd92fa", "#36a620", "#a834bf"]
customPalette = sns.color_palette(colors)
l_size=3

dash=[[2.8, 1.2] ,[7.8, 1.2, 1.8, 1.2], [1.1, 1.1],'' ,[2.8, 1.2] ,[7.8, 1.2, 1.8, 1.2], [1.1, 1.1],'' ,[2.8, 1.2] ,[7.8, 1.2, 1.8, 1.2],'']
l_size_l=[l_size,l_size,l_size,l_size,l_size,l_size,l_size,l_size,l_size,l_size,l_size]
dash_d=dict(zip(class_list,dash))
size_d=dict(zip(class_list,l_size_l))
sns.set(style="whitegrid")

# %%
fig, ax = plt.subplots()
fig.set_size_inches(18, 15)
sns.set(style="whitegrid")
sns.lineplot(ax=ax,x='false_positive_rate',y='recall',data=df_tt,hue='class',
             palette=colors,style='class',ci=None,size='class',sizes=size_d
            ,dashes=dash_d)
#plt.title('ROC_old', fontsize=27)
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
fig.savefig(os.path.join('08_figures','ROC',stop_time),bbox_inches="tight")
#fig.savefig('ROC_curves',bbox_inches="tight")

# %%
print("ROC area under curve")
f = open(os.path.join('08_figures','ROC',stop_time+'_area.txt'), "w")
for class_name in class_list:
    tmp_df=df_tt[df_tt['class']==class_name]
    print(class_name)
    f.write(class_name+"\n")
    print(numpy.around(numpy.trapz(numpy.flip(tmp_df['recall']),x=numpy.flip(tmp_df['false_positive_rate'])),decimals=3))
    f.write(str(numpy.around(numpy.trapz(numpy.flip(tmp_df['recall']),x=numpy.flip(tmp_df['false_positive_rate'])),decimals=3))+"\n")
f.close()

# %%
df_tt[df_tt['threshold']==7]

# %%
fig, ax = plt.subplots()
fig.set_size_inches(18, 15)
sns.set(style="whitegrid")
sns.lineplot(ax=ax,x='threshold',y='precision',data=df_test_score,hue='class',
             palette=colors,style='class',ci=None,size='class',sizes=size_d
            ,dashes=dash_d)
ax.tick_params(axis='y',labelsize=20)
ax.tick_params(axis='x',labelsize=20)
plt.setp(ax.get_legend().get_texts(), fontsize='27') # for legend text
plt.ylabel('Confidence',fontsize='20')
plt.xlabel('Threshold',fontsize='20')
plt.legend(handles[1:],labels[1:],handlelength=2,fontsize=27,markerfirst=False,handletextpad=0.1,
           loc='upper right',bbox_to_anchor=(1.34, 1)) # for legend tex
ax.set(xlim=(0, 10))
ax.set(ylim=(0, 1))
ax.set_aspect(10)

# %%
predicted_Y_index[predicted_Y.max(axis = 1) > 5].shape

# %%
labels_names=["Major capsid","Minor capsid","Baseplate","Major tail","Minor tail","Portal","Tail fiber",
                "Tail shaft","Collar","Head-Tail joining","Others"]
#exclude_d = {'precision': [], 'recall': [],'f1-score':[],'accuracy':[],'Portion excluded':[],'threshold':[]}
exclude_d = {'score_type':[],'value':[],'threshold':[]}
total_test =test_Y.shape[0]
df_exclude = pd.DataFrame(data=exclude_d)
score_range=numpy.arange(0,10.0001,0.1)
for score in score_range:
    include= predicted_Y.max(axis = 1) >= score
    c_report=classification_report(test_Y[include], predicted_Y_index[include],labels=[0,1,2,3,4,5,6,7,8,9,10], 
                                   target_names=labels_names, output_dict=True )
    print(score,end='\r')
#    data_row=[c_report['weighted avg']['precision'],c_report['weighted avg']['recall'],
#              c_report['weighted avg']['f1-score'],c_report['accuracy'],1-c_report['weighted avg']['support']/total_test,score]
    df_exclude=df_exclude.append(pd.Series(['Precision',c_report['weighted avg']['precision'],score],index=df_exclude.columns),sort=False,ignore_index=True)
#    df_exclude=df_exclude.append(pd.Series(['recall',c_report['weighted avg']['recall'],score],index=df_exclude.columns),sort=False,ignore_index=True)
    df_exclude=df_exclude.append(pd.Series(['F1-score',c_report['weighted avg']['f1-score'],score],index=df_exclude.columns),sort=False,ignore_index=True)
    try:
        df_exclude=df_exclude.append(pd.Series(['Accuracy',c_report['accuracy'],score],index=df_exclude.columns),sort=False,ignore_index=True)
    except:
        pass
    df_exclude=df_exclude.append(pd.Series(['Portion excluded',1-c_report['weighted avg']['support']/total_test,score],index=df_exclude.columns),sort=False,ignore_index=True)


# %%
fig, ax = plt.subplots()
fig.set_size_inches(5, 5)
sns.set(style="whitegrid")
sssize=3
sns.lineplot(ax=ax,x='threshold',y='value',data=df_exclude,size='score_type',hue='score_type',sizes=[3,3,3,3])
#             palette=colors,style='class',ci=None,size='class',sizes=size_d
#            ,dashes=dash_d)
ax.tick_params(axis='y',labelsize=12)
ax.tick_params(axis='x',labelsize=12)
#plt.setp(ax.get_legend().get_texts(), fontsize='27') # for legend text
plt.ylabel('',fontsize='12')
plt.xlabel('PhANNs Score',fontsize='12')
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[1:],labels[1:],handlelength=2,fontsize=12,markerfirst=False,handletextpad=0.1,
           loc='upper right',bbox_to_anchor=(1.55, 1)) # for legend tex
ax.set(xlim=(-0.3, 10.3))
ax.set(ylim=(-0.03, 1.03))
ax.set_aspect(10)

# %%
labels_names=["Major capsid","Minor capsid","Baseplate","Major tail","Minor tail","Portal","Tail fiber",
                "Tail shaft","Collar","Head-Tail joining","Others"]
#exclude_d = {'precision': [], 'recall': [],'f1-score':[],'accuracy':[],'Portion excluded':[],'threshold':[]}
exclude_d = {'score_type':[],'value':[],'threshold':[]}
total_test =test_Y.shape[0]
df_exclude2 = pd.DataFrame(data=exclude_d)
#score_range=numpy.arange(0,10.0001,0.1)
for score in [1,2,3,4,5,6,7,8,9,10]:
    include= predicted_Y.max(axis = 1) >= score
    c_report=classification_report(test_Y[include], predicted_Y_index[include],labels=[0,1,2,3,4,5,6,7,8,9,10], 
                                   target_names=labels_names, output_dict=True )
    print(score,end='\r')
#    data_row=[c_report['weighted avg']['precision'],c_report['weighted avg']['recall'],
#              c_report['weighted avg']['f1-score'],c_report['accuracy'],1-c_report['weighted avg']['support']/total_test,score]
    df_exclude2=df_exclude2.append(pd.Series(['Precision',c_report['weighted avg']['precision'],score],index=df_exclude2.columns),sort=False,ignore_index=True)
    df_exclude2=df_exclude2.append(pd.Series(['Recall',c_report['weighted avg']['recall'],score],index=df_exclude2.columns),sort=False,ignore_index=True)
    df_exclude2=df_exclude2.append(pd.Series(['F1-score',c_report['weighted avg']['f1-score'],score],index=df_exclude2.columns),sort=False,ignore_index=True)
#    try:
#        df_exclude=df_exclude.append(pd.Series(['Accuracy',c_report['accuracy'],score],index=df_exclude.columns),sort=False,ignore_index=True)
#    except:
#        pass
    df_exclude2=df_exclude2.append(pd.Series(['Portion included',c_report['weighted avg']['support']/total_test,score],index=df_exclude2.columns),sort=False,ignore_index=True)


# %%
fig, ax = plt.subplots()
ax.yaxis.grid(True)
#fig.set_size_inches(18, 15)
fig.set_size_inches(8, 6)
    
ax.tick_params(axis='y',labelsize=24)
ax.tick_params(axis='x',labelsize=24)
#ax.set_title('Model metrics : ' + df , fontsize=30,va='bottom')
    #ax.set_title('')
sns.barplot(ax=ax,y="value", x="threshold", hue="score_type", data=df_exclude2)
#ax.set_ylabel('')    
ax.set_xlabel('PhANNs Score',fontsize='20')
ax.set_ylabel('Sub-set Score',fontsize='20') 
l = ax.legend()
#plt.setp(ax.get_legend().get_texts(), fontsize='18') # for legend text
#print(dir(l))
ax.set(ylim=(0, 1))
    
ax.set_xticklabels(['1','2','3','4','5','6','7','8','9','10'])
handles3, labels3 = ax.get_legend_handles_labels()
plt.legend(handles3,labels3,handlelength=2,fontsize=18,markerfirst=False,handletextpad=0.1,
           loc='upper right',bbox_to_anchor=(1.5, 1))
plt.show()
fig.savefig('08_figures/exclude_by_score.png',bbox_inches="tight")
#fig.savefig(os.path.join('08_figures','avg_score_master',stop_time),bbox_inches="tight")

# %%
include= predicted_Y.max(axis = 1) >= 8
c_report=classification_report(test_Y[include], predicted_Y_index[include],labels=[0,1,2,3,4,5,6,7,8,9,10], 
                                   target_names=labels_names, output_dict=True )
c_report_8_nice=pd.DataFrame(c_report).transpose()
c_report_8_nice

# %%
print(test_Y.shape[0])
print(test_Y.shape[0] - sum(predicted_Y.max(axis = 1) >= 8))
print((test_Y.shape[0] - sum(predicted_Y.max(axis = 1) >= 8))/test_Y.shape[0])

# %%

c_report_t=classification_report(test_Y, predicted_Y_index,labels=[0,1,2,3,4,5,6,7,8,9,10], 
                                   target_names=labels_names, output_dict=True )
c_report_t_nice=pd.DataFrame(c_report_t).transpose()
c_report_t_nice
