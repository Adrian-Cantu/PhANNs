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
#import sys
#sys.path.append("..")
#import phage_init

# %%
#import ann_data
import get_arr

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
group_arr=pickle.load(open( os.path.join('06_features',"group_arr.p"), "rb" ))
class_arr=pickle.load(open( os.path.join('06_features',"class_arr.p"), "rb" ))


# %%
#from matplotlib import pyplot
def train_kfold(model_name,df,class_arr,group_arr):
    print("Doing cross validation on "+model_name)
    for model_number in range(10):
        print("\t "+ str(model_number) )
        (train_X,train_Y_index)=get_arr.get_train(model_name,model_number,class_arr,group_arr)
        (test_X,  test_Y_index)=get_arr.get_validation(model_name,model_number,class_arr,group_arr)
        model = LogisticRegression(class_weight='balanced')
        #model_number=model_number+1
        #print("Doing cross validation on "+model_name)
        model.fit(train_X, train_Y_index)
        predicted_Y_index = model.predict(test_X)
        df=add_to_df(df,test_Y_index, predicted_Y_index ,model_name)
        pickle.dump(model, open( os.path.join('09_logistic_models','log_'+model_name+'_'+"{:02d}".format(model_number)+'.p'), "wb"  ))
        del model
        del train_X
        del test_X
    return df
#    return 1


# %%
df = pd.DataFrame(data=d)
for model_name in all_models:
    df=train_kfold(model_name,df,class_arr,group_arr)


# %%
df

# %%
pickle.dump(df, open( open( os.path.join('09_logistic_models','log_kfold_df.p'),'rb') )
#df=pickle.load(open( os.path.join('09_logistic_models','log_kfold_df.p'),'rb'))

# %%
df_un=pickle.load(open( os.path.join('07_models','all_results_df.p'),'rb'))
df40=pickle.load(open( os.path.join('09_logistic_models','40_derep_results_df.p'),'rb'))

# %%
my_df_dic= {'40':df40,'logistic':df,'undereplicate_val':df_un}

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
fig3.savefig(os.path.join('09_logistic_models','derep_per_model.png'),bbox_inches="tight")
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
fig4.savefig(os.path.join('09_logistic_models','derep_per_class.png'),bbox_inches="tight")
plt.show()
