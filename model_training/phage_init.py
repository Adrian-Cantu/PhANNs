import os
path = os.path.abspath(__file__)
root_dir = os.path.dirname(path)
fasta_dir = os.path.join(root_dir, 'fasta')
#model_dir = os.path.join(root_dir, 'model')
#model_dir = os.path.join(root_dir, 'model_v2')
#data_dir =  os.path.join(root_dir, 'data')
#data_dir =  os.path.join(root_dir, 'data_v2')
#model_dir = os.path.join(root_dir, 'model_v3')
#data_dir =  os.path.join(root_dir, 'data_v3')

model_dir = os.path.join(root_dir, 'model_v3.3')
data_dir =  os.path.join(root_dir, 'data_v3.2')

kfold_dir  = os.path.join(root_dir,'k_fold_model')
figures_dir  = os.path.join(root_dir,'figures')
data_dir_2 = os.path.join(root_dir, 'data2')
