import os
import pickle
from tensorflow.keras.models import load_model
import tensorflow as tf
import ann_config


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
mean_arr=pickle.load(open( os.path.join(ann_config.model_dir,"mean_part.p"), "rb" ))
std_arr=pickle.load(open( os.path.join(ann_config.model_dir,"std_part.p"), "rb" ))
n_members = 10
models = list()
for model_number in range(n_members):
    # load model
        print('loading ...' + os.path.join(ann_config.model_dir,'tetra_sc_tri_p_'+"{:02d}".format(model_number+1)+'.h5'))
        model =  load_model( os.path.join(ann_config.model_dir,'tetra_sc_tri_p_'+"{:02d}".format(model_number+1)+'.h5') )
    # store in memory
        models.append(model)
