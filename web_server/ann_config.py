import os
import pickle
from tensorflow.keras.models import load_model
import tensorflow as tf

### chage the base path.
prefix='/phanns'
#prefix=''

path = os.path.abspath(__file__)
root_dir = os.path.dirname(path)
fasta_dir = os.path.join(root_dir, 'fasta')
model_dir = os.path.join(root_dir, 'deca_model')


mean_arr=pickle.load(open( os.path.join(model_dir,"mean_part.p"), "rb" ))
std_arr=pickle.load(open( os.path.join(model_dir,"std_part.p"), "rb" ))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

#models=pickle.load(open( os.path.join(model_dir,"deca_model.p"), "rb" ))
#models=pickle.load(open( os.path.join(model_dir,"single.p"), "rb" ))
#graph = tf.get_default_graph()
global graph
graph = tf.compat.v1.get_default_graph()


n_members = 10
models = list()
for model_number in range(n_members):
    # load model
        print('loading ...' + os.path.join(model_dir,'tetra_sc_tri_p_'+"{:02d}".format(model_number+1)+'.h5'))
        model =  load_model( os.path.join(model_dir,'tetra_sc_tri_p_'+"{:02d}".format(model_number+1)+'.h5') )
    # store in memory
        models.append(model)
