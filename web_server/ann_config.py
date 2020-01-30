import os
import pickle
from tensorflow.keras.models import load_model
import tensorflow as tf

### chage the base path.
#prefix='/phanns'
prefix=''

path = os.path.abspath(__file__)
root_dir = os.path.dirname(path)
fasta_dir = os.path.join(root_dir, 'fasta')
model_dir = os.path.join(root_dir, 'deca_model')






#models=pickle.load(open( os.path.join(model_dir,"deca_model.p"), "rb" ))
#models=pickle.load(open( os.path.join(model_dir,"single.p"), "rb" ))
#graph = tf.get_default_graph()
#global graph
#graph = tf.compat.v1.get_default_graph()



