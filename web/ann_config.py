import os
import pickle
from keras.models import load_model
import tensorflow as tf

path = os.path.abspath(__file__)
root_dir = os.path.dirname(path)
fasta_dir = os.path.join(root_dir, 'fasta')
model_dir = os.path.join(root_dir, 'deca_model')

mean_arr=pickle.load(open( os.path.join(model_dir,"mean_part.p"), "rb" ))
std_arr=pickle.load(open( os.path.join(model_dir,"std_part.p"), "rb" ))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

#models=pickle.load(open( os.path.join(model_dir,"deca_model.p"), "rb" ))
models=pickle.load(open( os.path.join(model_dir,"single.p"), "rb" ))
graph = tf.get_default_graph()
#web_test_X=pickle.load(open( os.path.join(model_dir,"web_test_X.p"), "rb" ))
n_members = 10
#models = list()
#yhats = numpy.empty((test_X.shape[0],10,11), dtype=numpy.float)
#for model_number in range(n_members):
    # load model
#        print('loading ...' + os.path.join(model_dir,'tetra_sc_tri_p_'+"{:02d}".format(model_number+1)+'.h5'))
#        model =  load_model( os.path.join(model_dir,'tetra_sc_tri_p_'+"{:02d}".format(model_number+1)+'.h5') )
    # store in memory
#        models.append(model)

