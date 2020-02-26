import cv2
import tensorflow as tf
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D # Visualize the test data
import scipy.io as io

# -------------------------------------------- Display the model.checkpoint -----------------------------------------


# import the inspect_checkpoint library
from tensorflow.python.tools import inspect_checkpoint as chkp

# print all tensors in checkpoint file
chkp.print_tensors_in_checkpoint_file("3DCNN_Model/model.checkpoint", tensor_name='', all_tensors=True)



# ---------------------------------------------- Display the Image ----------------------------------------------------
'''
test_voxels = np.load("Full_Testdata/cup/30/test/cup_000000315_1.npy")

test_voxels = np.pad(test_voxels,(1,1), 'constant', constant_values=(0,0)) # aus 30x30x30 wird 32x32x32
#voxels = nd.zoom(voxels,(2,2,2), mode='constant', order=0) #zoom zu 64x64x64


fig = plt.figure()
bx = fig.gca(projection='3d')
bx.set_aspect('equal')
bx.voxels(test_voxels, edgecolor='red')
plt.show(test_voxels)

'''