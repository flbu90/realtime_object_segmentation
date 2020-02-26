
'''
Trainingsdaten: Voxilized ShapeNet40 Dataset
'''
import tensorflow as tf
import os
import cv2
import random
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Visualize the test data
import scipy.io as io
import scipy.ndimage as nd

# ---------------------------------------------- Visualize the Data ----------------------------------------------------

training_voxels = numpy.load("3D_Test_Data/training/bottle/bottle_000000021_1.npy")
training_voxels = numpy.pad(training_voxels, (1, 1), 'constant', constant_values=(0, 0)) # macht aus 30x30x30 einen 32x32x32 vox
training_voxels = training_voxels.astype(numpy.float32)

test_voxels = numpy.load("3D_Test_Data/test/cup/cup_000000022_1.npy")
test_voxels = numpy.pad(test_voxels, (1, 1), 'constant', constant_values=(0, 0))  # macht aus 30x30x30 einen 32x32x32 vox
test_voxels = test_voxels.astype(numpy.float32)

#voxels = numpy.pad(voxels,(1,1), 'constant', constant_values=(0,0)) # aus 30x30x30 wird 32x32x32
#voxels = nd.zoom(voxels,(2,2,2), mode='constant', order=0) #zoom zu 64x64x64

# Trainingdata Example

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect('equal')
ax.voxels(training_voxels, edgecolor='red')
plt.savefig("Plotted_Train_Data.png")

# Testdata Example
fig = plt.figure()
bx = fig.gca(projection='3d')
bx.set_aspect('equal')
bx.voxels(test_voxels, edgecolor='red')
plt.savefig("Plotted_Test_Data.png")


# ---------------------------------------------- Import the Dataset ----------------------------------------------------


training_data = []
test_data = []

def create_trainig_data():

# Trainingsdata

    DATADIR = "3D_Test_Data/training"
    CATEGORIES = ["bottle", "cup"]

    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            img_array = os.path.join(path, img)
            voxel = numpy.load(img_array)
            voxel = numpy.pad(voxel, (1, 1), 'constant', constant_values=(0, 0)) # macht aus 30x30x30 einen 32x32x32 vox
            voxel = voxel.astype(numpy.float32)
            training_data.append([voxel, class_num])  # numpy.expand_dims(class_num, axis=0)

create_trainig_data()

def create_test_data():
    DATADIR = "3D_Test_Data/test"
    CATEGORIES = ["bottle", "cup"]

    for categoriy in CATEGORIES:
        path= os.path.join(DATADIR, categoriy)
        class_num = CATEGORIES.index(categoriy)
        for img in os.listdir(path):
            img_array = os.path.join(path, img)
            voxel = numpy.load(img_array)
            voxel = numpy.pad(voxel, (1, 1), 'constant', constant_values=(0, 0))
            voxel = voxel.astype(numpy.float32)  # Konvertieren von bool Array zu float array
            test_data.append([voxel, class_num])  # numpy.expand_dims(class_num, axis=0)

create_test_data()


# ----------------------------------------------- Prepare the Data ----------------------------------------------------


print("Lenght of Training Data:", len(training_data))
print("Lenght of Test Data:", len(test_data))

random.shuffle(training_data) # Netzwerk würde nicht gut lernen, wenn die trainig Daten sortiert sind

# Training Data
x_train_data = []  # Images
y_train_data = []  # Labels

for features, labels in training_data:
    x_train_data.append(features)
    y_train_data.append(labels)

print("First training data Sample", x_train_data[1])
print("Shape of Training Data x: ", x_train_data[0].shape)

#Test Data
x_test_data = []  # Images
y_test_data = []  # Labels

for features, labels in test_data:
    x_test_data.append(features)
    y_test_data.append(labels)

print("Shape of Test Data x:", x_test_data[0].shape)


# Test Data
#x_test_data = tf.placeholder('float')   # Images
#y_test_data = tf.placeholder('float')   # Labels


# ---------------------------------------------- Inputs --------------------------------------------------------------

graph = tf.Graph()

with graph.as_default():

    x = tf.placeholder('float')
    y = tf.placeholder('float')

    IMG_SIZE_X = 32  # eine Var reicht dient nur zum besseren Verständnis
    IMG_SIZE_Y = 32
    IMG_SIZE_Z = 32

    n_classes = 2
    batch_size = 2
    keep_rate = 0.8
    keep_prob = tf.placeholder(tf.float32)


# ------------------------------------------------- CNN Model ---------------------------------------------------------


    def conv3d(x,W):
        return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')

    def maxpool3d(x):
        #                          size of window         movement of window   in 3D kommt einfach eine Dimension dazu
        return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

    def convolutional_neural_network(x):

        weights = {'W_conv1': tf.Variable(tf.random_normal([3, 3, 3, 1, 32])), #3x3x3 Convolution, 1 Input und 32 Features/Outputs
                   'W_conv2': tf.Variable(tf.random_normal([3, 3, 3, 32, 64])),# 3x3x3 Convolution, 32 Inputs(von conv1) und 64 Outputs
                   'W_fc': tf.Variable(tf.random_normal([32768, 1024])),
                   'out': tf.Variable(tf.random_normal([1024, n_classes]))}
        '''
        Berechnung aus MNIST Pro Tut: IMG= 32*32*32 - nach maxpool3d mit ksize= 2*2*2 = 16*16*16
                                      IMG= 16*16*16 - nach dem zweiten Durchlauf: 8*8*8
                                      Je nachdem wie groß die Anzahl der Features ist berechnet sich der Wert : 8*8*8*64 = 
        '''
        biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
                  'b_conv2': tf.Variable(tf.random_normal([64])),
                  'b_fc': tf.Variable(tf.random_normal([1024])),
                  'out': tf.Variable(tf.random_normal([n_classes]))}

        # Reshape 'x' to a 4D tensor (2nd dim=image width, 3rd dim=image height, 4th dim=nColorChannel)
        x = tf.reshape(x, shape=[-1, IMG_SIZE_X, IMG_SIZE_Y, IMG_SIZE_Z, 1])

        conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
        conv1 = maxpool3d(conv1)

        conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
        conv2 = maxpool3d(conv2)

        fc = tf.reshape(conv2, [-1, 32768])
        fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
        fc = tf.nn.dropout(fc, keep_rate) #mimicing dead neurons, 80% of neurons will be kept (keep_rate = 0.8) helps fighting local maxima

        output = tf.matmul(fc, weights['out']) + biases['out']

        return output


# ----------------------------------------------TRAINING & TESTING-----------------------------------------------------


    def train_neural_network(x):

        NN_name = "3DCNN_Model"


        prediction = convolutional_neural_network(x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

        epochs = 1

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            iterations = int(len(x_train_data) / batch_size)
            import datetime
            start_time = datetime.datetime.now()

            for epoch in range(epochs):
                epoch_loss = 0
                start_time_epoch = datetime.datetime.now()
                #print('Epoch', epoch, 'started', end='')

                for itr in range(iterations):
                    mini_batch_x = x_train_data[itr * batch_size: (itr + 1) * batch_size]
                    mini_batch_y = y_train_data[itr * batch_size: (itr + 1) * batch_size]
                    _optimizer, _cost = sess.run([optimizer, cost], feed_dict={x: mini_batch_x, y: mini_batch_y})
                    epoch_loss += _cost

                print('Epoch', epoch + 1, 'complete out of', epochs, 'loss', epoch_loss.round(2))

                print("predictions:", prediction)
                correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y))
                accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

                # ----TESTING----
                acc = 0
                itrs = int(len(x_test_data) / batch_size)
                for itr in range(itrs):
                    mini_batch_x_test = x_test_data[itr * batch_size: (itr + 1) * batch_size]
                    mini_batch_y_test = y_test_data[itr * batch_size: (itr + 1) * batch_size]
                    acc += sess.run(accuracy, feed_dict={x: mini_batch_x_test, y: mini_batch_y_test})
                print("predictions:", prediction)

                end_time_epoch = datetime.datetime.now()
                print(' Testing Set Accuracy:', (acc / itrs).round(2), ' Time elapse: ', str(end_time_epoch - start_time_epoch))

            end_time = datetime.datetime.now()
            print('Time elapse: ', str(end_time - start_time))

            # Save the Model

            file_path = './' + NN_name + '/'
            if not os.path.exists(file_path):
                os.mkdir(file_path)
            saver = tf.train.Saver()
            saver.save(sess, file_path + 'model.checkpoint')
            print('Model saved')

    train_neural_network(x)
# ------------------------------------------- TESTING ON SINGLE IMAGE ------------------------------------------------



'''
test_voxels = numpy.load("Full_Testdata/cup/30/test/cup_000000315_1.npy")
#test_voxels = np.pad(test_voxels,(1,1), 'constant', constant_values=(0,0)) # aus 30x30x30 wird 32x32x32

#test_voxels = test_voxels.astype(int)

print ("Shape:", test_voxels.shape)

print ("Sample", test_voxels)

NN_name= "3DCNN_Model"
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    file_path = NN_name + '/'
    saver = tf.train.Saver()
    saver.restore(sess, file_path+ 'model.checkpoint')
    print('Model loaded')

    pred = convolutional_neural_network(x)

    test_prediction = sess.run(pred, feed_dict={test_voxels})

    #_, c, p = sess.run([pred], feed_dict={test_voxels})
    correct_prediction = tf.equal(tf.argmax(pred))

'''







# ------------------------------------------- OLD / Backup ------------------------------------------------------------

''' 

     #print('Epoch', epoch + 1, 'complete out of', epochs, 'loss', epoch_loss)
            #correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            #correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y))
            #accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            #print('Accuracy:', accuracy.eval)

    TRAING CODE B
    import datetime
    start_time = datetime.datetime.now()

    iterations = int(len(x_train_data) / batch_size) + 1

    # run epochs
    for epoch in range(epochs):

        start_time_epoch = datetime.datetime.now()
        print('Epoch', epoch, 'started', end='')
        epoch_loss = 0

        for itr in range(iterations):
            mini_batch_x = x_train_data[itr * batch_size: (itr + 1) * batch_size]
            mini_batch_y = y_train_data[itr * batch_size: (itr + 1) * batch_size]
            _optimizer, _cost = sess.run([optimizer, cost], feed_dict={x: mini_batch_x, y: mini_batch_y})
            epoch_loss += _cost

        # ----TESTING----
        # acc = 0
        # itrs = int(len(x_test_data) / batch_size) + 1
        # for itr in range(itrs):
        #    mini_batch_x_test = x_test_data[itr * batch_size: (itr + 1) * batch_size]
        #    mini_batch_y_test = y_test_data[itr * batch_size: (itr + 1) * batch_size]
        #    acc += sess.run(accuracy, feed_dict={x: mini_batch_x_test, y: mini_batch_y_test})

        # end_time_epoch = datetime.datetime.now()
        # print(' Testing Set Accuracy:', acc / itrs, ' Time elapse: ', str(end_time_epoch - start_time_epoch))

    end_time = datetime.datetime.now()
    print('Time elapse: ', str(end_time - start_time))
    '''
