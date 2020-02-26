import tensorflow as tf

n_classes = 10
batch_size = 10  # Depents on Input Data

IMG_SIZE_X = 32  # eine Var reicht dient nur zur besseren Verständnis
IMG_SIZE_Y = 32
IMG_SIZE_Z = 32

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 32, 3])
y = tf.placeholder(tf.float32, shape=[None, n_classes])

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

def conv3d(x,W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')

def maxpool3d(x):
    #                          size of window         movement of window   in 3D kommt einfach eine Dimension dazu
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

def convolutional_neural_network(x):

    weights = {'W_conv1': tf.Variable(tf.random_normal([5,5,5,1,32])), #5x5x5 Convolution, 1 Input und 32 Features/Outputs
               'W_conv2': tf.Variable(tf.random_normal([5,5,5,32,64])),# 5x5x5 Convolution, 32 Inputs(von conv1) und 64 Outputs
               'W_fc': tf.Variable(tf.random_normal([32768,1024])),
               'out': tf.Variable(tf.random_normal([1024, n_classes])),}
    '''
    Berechnung aus MNIST Pro Tut: IMG= 32*32*32 - nach maxpool3d mit ksize= 2*2*2 = 16*16*16
                                  IMG= 16*16*16 - nach dem zweiten Durchlauf: 8*8*8
                                  Je nachdem wie groß die Anzahl der Features ist berechnet sich der Wert : 8*8*8*64 = 
    '''
    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([n_classes])),}

    x = tf.reshape(x, shape=[-1, IMG_SIZE_X, IMG_SIZE_Y, IMG_SIZE_Z, 1])  #reshape a flat 784 image to 32*32*32

    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool3d(conv1)

    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool3d(conv2)

    fc = tf.reshape(conv2,[-1, 32768])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate) #mimicing dead neurons, 80% of neurons will be kept (keep_rate = 0.8) helps fighting local maxima

    output = tf.matmul(fc, weights['out']) + biases['out']

    print("finished output returned")
    return output