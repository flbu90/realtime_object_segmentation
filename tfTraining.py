import tensorflow as tf

import tfCNN_Model as convolutional_neural_network


def train_neural_network(x):

    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        import datetime
        start_time = datetime.datetime.now()

        iterations = int(len(x_train_data)/batch_size) + 1

        #run epochs
        for epoch in range(epochs):

            start_time_epoch = datetime.datetime.now()
            print('Epoch', epoch, 'started', end='')
            epoch_loss = 0

            for itr in range(iterations):

                mini_batch_x = x_train_data[itr*batch_size: (itr+1)*batch_size]
                mini_batch_y = y_train_data[itr*batch_size: (itr+1)*batch_size]
                _optimizer, _cost = sess.run([optimizer, cost], feed_dict={x: mini_batch_x, y: mini_batch_y})
                epoch_loss += _cost

            acc = 0
            itrs = int(len(x_test_data)/batch_size) + 1
            for itr in range(itrs):
                mini_batch_x_test = x_test_data[itr * batch_size: (itr + 1) * batch_size]
                mini_batch_y_test = y_test_data[itr * batch_size: (itr + 1) * batch_size]
                acc += sess.run(accuracy, feed_dict={x: mini_batch_x_test, y: mini_batch_y_test})

            end_time_epoch = datetime.datetime.now()
            print(' Testing Set Accuracy:', acc / itrs, ' Time elapse: ', str(end_time_epoch - start_time_epoch))

        end_time = datetime.datetime.now()
        print('Time elapse: ', str(end_time - start_time))

        #traning the network
        #for epoch in range(hm_epochs):
        #   start_time_epoch = datetime.datetime.now()
        #    print('Epoch', epoch, 'started', end='')
        #    epoch_loss = 0

            #for _ in range (int(mnist.train.num_examples / batch_size)): #How many times to cycle
            #    epoch_x, epoch_y = mnist.train.next_batch(batch_size)
            #    _ , c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y}) #c =cost
            #    epoch_loss += c

            #print('Epoch', epoch, 'complete out of', hm_epochs, 'loss', epoch_loss)

train_neural_network(x)