import tensorflow as tf
import math
import time
import fractions
from data_source import Mnist_CNN_shards
import numpy as np
from keras.utils.np_utils import to_categorical


# HIDDEN_LAYERS = [200, 200]


def weight_variable(shape, name, wd):
    var = tf.get_variable(name=name, shape=shape, initializer=tf.glorot_uniform_initializer())
    # var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def bias_variable(shape, val):
    initial = tf.constant(val, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='VALID')




def fedavg(client_weights):
    new_weights = [np.zeros(w.shape) for w in client_weights[0]]
    for c in range(len(client_weights)):
        for i in range(len(new_weights)):#第零层5，5，32，64第一层5，5，1，32第二层3136, 128第三层（128，10）
            new_weights[i]+=client_weights[c][i]

    return new_weights




class federated_learning():
    NUM_CLASSES = 10
    IMAGE_SIZE = 28
    IMAGE_PIXELS = IMAGE_SIZE**2
    img_size, num_channels, num_classes = 28, 1, 10
    x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
    y_ = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_')
    W_conv1 = weight_variable([5, 5, 1, 32], name='conv_1', wd=None)
    b_conv1 = bias_variable([32], 0.)
    conv1 = tf.nn.conv2d(x, W_conv1, [1, 1, 1, 1], padding='SAME')
    h_conv1 = tf.nn.relu(conv1 + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64], name='conv_2', wd=None)
    b_conv2 = bias_variable([64], 0.)
    conv2 = tf.nn.conv2d(h_pool1, W_conv2, [1, 1, 1, 1], padding='SAME')
    h_conv2 = tf.nn.relu(conv2 + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

    W_fc1 = weight_variable([7 * 7 * 64, 128], name='fc_1', wd=None)
    b_fc1 = bias_variable([128], 0.)
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    W_fc2 = weight_variable([128, 10], name='fc_2', wd=None)
    b_fc2 = bias_variable([10], 0.)
    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))
    lr = tf.placeholder(tf.float32, shape=[])
    opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
    train_step = opt.minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    pw1 = tf.placeholder(tf.float32, [5, 5, 1, 32])
    opw1 = tf.assign(W_conv1, pw1)

    pb1 = tf.placeholder(tf.float32, [32])
    opb1 = tf.assign(b_conv1, pb1)

    pw2 = tf.placeholder(tf.float32, [5, 5, 32, 64])
    opw2 = tf.assign(W_conv2, pw2)

    pb2 = tf.placeholder(tf.float32, [64])
    opb2 = tf.assign(b_conv2, pb2)

    pw3 = tf.placeholder(tf.float32, [7*7*64, 128])
    opw3 = tf.assign(W_fc1, pw3)

    pb3 = tf.placeholder(tf.float32, [128])
    opb3 = tf.assign(b_fc1, pb3)

    pw4 = tf.placeholder(tf.float32, [128, 10])
    opw4 = tf.assign(W_fc2, pw4)

    pb4 = tf.placeholder(tf.float32, [10])
    opb4 = tf.assign(b_fc2, pb4)
    
    rounds = 100
    E = 5
    batch_size = 50
    C = 1.0
    K = 10
    init_lr = 0.1
    decay = 0.995
    total_batch = 12
    data = Mnist_CNN_shards() # input data
    images_train_split_nonIID, labels_train_split_nonIID = data.mnist_noniid(100)

    def start (self):
        print("初始化tensorflow")
        learning_rate = self.init_lr

        sess_nonIID = tf.Session()
        sess_nonIID.run(tf.global_variables_initializer())
        start_time = time.time()

        temp_weights = sess_nonIID.run([self.W_conv1, self.W_conv2, self.W_fc1, self.W_fc2])
        temp_biases = sess_nonIID.run([self.b_conv1, self.b_conv2, self.b_fc1, self.b_fc2])

        # empty weights and biases from clients
        clients_weights, clients_biases = [], []
        start_time=time.time()
        # use the same shared weights and biases for all clients in the current round
        print("开始学习")
        for client in range(100):
            for epoch in range(self.E):
                start, end = 0, self.batch_size

                for i in range(self.total_batch):
                    batch_x, batch_y = self.images_train_split_nonIID[client][start:end, :], self.labels_train_split_nonIID[client][
                                                                                        start:end, :]
                    start += self.batch_size
                    end += self.batch_size
                    sess_nonIID.run(self.train_step, feed_dict={self.x: batch_x.reshape(self.batch_size, 28, 28, 1),
                                                        self.y_: batch_y, self.lr: learning_rate})

            clients_weights.append(sess_nonIID.run([self.W_conv1, self.W_conv2, self.W_fc1, self.W_fc2]))
            clients_biases.append(sess_nonIID.run([self.b_conv1, self.b_conv2, self.b_fc1, self.b_fc2]))

            # reset shared weights and biases
            # all clients use the parameter in the previous round
            sess_nonIID.run(self.opw1, feed_dict={self.pw1: temp_weights[0]})  # pw1->W_conv1 / temp_weights->trained_weights
            sess_nonIID.run(self.opw2, feed_dict={self.pw2: temp_weights[1]})
            sess_nonIID.run(self.opw3, feed_dict={self.pw3: temp_weights[2]})
            sess_nonIID.run(self.opw4, feed_dict={self.pw4: temp_weights[3]})

            sess_nonIID.run(self.opb1, feed_dict={self.pb1: temp_biases[0]})
            sess_nonIID.run(self.opb2, feed_dict={self.pb2: temp_biases[1]})
            sess_nonIID.run(self.opb3, feed_dict={self.pb3: temp_biases[2]})
            sess_nonIID.run(self.opb4, feed_dict={self.pb4: temp_biases[3]})

        # compute new federated averaging(fedavg) weights and biases

        print("权重更新")
        #fedavg_weights = fedavg(clients_weights)
        #fedavg_biases = fedavg(clients_biases)

        end_time = time.time()
        print('Time: %.2fs' % (end_time - start_time))
        return (clients_weights,clients_biases)
    def start_demo (self):
        print("初始化tensorflow")
        learning_rate = self.init_lr

        sess_nonIID = tf.Session()
        sess_nonIID.run(tf.global_variables_initializer())
        start_time = time.time()

        temp_weights = sess_nonIID.run([self.W_conv1, self.W_conv2, self.W_fc1, self.W_fc2])
        temp_biases = sess_nonIID.run([self.b_conv1, self.b_conv2, self.b_fc1, self.b_fc2])

        # empty weights and biases from clients
        clients_weights, clients_biases = [], []
        start_time=time.time()
        # use the same shared weights and biases for all clients in the current round
        print("开始学习")
        for client in range(100):
            for epoch in range(self.E):
                start, end = 0, self.batch_size

                for i in range(self.total_batch):
                    batch_x, batch_y = self.images_train_split_nonIID[client][start:end, :], self.labels_train_split_nonIID[client][
                                                                                        start:end, :]
                    start += self.batch_size
                    end += self.batch_size
                    sess_nonIID.run(self.train_step, feed_dict={self.x: batch_x.reshape(self.batch_size, 28, 28, 1),
                                                        self.y_: batch_y, self.lr: learning_rate})

            clients_weights.append(sess_nonIID.run([self.W_conv1, self.W_conv2, self.W_fc1, self.W_fc2]))
            clients_biases.append(sess_nonIID.run([self.b_conv1, self.b_conv2, self.b_fc1, self.b_fc2]))

            # reset shared weights and biases
            # all clients use the parameter in the previous round
            sess_nonIID.run(self.opw1, feed_dict={self.pw1: temp_weights[0]})  # pw1->W_conv1 / temp_weights->trained_weights
            sess_nonIID.run(self.opw2, feed_dict={self.pw2: temp_weights[1]})
            sess_nonIID.run(self.opw3, feed_dict={self.pw3: temp_weights[2]})
            sess_nonIID.run(self.opw4, feed_dict={self.pw4: temp_weights[3]})

            sess_nonIID.run(self.opb1, feed_dict={self.pb1: temp_biases[0]})
            sess_nonIID.run(self.opb2, feed_dict={self.pb2: temp_biases[1]})
            sess_nonIID.run(self.opb3, feed_dict={self.pb3: temp_biases[2]})
            sess_nonIID.run(self.opb4, feed_dict={self.pb4: temp_biases[3]})

        # compute new federated averaging(fedavg) weights and biases

        print("权重更新")
        fedavg_weights = fedavg(clients_weights)
        fedavg_biases = fedavg(clients_biases)

        # update shared weights and biases with fedavg
        sess_nonIID.run(self.opw1, feed_dict={self.pw1: fedavg_weights[0]})
        sess_nonIID.run(self.opw2, feed_dict={self.pw2: fedavg_weights[1]})
        sess_nonIID.run(self.opw3, feed_dict={self.pw3: fedavg_weights[2]})
        sess_nonIID.run(self.opw4, feed_dict={self.pw4: fedavg_weights[3]})

        sess_nonIID.run(self.opb1, feed_dict={self.pb1: fedavg_biases[0]})
        sess_nonIID.run(self.opb2, feed_dict={self.pb2: fedavg_biases[1]})
        sess_nonIID.run(self.opb3, feed_dict={self.pb3: fedavg_biases[2]})
        sess_nonIID.run(self.opb4, feed_dict={self.pb4: fedavg_biases[3]})

        test_accuracy = sess_nonIID.run(self.accuracy, feed_dict={self.x: self.data.x_test,
                                                            self.y_: to_categorical(self.data.y_test, 10)})
        test_loss = sess_nonIID.run(self.cross_entropy, feed_dict={self.x: self.data.x_test,
                                                            self.y_: to_categorical(self.data.y_test, 10)})
        # fedavg_nonIID_accuracy.append(test_accuracy)
        # fedavg_nonIID_loss.append(test_loss)

        print(' loss={0}, accuracy={1}, lr={2}'.format( test_loss, test_accuracy, learning_rate))

        learning_rate *= self.decay
        self.learning_rate=learning_rate

        end_time = time.time()
        print('Time: %.2fs' % (end_time - start_time))
    def demo (self):#用于测试的可用方程
        learning_rate = self.init_lr

        sess_nonIID = tf.Session()
        sess_nonIID.run(tf.global_variables_initializer())
        start_time = time.time()
        for r in range(self.rounds):
        # randomly sielect m clients
        # m = max(C*K, 1)
        # s = np.random.choice(int(K), int(m), replace=False)

        # store shared weights and biases in the current round
            temp_weights = sess_nonIID.run([self.W_conv1, self.W_conv2, self.W_fc1, self.W_fc2])
            temp_biases = sess_nonIID.run([self.b_conv1, self.b_conv2, self.b_fc1, self.b_fc2])

            # empty weights and biases from clients
            clients_weights, clients_biases = [], []

            # use the same shared weights and biases for all clients in the current round
            for client in range(100):
                for epoch in range(self.E):
                    start, end = 0, self.batch_size

                    for i in range(self.total_batch):
                        batch_x, batch_y = self.images_train_split_nonIID[client][start:end, :], self.labels_train_split_nonIID[client][
                                                                                            start:end, :]
                        start += self.batch_size
                        end += self.batch_size
                        sess_nonIID.run(self.train_step, feed_dict={self.x: batch_x.reshape(self.batch_size, 28, 28, 1),
                                                            self.y_: batch_y, self.lr: learning_rate})

                clients_weights.append(sess_nonIID.run([self.W_conv1, self.W_conv2, self.W_fc1, self.W_fc2]))
                clients_biases.append(sess_nonIID.run([self.b_conv1, self.b_conv2, self.b_fc1, self.b_fc2]))

                # reset shared weights and biases
                # all clients use the parameter in the previous round
                sess_nonIID.run(self.opw1, feed_dict={self.pw1: temp_weights[0]})  # pw1->W_conv1 / temp_weights->trained_weights
                sess_nonIID.run(self.opw2, feed_dict={self.pw2: temp_weights[1]})
                sess_nonIID.run(self.opw3, feed_dict={self.pw3: temp_weights[2]})
                sess_nonIID.run(self.opw4, feed_dict={self.pw4: temp_weights[3]})

                sess_nonIID.run(self.opb1, feed_dict={self.pb1: temp_biases[0]})
                sess_nonIID.run(self.opb2, feed_dict={self.pb2: temp_biases[1]})
                sess_nonIID.run(self.opb3, feed_dict={self.pb3: temp_biases[2]})
                sess_nonIID.run(self.opb4, feed_dict={self.pb4: temp_biases[3]})

            # compute new federated averaging(fedavg) weights and biases


            fedavg_weights = fedavg(clients_weights)
            fedavg_biases = fedavg(clients_biases)

            # update shared weights and biases with fedavg
            sess_nonIID.run(self.opw1, feed_dict={self.pw1: fedavg_weights[0]})
            sess_nonIID.run(self.opw2, feed_dict={self.pw2: fedavg_weights[1]})
            sess_nonIID.run(self.opw3, feed_dict={self.pw3: fedavg_weights[2]})
            sess_nonIID.run(self.opw4, feed_dict={self.pw4: fedavg_weights[3]})

            sess_nonIID.run(self.opb1, feed_dict={self.pb1: fedavg_biases[0]})
            sess_nonIID.run(self.opb2, feed_dict={self.pb2: fedavg_biases[1]})
            sess_nonIID.run(self.opb3, feed_dict={self.pb3: fedavg_biases[2]})
            sess_nonIID.run(self.opb4, feed_dict={self.pb4: fedavg_biases[3]})

            test_accuracy = sess_nonIID.run(self.accuracy, feed_dict={self.x: self.data.x_test,
                                                                self.y_: to_categorical(self.data.y_test, 10)})
            test_loss = sess_nonIID.run(self.cross_entropy, feed_dict={self.x: self.data.x_test,
                                                                self.y_: to_categorical(self.data.y_test, 10)})
            # fedavg_nonIID_accuracy.append(test_accuracy)
            # fedavg_nonIID_loss.append(test_loss)

            if r % 1 == 0:
                print('Epoch {0}: loss={1}, accuracy={2}, lr={3}'.format(r, test_loss, test_accuracy, learning_rate))

            learning_rate *= self.decay
            self.learning_rate=learning_rate
        end_time = time.time()
        print('Time: %.2fs' % (end_time - start_time))


federated=federated_learning()
def main():
    temp=federated.start()
    print("hh")
    print("hh")
if __name__ == "__main__":
    main()
    