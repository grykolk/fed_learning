import tensorflow as tf
import math
import time
import fractions
from data_source import Mnist_shards
import numpy as np
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

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
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='VALID')




def fedavg(client_weights):
    new_weights = [np.zeros(w.shape) for w in client_weights[0]]
    for c in range(len(client_weights)):
        for i in range(len(new_weights)):
            new_weights[i]+=client_weights[c][i]

    return new_weights




class LeNet_5():
    print("初始化tensorflow")
    NUM_CLASSES = 10
    IMAGE_SIZE = 28
    IMAGE_PIXELS = IMAGE_SIZE**2
    img_size, num_channels, num_classes = 28, 1, 10
    x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
    x_image=tf.pad(tf.reshape(x,[-1,28,28,1]),[[0,0],[2,2],[2,2],[0,0]])
    y_ = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_')
    W_conv1 = weight_variable([5, 5, 1, 6], name='conv_1', wd=None)
    b_conv1 = bias_variable([6], 0.1)
    conv1 = conv2d(x_image, W_conv1)
    h_conv1 = tf.nn.relu(conv1 + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 6, 16], name='conv_2', wd=None)
    b_conv2 = bias_variable([16], 0.1)
    conv2 = conv2d(h_pool1, W_conv2)
    h_conv2 = tf.nn.relu(conv2 + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    #h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

    W_conv3 = weight_variable([5, 5, 16, 120], name='conv_3', wd=None)
    b_conv3 = bias_variable([120], 0.1)
    conv3 = conv2d(h_pool2, W_conv3)+ b_conv3
    h_conv3=tf.nn.relu(conv3)
    conv3_flat = tf.reshape(h_conv3, [-1, 120])

    W_fc1 = weight_variable([120, 84], name='fc_1', wd=None)
    b_fc1 = bias_variable([84], 0.1)
    h_fc1 = tf.nn.relu(tf.matmul(conv3_flat, W_fc1) + b_fc1)

    W_fc2 = weight_variable([84, 10], name='fc_2', wd=None)
    b_fc2 = bias_variable([10], 0.1)
    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))
    lr = tf.placeholder(tf.float32, shape=[])
    opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
    train_step = opt.minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    pw1 = tf.placeholder(tf.float32, [5, 5, 1, 6])
    opw1 = tf.assign(W_conv1, pw1)

    pb1 = tf.placeholder(tf.float32, [6])
    opb1 = tf.assign(b_conv1, pb1)

    pw2 = tf.placeholder(tf.float32, [5, 5, 6, 16])
    opw2 = tf.assign(W_conv2, pw2)

    pb2 = tf.placeholder(tf.float32, [16])
    opb2 = tf.assign(b_conv2, pb2)

    pw3 = tf.placeholder(tf.float32, [5, 5, 16, 120])
    opw3 = tf.assign(W_conv3, pw3)

    pb3 = tf.placeholder(tf.float32, [120])
    opb3 = tf.assign(b_conv3, pb3)

    pw4 = tf.placeholder(tf.float32, [120, 84])
    opw4 = tf.assign(W_fc1, pw4)

    pb4 = tf.placeholder(tf.float32, [84])
    opb4 = tf.assign(b_fc1, pb4)
    
    pw5 = tf.placeholder(tf.float32, [84, 10])
    opw5 = tf.assign(W_fc2, pw5)

    pb5 = tf.placeholder(tf.float32, [10])
    opb5 = tf.assign(b_fc2, pb5)

    rounds = 100
    E = 0
    batch_size = 500
    client_num=0
    init_lr = 0.05
    decay = 0.995
    total_batch = 12
    data = Mnist_shards() # input data
    images_train_split, labels_train_split = None,None
    learning_rate = init_lr

    sess_nonIID = tf.Session()
    sess_nonIID.run(tf.global_variables_initializer())
    start_time = time.time()
    y_test=to_categorical(data.y_test, 10)
    def init(self,Client_num,iid_data,epoch,batch_size):
        self.E=epoch
        self.batch_size=batch_size
        if (iid_data==1):
            self.images_train_split, self.labels_train_split=self.data.mnist_iid_shards(Client_num)
        elif (iid_data==0):
            self.images_train_split, self.labels_train_split=self.data.mnist_noniid(Client_num,2000)
        self.client_num=Client_num
        self.total_batch=60000//self.client_num//self.batch_size
    def start (self):

        temp_weights = self.sess_nonIID.run([self.W_conv1, self.W_conv2,self.W_conv3, self.W_fc1, self.W_fc2])
        temp_biases = self.sess_nonIID.run([self.b_conv1, self.b_conv2, self.b_conv3,self.b_fc1, self.b_fc2])
        # empty weights and biases from clients
        clients_weights, clients_biases = [], []
        start_time=time.time()
        # use the same shared weights and biases for all clients in the current round
        print("开始学习")
        for client in range(self.client_num):
            for epoch in range(self.E):
                start, end = 0, self.batch_size

                for i in range(self.total_batch):
                    batch_x, batch_y = self.images_train_split[client][start:end, :], self.labels_train_split[client][
                                                                                        start:end, :]
                    start += self.batch_size
                    end += self.batch_size
                    self.sess_nonIID.run(self.train_step, feed_dict={self.x: batch_x.reshape(self.batch_size, 28, 28, 1),
                                                        self.y_: batch_y, self.lr: self.learning_rate})
                # test_accuracy = self.sess_nonIID.run(self.accuracy, feed_dict={self.x: self.data.x_test,
                #                                          self.y_: to_categorical(self.data.y_test, 10)})
                # test_loss = self.sess_nonIID.run(self.cross_entropy, feed_dict={self.x: self.data.x_test,self.y_: to_categorical(self.data.y_test, 10)})

                # print(' training_arruracy:', test_accuracy,' loss:',test_loss)

            clients_weights.append(self.sess_nonIID.run([self.W_conv1, self.W_conv2,self.W_conv3, self.W_fc1, self.W_fc2]))
            clients_biases.append(self.sess_nonIID.run([self.b_conv1, self.b_conv2,self.b_conv3, self.b_fc1, self.b_fc2]))
            test_accuracy = self.sess_nonIID.run(self.accuracy, feed_dict={self.x: self.data.x_test,
                                                         self.y_: to_categorical(self.data.y_test, 10)})
            test_loss = self.sess_nonIID.run(self.cross_entropy, feed_dict={self.x: self.data.x_test,self.y_: to_categorical(self.data.y_test, 10)})

            print(' training_arruracy:', test_accuracy,' loss:',test_loss)
            # reset shared weights and biases
            # all clients use the parameter in the previous round
            self.sess_nonIID.run(self.opw1, feed_dict={self.pw1: temp_weights[0]})  # pw1->W_conv1 / temp_weights->trained_weights
            self.sess_nonIID.run(self.opw2, feed_dict={self.pw2: temp_weights[1]})
            self.sess_nonIID.run(self.opw3, feed_dict={self.pw3: temp_weights[2]})
            self.sess_nonIID.run(self.opw4, feed_dict={self.pw4: temp_weights[3]})
            self.sess_nonIID.run(self.opw5, feed_dict={self.pw5: temp_weights[4]})

            self.sess_nonIID.run(self.opb1, feed_dict={self.pb1: temp_biases[0]})
            self.sess_nonIID.run(self.opb2, feed_dict={self.pb2: temp_biases[1]})
            self.sess_nonIID.run(self.opb3, feed_dict={self.pb3: temp_biases[2]})
            self.sess_nonIID.run(self.opb4, feed_dict={self.pb4: temp_biases[3]})
            self.sess_nonIID.run(self.opb5, feed_dict={self.pb5: temp_biases[4]})
            

        # compute new federated averaging(fedavg) weights and biases

        print("权重更新")
        # fedavg_weights = fedavg(clients_weights)
        # fedavg_biases = fedavg(clients_biases)

        end_time = time.time()
        print('Time: %.2fs' % (end_time - start_time))
        return [clients_weights,clients_biases]
    def next(self,input_weight,input_biases):
        #取得传入的网络参数
        # print("1")
        temp_weights=self.sess_nonIID.run([self.W_conv1, self.W_conv2,self.W_conv3, self.W_fc1, self.W_fc2])
        input_weight=np.array(input_weight)
        input_biases=np.array(input_biases)
        #平均
        input_weight/=self.client_num
        input_biases/=self.client_num
        #喂入参数
        # print("1")
        temp_weights[0] = input_weight[0:150].reshape(5,5,1,6)
        temp_weights[1] = input_weight[150:2550].reshape(5,5,6,16)
        temp_weights[2] = input_weight[2550:50550].reshape(5,5,16,120)
        temp_weights[3] = input_weight[50550:60630].reshape(120,84)
        temp_weights[4] = input_weight[60630:61470].reshape(84,10)
        # print("1")
        self.sess_nonIID.run(self.opw1, feed_dict={self.pw1: temp_weights[0]})  # pw1->W_conv1 / temp_weights->trained_weights
        self.sess_nonIID.run(self.opw2, feed_dict={self.pw2: temp_weights[1]})
        self.sess_nonIID.run(self.opw3, feed_dict={self.pw3: temp_weights[2]})
        self.sess_nonIID.run(self.opw4, feed_dict={self.pw4: temp_weights[3]})
        self.sess_nonIID.run(self.opw5, feed_dict={self.pw5: temp_weights[4]})
        self.sess_nonIID.run(self.opb1, feed_dict={self.pb1: input_biases[0:6]})
        self.sess_nonIID.run(self.opb2, feed_dict={self.pb2: input_biases[6:22]})
        self.sess_nonIID.run(self.opb3, feed_dict={self.pb3: input_biases[22:142]})
        self.sess_nonIID.run(self.opb4, feed_dict={self.pb4: input_biases[142:226]})
        self.sess_nonIID.run(self.opb5, feed_dict={self.pb5: input_biases[226:236]})
        #print("1")
        temp_weights = self.sess_nonIID.run([self.W_conv1, self.W_conv2,self.W_conv3, self.W_fc1, self.W_fc2])
        temp_biases = self.sess_nonIID.run([self.b_conv1, self.b_conv2, self.b_conv3,self.b_fc1, self.b_fc2])

        # empty weights and biases from clients
        clients_weights, clients_biases = [], []
        start_time=time.time()
        # use the same shared weights and biases for all clients in the current round
        print("开始学习")
        for client in range(self.client_num):
            for epoch in range(self.E):
                start, end = 0, self.batch_size

                for i in range(self.total_batch):
                    #print("1")
                    batch_x, batch_y = self.images_train_split[client][start:end, :], self.labels_train_split[client][
                                                                                        start:end, :]
                    start += self.batch_size
                    end += self.batch_size
                    self.sess_nonIID.run(self.train_step, feed_dict={self.x: batch_x.reshape(self.batch_size, 28, 28, 1),
                                                        self.y_: batch_y, self.lr: self.learning_rate})

            clients_weights.append(self.sess_nonIID.run([self.W_conv1, self.W_conv2,self.W_conv3, self.W_fc1, self.W_fc2]))
            clients_biases.append(self.sess_nonIID.run([self.b_conv1, self.b_conv2,self.b_conv3, self.b_fc1, self.b_fc2]))
            # test_accuracy = self.sess_nonIID.run(self.accuracy, feed_dict={self.x: self.data.x_test,
            #                                              self.y_: to_categorical(self.data.y_test, 10)})
            # test_loss = self.sess_nonIID.run(self.cross_entropy, feed_dict={self.x: self.data.x_test,self.y_: to_categorical(self.data.y_test, 10)})
            # print(' training_arruracy:', test_accuracy,' loss:',test_loss)
            # reset shared weights and biases
            # all clients use the parameter in the previous round
            self.sess_nonIID.run(self.opw1, feed_dict={self.pw1: temp_weights[0]})  # pw1->W_conv1 / temp_weights->trained_weights
            self.sess_nonIID.run(self.opw2, feed_dict={self.pw2: temp_weights[1]})
            self.sess_nonIID.run(self.opw3, feed_dict={self.pw3: temp_weights[2]})
            self.sess_nonIID.run(self.opw4, feed_dict={self.pw4: temp_weights[3]})
            self.sess_nonIID.run(self.opw5, feed_dict={self.pw5: temp_weights[4]})

            self.sess_nonIID.run(self.opb1, feed_dict={self.pb1: temp_biases[0]})
            self.sess_nonIID.run(self.opb2, feed_dict={self.pb2: temp_biases[1]})
            self.sess_nonIID.run(self.opb3, feed_dict={self.pb3: temp_biases[2]})
            self.sess_nonIID.run(self.opb4, feed_dict={self.pb4: temp_biases[3]})
            self.sess_nonIID.run(self.opb5, feed_dict={self.pb5: temp_biases[4]})

        # compute new federated averaging(fedavg) weights and biases

        print("权重更新")
        #fedavg_weights = fedavg(clients_weights)
        #fedavg_biases = fedavg(clients_biases)

        end_time = time.time()
        print('Time: %.2fs' % (end_time - start_time))
        return [clients_weights,clients_biases]
    def test (self,input_weight,input_biases):
        print("进入评估流程")
        # print(input_weight[1:10])
        temp_weights=self.sess_nonIID.run([self.W_conv1, self.W_conv2,self.W_conv3, self.W_fc1, self.W_fc2])
        #取得传入的网络参数
        input_weight=np.array(input_weight)
        input_biases=np.array(input_biases)
        #平均
        input_weight/=self.client_num
        input_biases/=self.client_num
        #喂入参数
        temp_weights[0] = input_weight[0:150].reshape(5,5,1,6)
        temp_weights[1] = input_weight[150:2550].reshape(5,5,6,16)
        temp_weights[2] = input_weight[2550:50550].reshape(5,5,16,120)
        temp_weights[3] = input_weight[50550:60630].reshape(120,84)
        temp_weights[4] = input_weight[60630:61470].reshape(84,10)
        self.sess_nonIID.run(self.opw1, feed_dict={self.pw1: temp_weights[0]})  # pw1->W_conv1 / temp_weights->trained_weights
        self.sess_nonIID.run(self.opw2, feed_dict={self.pw2: temp_weights[1]})
        self.sess_nonIID.run(self.opw3, feed_dict={self.pw3: temp_weights[2]})
        self.sess_nonIID.run(self.opw4, feed_dict={self.pw4: temp_weights[3]})
        self.sess_nonIID.run(self.opw5, feed_dict={self.pw5: temp_weights[4]})
        self.sess_nonIID.run(self.opb1, feed_dict={self.pb1: input_biases[0:6]})
        self.sess_nonIID.run(self.opb2, feed_dict={self.pb2: input_biases[6:22]})
        self.sess_nonIID.run(self.opb3, feed_dict={self.pb3: input_biases[22:142]})
        self.sess_nonIID.run(self.opb4, feed_dict={self.pb4: input_biases[142:226]})
        self.sess_nonIID.run(self.opb5, feed_dict={self.pb5: input_biases[226:236]})
        print("网络更新完毕，开始评估模型")
        #images_test, labels_test = self.data.get_test()
        #temp_weights = self.sess_nonIID.run([self.W_conv1, self.W_conv2, self.W_fc1, self.W_fc2])
        #temp_biases = self.sess_nonIID.run([self.b_conv1, self.b_conv2, self.b_fc1, self.b_fc2])
        # acc=0
        # for i in range(100):
        #     acc+=self.sess_nonIID.run(self.accuracy, feed_dict={self.x: images_test[i],self.y_: labels_test[i]})
        # acc/=100
        # test_accuracy=0
        # start, end = 0, self.batch_size
        # print("网络更新完毕，开始评估模型2")
        # for i in range(self.total_batch):
        #         batch_x, batch_y = self.data.x_test[start:end, :], self.y_test[start:end, :]
        #         start += self.batch_size
        #         end += self.batch_size
        #         test_accuracy+=self.sess_nonIID.run(self.accuracy, feed_dict={self.x: batch_x.reshape(self.batch_size, 28, 28, 1),
        #                                             self.y_: batch_y})
        test_accuracy = self.sess_nonIID.run(self.accuracy, feed_dict={self.x: self.data.x_test,
                                                         self.y_: to_categorical(self.data.y_test, 10)})
        # # test_loss = self.sess_nonIID.run(self.cross_entropy, feed_dict={self.x: self.data.x_test,
        # #                                                   self.y_: to_categorical(self.data.y_test, 10)})
        # test_accuracy/=10000//self.batch_size
        print(' training_arruracy:', test_accuracy)
        return test_accuracy
    def Benchmark(self):
        test_accuracy=[]
        start_time=time.time()
        # use the same shared weights and biases for all clients in the current round
        print("开始学习")
        for client in range(self.client_num):
            for epoch in range(self.E):
                start, end = 0, self.batch_size

                for i in range(self.total_batch):
                    batch_x, batch_y = self.images_train_split[client][start:end, :], self.labels_train_split[client][
                                                                                        start:end, :]
                    start += self.batch_size
                    end += self.batch_size
                    self.sess_nonIID.run(self.train_step, feed_dict={self.x: batch_x.reshape(self.batch_size, 28, 28, 1),
                                                        self.y_: batch_y, self.lr: self.learning_rate})
                test_accuracy.append(  self.sess_nonIID.run(self.accuracy, feed_dict={self.x: self.data.x_test,
                                                          self.y_: to_categorical(self.data.y_test, 10)}))
                # test_loss = self.sess_nonIID.run(self.cross_entropy, feed_dict={self.x: self.data.x_test,self.y_: to_categorical(self.data.y_test, 10)})

                print('Epoch:',epoch, 'training_arruracy:', test_accuracy)
        print("权重更新")

        end_time = time.time()
        end_time-= start_time
        print('Time: %.2fs' % (end_time))
        X=[i for i in range(self.E)]
        return [test_accuracy,end_time,X]
def draw_plt(X,Y,time):
    l1=plt.plot(X,Y,'r-')
    plt.title('Time: %.2fs' % (time))
    plt.xlabel('total epoch')
    plt.ylabel('accuracy')
    plt.show()

lenet5=LeNet_5()
def init(Client_num,iid_data,epoch,batch_size):
    lenet5.init(Client_num,iid_data,epoch,batch_size)
def Benchmark():
    test_accuracy,end_time,X= lenet5.Benchmark()
    draw_plt(X,test_accuracy,end_time)
    return test_accuracy
def run():
    return lenet5.start()#61470          
def test(input_weight,input_biases):
    return lenet5.test(input_weight,input_biases)
def next(input_weight,input_biases):
    return lenet5.next(input_weight,input_biases)

#init(10,1,10,500)
# # #Benchmark()
# run()
# test(np.zeros(61470),np.zeros(236))
#next(np.zeros(61470),np.zeros(236))
# test(np.zeros(61470),np.zeros(236))
#draw_plt([1,2,3],[30,50,70],3.14)
                