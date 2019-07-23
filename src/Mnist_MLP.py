import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from data_source import Mnist_shards
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import time
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

class Mnist_MLP:
    in_units = 784 #输入节点数
    h1_units = 400 #隐含层节点数
    W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1)) #初始化隐含层权重W1，服从默认均值为0，标准差为0.1的截断正态分布
    b1 = tf.Variable(tf.zeros([h1_units])) #隐含层偏置b1全部初始化为0
    W2 = tf.Variable(tf.zeros([h1_units, 10])) 
    b2 = tf.Variable(tf.zeros([10]))
    x = tf.placeholder(tf.float32, [None, in_units])
    keep_prob = tf.placeholder(tf.float32) #Dropout失活率

    pW1=tf.placeholder(tf.float32,[in_units,h1_units])
    opW1=tf.assign(W1,pW1)

    pW2=tf.placeholder(tf.float32,[h1_units,10])
    opW2=tf.assign(W2,pW2)

    pb1=tf.placeholder(tf.float32,[h1_units])
    opb1=tf.assign(b1,pb1)

    pb2=tf.placeholder(tf.float32,[10])
    opb2=tf.assign(b2,pb2)

    #定义模型结构
    hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
    y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)
    #训练部分
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)
    data = Mnist_shards()
    #定义一个Interactivesession会话并初始化全部变量
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #初始化容器
    clients_weights,clients_biases=[],[]
    temp_weights=sess.run([W1,W2])
    temp_biases=sess.run([b1,b2])
    #federated参数
    Client_num=0
    Epoch=0
    Batch_size=0
    images_train_split,labels_train_split=None,None
    total_batch=0
    data.y_test=to_categorical(data.y_test, 10)
    def init(self,client_num,iid_data,epoch,batch_size):
        self.Epoch=epoch
        self.Client_num=client_num
        self.Batch_size=batch_size
        if (iid_data==1):
            self.images_train_split, self.labels_train_split=self.data.mnist_iid_shards(self.Client_num)
        elif (iid_data==0):
            self.images_train_split, self.labels_train_split=self.data.mnist_noniid(self.Client_num,2000)
        self.total_batch=60000//self.Client_num//self.Batch_size
    def run(self):
        temp_weights=self.sess.run([self.W1,self.W2])
        temp_biases=self.sess.run([self.b1,self.b2])
        clients_weights,clients_biases=[],[]
        for client in range(self.Client_num):
            for epoch in range(self.Epoch):
                start, end = 0, self.Batch_size

                for i in range(self.total_batch):
                    batch_x, batch_y = self.images_train_split[client][start:end, :], self.labels_train_split[client][
                                                                                        start:end, :]
                    start += self.Batch_size
                    end += self.Batch_size
                    self.sess.run(self.train_step, feed_dict={self.x: batch_x.reshape(self.Batch_size, 784),
                                                        self.y_: batch_y, self.keep_prob: 0.75})
            #取出节点参数
            clients_weights.append(self.sess.run([self.W1, self.W2]))
            clients_biases.append(self.sess.run([self.b1, self.b2]))
            #每个节点的训练结果
            print('client:',client, ' training_arruracy:', self.accuracy.eval({self.x: self.data.x_test.reshape(10000, 784), self.y_: self.data.y_test, 
                                self.keep_prob: 1.0}))
            #参数归零
            self.sess.run(self.opW1, feed_dict={self.pW1: temp_weights[0]}) 
            self.sess.run(self.opW2, feed_dict={self.pW2: temp_weights[1]}) 
            self.sess.run(self.opb1, feed_dict={self.pb1: temp_biases[0]}) 
            self.sess.run(self.opb2, feed_dict={self.pb2: temp_biases[1]}) 
        return [clients_weights,clients_biases]
    def next(self,input_weight,input_biases):
        clients_weights,clients_biases=[],[]
        temp_weights=self.sess.run([self.W1,self.W2])
        input_weight=np.array(input_weight)
        input_biases=np.array(input_biases)
        input_weight/=self.Client_num
        input_biases/=self.Client_num
        temp_weights[0]=input_weight[0:313600].reshape(784,400)
        temp_weights[1]=input_weight[313600:].reshape(400,10)
        self.sess.run(self.opW1, feed_dict={self.pW1: temp_weights[0]}) 
        self.sess.run(self.opW2, feed_dict={self.pW2: temp_weights[1]}) 
        self.sess.run(self.opb1, feed_dict={self.pb1:input_biases[0:400]}) 
        self.sess.run(self.opb2, feed_dict={self.pb2:input_biases[400:]}) 
        print("网络更新完毕，开始新一轮训练")
        temp_weights=self.sess.run([self.W1,self.W2])
        temp_biases=self.sess.run([self.b1,self.b2])
        for client in range(self.Client_num):
            for epoch in range(self.Epoch):
                start, end = 0, self.Batch_size

                for i in range(self.total_batch):
                    batch_x, batch_y = self.images_train_split[client][start:end, :], self.labels_train_split[client][
                                                                                        start:end, :]
                    start += self.Batch_size
                    end += self.Batch_size
                    self.sess.run(self.train_step, feed_dict={self.x: batch_x.reshape(self.Batch_size, 784),
                                                        self.y_: batch_y, self.keep_prob: 0.75})
             #取出节点参数
            clients_weights.append(self.sess.run([self.W1, self.W2]))
            clients_biases.append(self.sess.run([self.b1, self.b2]))
            #每个节点的训练结果
            print('client:',client, ' training_arruracy:', self.accuracy.eval({self.x: self.data.x_test.reshape(10000, 784), self.y_: self.data.y_test, 
                                self.keep_prob: 1.0}))
            #参数归零
            self.sess.run(self.opW1, feed_dict={self.pW1: temp_weights[0]}) 
            self.sess.run(self.opW2, feed_dict={self.pW2: temp_weights[1]}) 
            self.sess.run(self.opb1, feed_dict={self.pb1: temp_biases[0]}) 
            self.sess.run(self.opb2, feed_dict={self.pb2: temp_biases[1]}) 
        return [clients_weights,clients_biases]
    def test(self,input_weight,input_biases):
        #取得传入的网络参数
        temp_weights=self.sess.run([self.W1,self.W2])
        input_weight=np.array(input_weight)
        input_biases=np.array(input_biases)
        #平均
        input_weight/=self.Client_num
        input_biases/=self.Client_num
        #喂入参数
        temp_weights[0]=input_weight[0:313600].reshape(784,400)
        temp_weights[1]=input_weight[313600:].reshape(400,10)
        self.sess.run(self.opW1, feed_dict={self.pW1: temp_weights[0]}) 
        self.sess.run(self.opW2, feed_dict={self.pW2: temp_weights[1]}) 
        self.sess.run(self.opb1, feed_dict={self.pb1:input_biases[0:400]}) 
        self.sess.run(self.opb2, feed_dict={self.pb2:input_biases[400:]}) 
        print("网络更新完毕，开始评估模型")

       
        _acc = self.sess.run(self.accuracy, feed_dict={self.x: self.data.x_test.reshape(10000, 784), self.y_: self.data.y_test, self.keep_prob: 1.0})

        # train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 1})
        print(' training_arruracy:', _acc)
        return _acc
    def Benchmark(self):
        test_accuracy=[]
        start_time=time.time()
        # use the same shared weights and biases for all clients in the current round
        print("开始学习")
        for client in range(self.Client_num):
            for epoch in range(self.Epoch):
                start, end = 0, self.Batch_size

                for i in range(self.total_batch):
                    batch_x, batch_y = self.images_train_split[client][start:end, :], self.labels_train_split[client][
                                                                                        start:end, :]
                    start += self.Batch_size
                    end += self.Batch_size
                    self.sess.run(self.train_step, feed_dict={self.x: batch_x.reshape(self.Batch_size, 784),
                                                        self.y_: batch_y, self.keep_prob: 0.75})
                test_accuracy.append(  self.sess.run(self.accuracy, feed_dict={self.x: self.data.x_test.reshape(10000, 784),
                                                          self.y_: self.data.y_test,self.keep_prob: 0.75}))
                # test_loss = self.sess_nonIID.run(self.cross_entropy, feed_dict={self.x: self.data.x_test,self.y_: to_categorical(self.data.y_test, 10)})

                print('Epoch:',epoch, 'training_arruracy:', test_accuracy)
        print("权重更新")

        end_time = time.time()
        end_time-= start_time
        print('Time: %.2fs' % (end_time))
        X=[i for i in range(self.Epoch)]
        return [test_accuracy,end_time,X]
#run()
#test(np.zeros(238200),np.zeros(310))
def draw_plt(X,Y):
    l1=plt.plot(X,Y,'r-')
    #plt.title('Time: %.2fs' % (time))
    plt.xlabel('total epoch')
    plt.ylabel('accuracy')
    plt.show()

mnist_mlp=Mnist_MLP()
def init(client_num,iid_data,epoch,batch_size):
    mnist_mlp.init(client_num,iid_data,epoch,batch_size)
def Benchmark():
    test_accuracy,end_time,X= mnist_mlp.Benchmark()
    draw_plt(X,test_accuracy)
    return test_accuracy
def run():
    return mnist_mlp.run()
def next(input_weight,input_biases):
    return mnist_mlp.next(input_weight,input_biases)
def test(input_weight,input_biases):
    
    return mnist_mlp.test(input_weight,input_biases)
#init(1,1,5,500)
#Benchmark()
#run()
#next(np.zeros(317600),np.zeros(410))
#test(np.zeros(317600),np.zeros(410))