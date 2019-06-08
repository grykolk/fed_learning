import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
in_units = 784 #输入节点数
h1_units = 300 #隐含层节点数
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1)) #初始化隐含层权重W1，服从默认均值为0，标准差为0.1的截断正态分布
b1 = tf.Variable(tf.zeros([h1_units])) #隐含层偏置b1全部初始化为0
W2 = tf.Variable(tf.zeros([h1_units, 10])) 
b2 = tf.Variable(tf.zeros([10]))
x = tf.placeholder(tf.float32, [None, in_units])
keep_prob = tf.placeholder(tf.float32) #Dropout失活率

pw1=tf.placeholder(tf.float32,[in_units,h1_units])
opw1=tf.assign(W1,pw1)

pw2=tf.placeholder(tf.float32,[h1_units,10])
opw2=tf.assign(W2,pw2)

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

#定义一个InteractiveSession会话并初始化全部变量
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#初始化容器
clients_weights,clients_biases=[],[]
temp_weights=sess.run([W1,W2])
temp_biases=sess.run([b1,b2])
def run():
    temp_weights=sess.run([W1,W2])
    temp_biases=sess.run([b1,b2])
    for i in range(10):#十个客户端
        for j in range(300):#每个客户端300轮
            batch_xs, batch_ys = mnist.train.next_batch(100)
            train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})
        #取出节点参数
        clients_weights.append(sess.run([W1, W2]))
        clients_biases.append(sess.run([b1, b2]))
        #每个节点的训练结果
        print('client:',i, ' training_arruracy:', accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, 
                            keep_prob: 1.0}))
        #参数归零
        sess.run(opw1, feed_dict={pw1: temp_weights[0]}) 
        sess.run(opw2, feed_dict={pw2: temp_weights[1]}) 
        sess.run(opb1, feed_dict={pb1: temp_biases[0]}) 
        sess.run(opb2, feed_dict={pb2: temp_biases[1]}) 
    return [clients_weights,clients_biases]
def next(input_weight,input_biases):
    temp_weights=sess.run([W1,W2])
    input_weight=np.array(input_weight)
    input_biases=np.array(input_biases)
    input_weight/=10
    input_biases/=10
    temp_weights[0]=input_weight[0:235200].reshape(784,300)
    temp_weights[1]=input_weight[235200:].reshape(300,10)
    sess.run(opw1, feed_dict={pw1: temp_weights[0]}) 
    sess.run(opw2, feed_dict={pw2: temp_weights[1]}) 
    sess.run(opb1, feed_dict={pb1:input_biases[0:300]}) 
    sess.run(opb2, feed_dict={pb2:input_biases[300:]}) 
    print("网络更新完毕，开始新一轮训练")
    temp_weights=sess.run([W1,W2])
    temp_biases=sess.run([b1,b2])
    for i in range(10):#十个客户端
        for j in range(300):#每个客户端300轮
            batch_xs, batch_ys = mnist.train.next_batch(100)
            train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})
        #取出节点参数
        clients_weights.append(sess.run([W1, W2]))
        clients_biases.append(sess.run([b1, b2]))
        #每个节点的训练结果
        print('client:',i, ' training_arruracy:', accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, 
                            keep_prob: 1.0}))
        #参数归零
        sess.run(opw1, feed_dict={pw1: temp_weights[0]}) 
        sess.run(opw2, feed_dict={pw2: temp_weights[1]}) 
        sess.run(opb1, feed_dict={pb1: temp_biases[0]}) 
        sess.run(opb2, feed_dict={pb2: temp_biases[1]}) 
    return [clients_weights,clients_biases]
def test(input_weight,input_biases):
    #取得传入的网络参数
    temp_weights=sess.run([W1,W2])
    input_weight=np.array(input_weight)
    input_biases=np.array(input_biases)
    #平均
    input_weight/=10
    input_biases/=10
    #喂入参数
    temp_weights[0] = input_weight[0:235200].reshape(784, 300)
    temp_weights[1] = input_weight[235200:].reshape(300,10)
    sess.run(opw1, feed_dict={pw1: temp_weights[0]}) 
    sess.run(opw2, feed_dict={pw2: temp_weights[1]}) 
    sess.run(opb1, feed_dict={pb1: input_biases[0:300]})
    sess.run(opb2, feed_dict={pb2: input_biases[300:]})
    print("网络更新完毕，开始评估模型")

    _acc = 0
    for i_batch in range(100):
        batch_xs, batch_ys = mnist.test.next_batch(100)
        _acc += sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
    _acc /= (i_batch+1)
    # train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 1})
    print('client:', ' training_arruracy:', _acc)
#run()
#test(np.zeros(238200),np.zeros(310))