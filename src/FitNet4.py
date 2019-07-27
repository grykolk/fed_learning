from __future__ import print_function
import keras
#from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalMaxPooling2D
from lsuv_init import LSUVinit
from data_source import Cifar10_shards
from keras.utils.np_utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
    def acc_plot(self,loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
        plt.grid(True)
        plt.xlabel("epoch")
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()
    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

class FitNet4:
    history=LossHistory()
    batch_size = 0
    num_classes = 10
    epochs = 0
    client_num=0
    Input_Shape=(32,32,3)
    #data_augmentation = True
    data_augmentation = False
    # The data, shuffled and split between train and test sets:
    #(x_train, y_train), (x_test, y_test) = cifar10.load_data()
    data=Cifar10_shards()
    # print('x_train shape:', x_train.shape)
    # print(x_train.shape[0], 'train samples')
    # print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    # y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                    input_shape=Input_Shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same',
                    input_shape=Input_Shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same',
                    input_shape=Input_Shape))
    model.add(Activation('relu'))
    model.add(Conv2D(48, (3, 3), padding='same',
                    input_shape=Input_Shape))
    model.add(Activation('relu'))
    model.add(Conv2D(48, (3, 3), padding='same',
                    input_shape=Input_Shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(80, (3, 3), padding='same',
                    input_shape=Input_Shape))
    model.add(Activation('relu'))
    model.add(Conv2D(80, (3, 3), padding='same',
                    input_shape=Input_Shape))
    model.add(Activation('relu'))
    model.add(Conv2D(80, (3, 3), padding='same',
                    input_shape=Input_Shape))
    model.add(Activation('relu'))
    model.add(Conv2D(80, (3, 3), padding='same',
                    input_shape=Input_Shape))
    model.add(Activation('relu'))
    model.add(Conv2D(80, (3, 3), padding='same',
                    input_shape=Input_Shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same',
                    input_shape=Input_Shape))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same',
                    input_shape=Input_Shape))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same',
                    input_shape=Input_Shape))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same',
                    input_shape=Input_Shape))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same',
                    input_shape=Input_Shape))
    model.add(Activation('relu'))
    model.add(GlobalMaxPooling2D())
    model.add(Dropout(0.25))

    #model.add(ZeroPadding2D((1, 1)))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Flatten())
    #model.add(Dropout(0.2))
    '''
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    '''

    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.Adam(lr=0.0001)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
    def init(self,Client_num,iid_data,epoch,batch_size):
        self.client_num=Client_num
        self.epochs=epoch
        self.batch_size=batch_size
        if (iid_data==1):
            self.images_train_split, self.labels_train_split=self.data.iid_shards(Client_num)
        elif (iid_data==0):
            self.images_train_split, self.labels_train_split=self.data.noniid(Client_num,2000)
        # self.images_train_split=self.images_train_split.astype('float32')
        # self.data.x_test=self.data.x_test.astype('float32')
        self.data.y_test=to_categorical(self.data.y_test, self.num_classes)
        self.model = LSUVinit(self.model,self.images_train_split[0][:batch_size].reshape(batch_size,32,32,3)) 
       # self.model = LSUVinit(self.model,self.images_train_split[0][:batch_size,:,:,:]) 
        #self.tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph2', histogram_freq=0, write_graph=True, write_images=True)
    # x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')
    def run(self):
        Weight,Biases=[],[]
        init_weight,init_biases=self.get_parameter()

        for c in range(self.client_num):
            self.model.fit(self.images_train_split[c].reshape(50000//self.client_num,32,32,3),self.labels_train_split[c].reshape(50000//self.client_num,10),
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    validation_data=(self.data.x_test.reshape(10000,32,32,3), self.data.y_test.reshape(10000,10)),
                    shuffle=True, callbacks=[])
            temp_weight,temp_biases=self.get_parameter()
            Weight.append(temp_weight)
            Biases.append(temp_biases)
            self.set_parameter(init_weight,init_biases)
        #print(Weight)
        return [Weight,Biases]
    def next(self,weight_vector,biases_vector):
        Weight,Biases=[],[]
        self.set_parameter_from_vector(weight_vector,biases_vector)
        init_weight,init_biases=self.get_parameter()
        #print("init_weight")
       # print(init_weight)
        for c in range(self.client_num):
            self.model.fit(self.images_train_split[c].reshape(50000//self.client_num,32,32,3),self.labels_train_split[c].reshape(50000//self.client_num,10),
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    validation_data=(self.data.x_test.reshape(10000,32,32,3), self.data.y_test.reshape(10000,10)),
                    shuffle=True, callbacks=[])
            temp_weight,temp_biases=self.get_parameter()
            Weight.append(temp_weight)
            Biases.append(temp_biases)
            self.set_parameter(init_weight,init_biases)
        return [Weight,Biases]
    def benchmark(self):
        self.model.fit(self.images_train_split[0].reshape(50000,32,32,3),self.labels_train_split[0].reshape(50000,10),
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_data=(self.data.x_test.reshape(10000,32,32,3), self.data.y_test.reshape(10000,10)),
                shuffle=True, callbacks=[self.history])
        self.history.acc_plot('epoch')
    def test(self,weight_vector,biases_vector):
        self.set_parameter_from_vector(weight_vector,biases_vector)
        acc=self.model.evaluate(self.data.x_test.reshape(10000,32,32,3), self.data.y_test.reshape(10000,10))
        return acc[1]

            
    def get_parameter(self):
        temp_weight,temp_biases=[],[]
        for layer in self.model.layers:
            if isinstance(layer,(Dense, Conv2D)):
                parameter=layer.get_weights()
                temp_weight.append(parameter[0])
                temp_biases.append(parameter[1])
        return temp_weight,temp_biases
    def set_parameter(self,weight,biases):
        i=0
        for layer in self.model.layers:
            if isinstance(layer,(Dense, Conv2D)):
                layer.set_weights([weight[i],biases[i]])
                i+=1
    def set_parameter_from_vector(self,weight_vector,biases_vector):
        #0-2 weight: 3*3*32=288  biases:32
        #3-4 weight: 3*3*48=432  biases:48
        #5-9 weight: 3*3*80=720  biases:80
        #10-14 weight: 3*3*128=1152  biases:128
        #15 weight: 128*500=64000  biases:500
        #16 weight: 500*10=5000  biases:10
       # print(weight_vector[0:10])
        weight_vector=np.array(weight_vector)
        biases_vector=np.array(biases_vector)
        weight_vector/=self.client_num
        biases_vector/=self.client_num
        weight_pointer=0
        biases_pointer=0
        i=0
        for layer in self.model.layers:
            if isinstance(layer,(Dense, Conv2D)):
                if(i==0):
                   # print(weight_pointer,weight_pointer)
                   # print(weight_vector[weight_pointer:weight_pointer+864])
                    weight=weight_vector[weight_pointer:weight_pointer+864].reshape(3,3,3,32)
                    weight_pointer+=864
                   # print(weight)
                    biases=biases_vector[biases_pointer:biases_pointer+32]
                    biases_pointer+=32
                elif (i<=2):
                    weight=weight_vector[weight_pointer:weight_pointer+9216].reshape(3,3,32,32)
                    weight_pointer+=9216
                    biases=biases_vector[biases_pointer:biases_pointer+32]
                    biases_pointer+=32
                elif(i==3):
                    weight=weight_vector[weight_pointer:weight_pointer+13824].reshape(3,3,32,48)
                    weight_pointer+=13824
                    biases=biases_vector[biases_pointer:biases_pointer+48]
                    biases_pointer+=48
                elif(i==4):
                    weight=weight_vector[weight_pointer:weight_pointer+20736].reshape(3,3,48,48)
                    weight_pointer+=20736
                    biases=biases_vector[biases_pointer:biases_pointer+48]
                    biases_pointer+=48
                elif(i==5):
                    weight=weight_vector[weight_pointer:weight_pointer+34560].reshape(3,3,48,80)
                    weight_pointer+=34560
                    biases=biases_vector[biases_pointer:biases_pointer+80]
                    biases_pointer+=80
                elif(i<=9):
                    weight=weight_vector[weight_pointer:weight_pointer+57600].reshape(3,3,80,80)
                    weight_pointer+=57600
                    biases=biases_vector[biases_pointer:biases_pointer+80]
                    biases_pointer+=80
                elif(i==10):
                    weight=weight_vector[weight_pointer:weight_pointer+92160].reshape(3,3,80,128)
                    weight_pointer+=92160
                    biases=biases_vector[biases_pointer:biases_pointer+128]
                    biases_pointer+=128
                elif(i<=14):
                    weight=weight_vector[weight_pointer:weight_pointer+147456].reshape(3,3,128,128)
                    weight_pointer+=147456
                    biases=biases_vector[biases_pointer:biases_pointer+128]
                    biases_pointer+=128
                elif(i==15):
                    weight=weight_vector[weight_pointer:weight_pointer+64000].reshape(128,500)
                    weight_pointer+=64000
                    biases=biases_vector[biases_pointer:biases_pointer+500]
                    biases_pointer+=500
                elif (i==16):
                    weight=weight_vector[weight_pointer:weight_pointer+5000].reshape(500,10)
                    weight_pointer+=5000
                    biases=biases_vector[biases_pointer:biases_pointer+10]
                    biases_pointer+=10
                layer.set_weights([weight,biases])
                i+=1
    
    
    
fitnet4=FitNet4()
def draw_plt(X,Y,time):
    l1=plt.plot(X,Y,'r-')
    plt.title('Time: %.2fs' % (time))
    plt.xlabel('total epoch')
    plt.ylabel('accuracy')
    plt.show()
def init(Client_num,iid_data,epoch,batch_size):
    return fitnet4.init(Client_num,iid_data,epoch,batch_size)
def Benchmark():
    return fitnet4.benchmark()
def run():
    return fitnet4.run()
def next(weight,biases):
    return fitnet4.next(weight,biases)
def test(weight,biases):
    return fitnet4.test(weight,biases)
#init(1,1,5,500)
#run()
#Benchmark()
#next(np.ones(1069800),np.ones(1742))
#test(np.zeros(1069800),np.zeros(1742))