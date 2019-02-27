from keras.utils.data_utils import get_file
from keras.datasets import mnist, cifar10, reuters
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
import random
import numpy as np


class Mnist_CNN_shards():  # google paper split
    NUM_SHARDS_PER_CLIENT = 2
    NAME = 'mnist'

    def __init__(self):
        self.nb_classes = 10
        self.input_shape = (28, 28, 1)

        (x_train, self.y_train), (x_test, self.y_test) = mnist.load_data()
        self.x_train = x_train.astype('float')
        self.x_test = x_test.astype('float')
        self.x_train = np.expand_dims(self.x_train, axis=-1) / 255.
        self.x_test = np.expand_dims(self.x_test, axis=-1) / 255.
        self.num_shards = 200
        self.num_imgs = 300

    def mnist_iid_shards(self, num_users):
        num_items = int(len(self.y_train) / 100)  # 600
        dict_users, all_idxs = {}, [i for i in range(len(self.y_train))]
        for i in range(num_users):
            dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])
            dict_users[i] = np.array(list(dict_users[i]))
        x_train_set = {i: self.x_train[dict_users[i]] for i in range(num_users)}
        y_train_set = {i: self.y_train[dict_users[i]] for i in range(num_users)}
        print('client labels: ', y_train_set)
        y_train_set = {i: to_categorical(self.y_train[dict_users[i]], self.nb_classes) for i in range(num_users)}
        return (x_train_set, y_train_set)

    def mnist_noniid(self, num_users):
        idx_shard = [i for i in range(self.num_shards)]
        dict_users = {i: np.array([]) for i in range(num_users)}

        # sort labels
        idxs = self.y_train.argsort()

        # divide and assign
        for i in range(num_users):
            rand_set = set(np.random.choice(idx_shard, Mnist_CNN_shards.NUM_SHARDS_PER_CLIENT, replace=False))  # randomly select 2 different shards
            idx_shard = list(set(idx_shard) - rand_set)  # remove selected shards from total shards
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand * self.num_imgs:(rand + 1) * self.num_imgs]), axis=0).astype('int')
                idx = np.arange(len(dict_users[i]))
                np.random.shuffle(idx)
                dict_users[i] = dict_users[i][idx]
        x_train_set = {i: self.x_train[dict_users[i]] for i in range(num_users)}
        y_train_set = {i: self.y_train[dict_users[i]] for i in range(num_users)}
        print('client labels: ', y_train_set)
        y_train_set = {i: to_categorical(self.y_train[dict_users[i]], self.nb_classes) for i in range(num_users)}
        return (x_train_set, y_train_set)
