import models
import nndata
import numpy as np
import riemannian
import  tensorflow as tf
from    tensorflow.keras import  layers, optimizers, Sequential, metrics
from 	tensorflow import keras
from keras.utils.np_utils import *
import  os
import  tensorflow as tf
from    tensorflow.keras import layers, optimizers, datasets, Sequential
import  os


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.random.set_seed(2345)

conv_layers = [ # 5 units of conv + max pooling
    # unit 1
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit 2
    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit 3
    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit 4
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit 5
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')

]

from keras.callbacks import ModelCheckpoint
augmulitple = 5

SPLITS = 5
input_length = 6 * 160  # = 3s
# input_length1 = 3 * 160 #移动窗口增强法，匹配输入卷积层数据维度
electrodes = list(range(64))
nch = 64
epochs = 3
epoch_steps = 5  # record performance 5 times per epoch
batch = 16
# nclasses = [2, 3, 4]
nclasses = [4]
# splits = list(range(5))
splits = [0]



def data_split(x, y, run, nruns, idx):
    train_idx, test_idx = nndata.split_idx(run, nruns, idx, seed=1337)
    x_train, y_train = signal_train(x[train_idx], y[train_idx])
    x_test, y_test = signal_test(x[test_idx], y[test_idx])

    x_train = x_train.reshape(-1, 64,64,1)
    y_train = y_train.flatten()
    x_test = x_test.reshape(-1, 64, nch,1)
    y_test = y_test.flatten()
    return x_train, y_train, x_test,y_test


def signal_train(x,y):

    x, y = nndata.AddGussio(x, y, 0, 0.01, augmulitple=augmulitple)
    # 预处理
    signal = nndata.Process(x)
    # 协方差
    signals = riemannian.Signals_Covariance(signal)
    # 找均值点
    # x, y = riemannian.training_data_cov_means(signals, y, num_classes=4)
    x = riemannian.TangentSpaceMapping(signals, y, num_classes=4)

    return  x,y

def signal_test(x,y):

    # 预处理
    signal = nndata.Process(x)
    # 协方差
    signals = riemannian.Signals_Covariance(signal)
    # 找均值点
    # x, y = riemannian.training_data_cov_means(signals, y, num_classes=4)
    x = riemannian.TangentSpaceMapping(signals, y, num_classes=4)
    return  x,y



# for j, nclasses in enumerate(nclasses):
#     try:
#         del X, y
#     except:
#         pass
#     X, y = nndata.load_raw_data(electrodes=electrodes, num_classes=nclasses)
#     for ii, i in enumerate(splits):
#         idx = list(range(len(X)))
#         x, y ,x_val, y_val = data_split(X, y, i, 5, idx)



def main():

    batch_size = 16
    epochs = 100
    X, y = nndata.load_raw_data(electrodes=electrodes, num_classes=4)

    idx = list(range(len(X)))
    x, y, x_test, y_test = data_split(X, y, 0, 5, idx)
    print(1)

    train_db = tf.data.Dataset.from_tensor_slices((x, y))
    train_db = train_db.shuffle(1000).batch(batch_size=batch_size)

    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.batch(batch_size=batch_size)

    sample = next(iter(train_db))
    print('sample:', sample[0].shape, sample[1].shape,
          tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))

    # [b, 32, 32, 3] => [b, 1, 1, 512]
    conv_net = Sequential(conv_layers)

    fc_net = Sequential([
        layers.Dense(1024, activation=tf.nn.relu),
        layers.Dense(512, activation=tf.nn.relu),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(4, activation=None),
    ])

    conv_net.build(input_shape=[None, 64, 64, 1])
    fc_net.build(input_shape=[None, 2048])
    optimizer = optimizers.Adam(lr=1e-3)

    # [1, 2] + [3, 4] => [1, 2, 3, 4]对所有层
    variables = conv_net.trainable_variables + fc_net.trainable_variables

    for epoch in range(epochs):

        for step, (x,y) in enumerate(train_db):

            with tf.GradientTape() as tape:
                # [b, 32, 32, 3] => [b, 1, 1, 512]
                out = conv_net(x)
                # flatten, => [b, 512]
                out = tf.reshape(out, [-1, 2048])
                # [b, 2048] => [b, 4]
                logits = fc_net(out)
                # [b] => [b, 100]
                # y_onehot = tf.one_hot(y, depth=4)
                y_onehot = to_categorical(y, 4)
                # compute loss
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)


            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))

            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss))



        total_num = 0
        total_correct = 0
        for x,y in test_db:

            out = conv_net(x)
            out = tf.reshape(out, [-1, 2048])
            logits = fc_net(out)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            # pred = tf.one_hot(pred, depth=4)
            pred = tf.cast(pred, dtype=tf.int32)
            y = tf.cast(y, dtype=tf.int32)

            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_num += x.shape[0]
            total_correct += int(correct)

        acc = total_correct / total_num
        print(epoch, 'acc:', acc)



if __name__ == '__main__':
    main()
