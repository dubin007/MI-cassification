import models
import nndata
from keras.utils.np_utils import *
from keras.callbacks import ModelCheckpoint
import  tensorflow as tf
import  os
import numpy as np
import pandas as pd
from scipy import signal

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.random.set_seed(2345)

augmulitple = 40
nch =64
SPLITS = 5

def data_split(X, y, augmulitple,train_idx, val_idx, test_idx):

    #对训练集做高斯噪声数据增强，并对所有数据分频处理
    # x1_train, y1_train, x1_val, y1_val, x_test, y_test = nndata.AugmentG(X, y, augmulitple,train_idx, val_idx, test_idx)
    x1_train, y1_train, x1_val, y1_val, x_test, y_test = nndata.AugmentAver(X, y, augmulitple, train_idx, val_idx,
                                                                            test_idx)

    #对数据进行协方差并映射
    # x_train, y_train, x_val, y_val = nndata.n_class_signal_mapping(x1_train, y1_train, x1_val, y1_val)

    # return x_train, y_train, x_val,y_val
    return x1_train, y1_train, x1_val,y1_val, x_test, y_test

def scheduler(epoch):
    # decayrate = 1
    # a0 = 0.01
    # lr1 = 1.0/(1.0+decayrate*epoch)*a0
    if epoch <= 10:
        return 0.01
    elif epoch <= 20:
        return 0.001
    # elif epoch <= 15:
    #     return 0.005
    else:
        return  0.0001



def main():

    electrodes = list(range(64))
    epoch_steps = 4
    epochs = 30
    batch = 32
    nsample = 960
    # nclasses = [2, 3, 4]
    nclasses = [4]
    splits = list(range(5))
    # splits = [0]

    # results = np.zeros((len(nclasses), len(splits), 4, epochs ))
    for j, nclasses in enumerate(nclasses):
        try:
            del X, y
        except:
            pass
        X, y = nndata.load_raw_data(electrodes=electrodes, num_classes=nclasses)
        # X = signal.resample(X,nsample,axis=2)
        # steps_per_epoch = np.prod(X.shape[:2]) / batch * (1 - 1. / SPLITS) / epoch_steps
        for i in enumerate(splits):
            idx = list(range(len(X)))
            train_idx, val_idx, test_idx = nndata.split_idx(idx, a=6,b=2)
            path = r'D:\deeplearing\db-eeg-cnn - mapping _tian_test1\data_index.csv'
            test_idx1 = {'test_index':test_idx}
            pd.DataFrame(test_idx1).to_csv(path)

            x_train, y_train, x_val, y_val, x_test, y_test = data_split(X, y, augmulitple, train_idx, val_idx, test_idx)
            steps_per_epoch = np.prod(x_train.shape[:3]) / batch


            model = models.create_raw_model(
                    nsample=nsample,
                    nclasses=nclasses
                )

            checkpoint_path = "traing1/cp--%dcl-%d.ckpt" % (nclasses, i)

            # checkpoint_path = "traing1/cp--{epoch:d}.ckpt"

            checkpoint_dir = os.path.dirname(checkpoint_path)
            # 创建一个回调，每5个epochs保存模型的权重
            cp_callback = [
                tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1,
                                                             save_weights_only=True, period=1),
                tf.keras.callbacks.LearningRateScheduler(scheduler)]
            model.save_weights(checkpoint_path.format(epoch=0))


            # run training
            h = model.fit_generator(
                nndata.crossval_gen(x_train, y_train,  batch,nsample), verbose=1,
                validation_data=nndata.crossval_test(x_val, y_val),
                steps_per_epoch=steps_per_epoch, epochs=epochs , callbacks = cp_callback
            )

            print(h.history)
            path = r'D:\deeplearing\db-eeg-cnn - mapping _tian_test1\data_109.csv'
            pd.DataFrame(h.history).to_csv(path)

            new1_model = model
            x_test, y_test = nndata.crossval_test_fi(x_test, y_test)
            loss1 = np.zeros(epochs)
            acc1 = np.zeros(epochs)
            for i in range(epochs):

                path1 = "traing1/cp--%d.ckpt" % i
                new1_model.load_weights(path1)
                loss, acc = new1_model.evaluate(x_test, y_test,verbose=2)
                loss1[i] = loss
                acc1[i] = acc

            Result = {'res':acc1,'loss':loss1}
            path = r'D:\deeplearing\db-eeg-cnn - mapping _tian_test1\data_res.csv'
            pd.DataFrame(Result).to_csv(path)

            print("i can do it")


if __name__ == '__main__':
    main()
