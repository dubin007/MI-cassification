import util
import numpy as np
import tensorflow as tf
from keras.utils.np_utils import *
import riemannian
from scipy import signal
import pyriemann
from pyriemann.utils.mean import mean_covariance

MOVEMENT_START = 1 * 160  # MI starts 1s after trial begin
MOVEMENT_END = 5 * 160  # MI lasts 4 seconds
NOISE_LEVEL = 0.01
clas = 4
fc = 160
aug = 40
ntrials = 84

def load_raw_data(electrodes, subject=None, num_classes=2, long_edge=False):
    # load from file
    trials = []
    labels = []

    if subject == None:
        # subject_ids = range(1, 110)
        subject_ids = range(1, 11)
    else:
        try:
            subject_ids = [int(subject)]
        except:
            subject_ids = subject

    for subject_id in subject_ids:
        print("load subject %d" % (subject_id,))
        t, l, loc, fs = util.load_physionet_data(subject_id, num_classes, long_edge=long_edge)
        if num_classes == 2 and t.shape[0] != 42:
            # drop subjects with less trials
            continue
        trials.append(t[:, :, electrodes])
        labels.append(l)

    return np.array(trials).reshape((len(trials),) + trials[0].shape), np.array(labels)




def split_idx( idx,a,b):
    """
    Shuffle and split a list of indexes into training and test data with a fixed
    random seed for reproducibility

    run: index of the current split (zero based)
    nruns: number of splits (> run)
    idx: list of indices to split
    """
    rs = np.random.RandomState()
    rs.shuffle(idx)
    start = int(a / 10. * len(idx))
    end = int((b+a) / 10. * len(idx))
    train_idx = idx[0:start]
    test_idx = idx[start:end]
    val_idx = idx[end:]
    return train_idx, val_idx, test_idx
    # return train_idx, test_idx

def n_classfilter(x,y,arg):

    # x = np.squeeze(x)
    signal = np.zeros((5,)+x.shape)
    label = np.zeros((5,)+y.shape)

    signal[0,:] = filter(x,0.5,4,arg)
    label[0,:] = y
    signal[1, :] = filter(x, 4, 8,arg)
    label[1, :] = y
    signal[2, :] = filter(x, 8, 13,arg)
    label[2, :] = y
    signal[3, :] = filter(x, 13, 32,arg)
    label[3, :] = y
    signal[4, :] = filter(x, 32, 50,arg)
    label[4, :] = y

    return  signal,label

def filter(x,low_filter,high_filter, aru):
    Wn = [low_filter*2/fc,high_filter*2/fc]
    b, a = signal.butter(3, Wn, 'bandpass')
    # x = x.transpose((0, 1, 2, 4, 3))
    fdata = np.zeros(x.shape)
    if aru:
        for i in range(len(x)):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    for l in range(x.shape[4]):
                        fdata[i, j, k, :, l] = signal.filtfilt(b, a, x[i, j, k, :, l])
        # fdata = fdata.transpose((0, 1,  2, 4, 3))
        return  fdata
    else:
        for i in range(len(x)):
            for j in range(x.shape[1]):
                for l in range(x.shape[3]):
                    fdata[i, j, :, l] = signal.filtfilt(b, a, x[i, j, :, l])
        # fdata = fdata.transpose((0, 1,  2, 4, 3))
        return  fdata

def n_class_signal_mapping(x_train, y_train, x_val, y_val):
    x1_train = np.zeros((x_train.shape[0:4] + (64,64, )))
    y1_train = np.zeros((y_train.shape))
    x1_val = np.zeros((x_val.shape[0:3] + (64, 64,)))
    y1_val = np.zeros((y_val.shape))
    for j in range(len(x_train)):
        x1_train[j, :], y1_train[j, :], x1_val[j, :], y1_val[j, :] = signal_mapping(x_train[j], y_train[j], x_val[j], y_val[j])
        print("yes")
    x1_train = x1_train.transpose(1, 2, 3, 4, 5, 0)
    x1_val = x1_val.transpose(1, 2, 3, 4, 0)
    # y1_train = y1_train[0]
    # y1_val = y1_val[0]
    return x1_train, y1_train, x1_val, y1_val


def signal_mapping(x_train, y_train, x_val, y_val):
    #训练集
    signals1,core = Signals_Covariance(x_train,None,mean_all=True)

    #测试集
    signals2 = Signals_Covariance(x_val, core, mean_all=False)
    # y_test = y_val.reshape((-1,))
    # y_train = y_train.reshape((-1,))

    return signals1,y_train,signals2,y_val

def Signals_Covariance(signals,core_test,mean_all=True):

    if mean_all:
        signal = signals.reshape((-1,) + (signals.shape[-2:]))
        signal = np.transpose(signal, axes=[0, 2, 1])
        x_out = pyriemann.estimation.Covariances().fit_transform(signal)

        core = mean_covariance(x_out, metric='riemann')
        # core = training_data_cov_means(X,y,num_classes=4)
        core = core ** (-1 / 2)

        signal1, core = signal_covar(signals, core, mean_all)
        return signal1,core
    else:
        core = core_test ** (-1 / 2)
        signal1 = signal_covar(signals, core, mean_all)
        return signal1

#对输入的数组进行协方差，并返回同纬度结果[n,84,q,960,64] -> [n,84,q,64,960] -> [n,84,q,64,64]
        # [n, 84, 960, 64] -> [n, 84, 64, 960] -> [n, 84, 64, 64]
def signal_covar(signal,core, mean_all):
    if mean_all:
        signal = np.transpose(signal, axes=[0,  1, 2, 4, 3])
        signals = np.zeros((signal.shape[0:4])+(64,))
        for i in range(len(signal)):
            for j in range(signal.shape[1]):
                signal1 = pyriemann.estimation.Covariances().fit_transform(signal[i, j, :])
                signal2 = core * signal1 * core
                signals[i, j, :] = np.log(signal2)
        return signals, core
    else:
        signal = np.transpose(signal, axes=[0, 1, 3, 2])
        signals = np.zeros((signal.shape[0:3]) + (64,))
        for i in range(len(signal)):
            signal1 = pyriemann.estimation.Covariances().fit_transform(signal[i, :])
            signal2 = core * signal1 * core
            signals[i, :] = np.log(signal2)

        return signals


#训练集高斯增强，在对数据进行频段划分
def AugmentG(X, y, augmulitple,train_idx, val_idx, test_idx):
    x_train, y_train = AddGussio(X[train_idx], y[train_idx], 0, 0.01, augmulitple=augmulitple)
    # x_train, y_train = n_classfilter(x_train, y_train, arg = True)
    x_val = X[val_idx]
    # x_val, y_val = n_classfilter(X[val_idx], y[val_idx], arg = False)
    y_val = y[val_idx]
    # x_train = np.transpose(x_train, axes=[1, 2, 3, 4, 5, 0])
    # y_train = np.transpose(y_train,axes=[1,2,3,0])
    # x_val = np.transpose(x_val,axes=[1,2,3,4,0])
    # y_val = np.transpose(y_val,axes=[1,2,0])
    x_test = X[test_idx]
    y_test = y[test_idx]

    return x_train, y_train, x_val, y_val, x_test, y_test

def AugmentAver(X, y, augmulitple,train_idx, val_idx, test_idx):
    x_train = X[train_idx]
    y_train = y[train_idx]
    x_val = X[val_idx]
    y_val = y[val_idx]
    x_test = X[test_idx]
    y_test = y[test_idx]

    xout = np.zeros((x_train.shape[:2]+(augmulitple,)+x_train.shape[-2:]))
    yout = np.zeros((x_train.shape[:2]+(augmulitple,)))
    for i in range(len(x_train)):
        for j in range(x_train.shape[1]):
            k = 0
            while k < augmulitple:
                subject = np.random.choice(len(x_train))
                trial = np.random.randint(0, x_train.shape[1])
                if y_train[i,j] == y_train[subject,trial]:
                    xout[i, j, k, :] = (x_train[i,j,:] + x_train[subject,trial,:])/2.
                    yout[i, j, k] = y_train[i, j]
                    k = k+1

    return xout, yout, x_val, y_val, x_test, y_test

def AddGussio(x,y,sigam,mu,augmulitple):
    # x = np.squeeze(x)
    signal = np.zeros((len(x),) + (x.shape[1],) + (augmulitple,) + x.shape[-2:])
    labels=np.zeros((len(x),)+((x.shape[1]),) + (augmulitple,))

    for i in range(augmulitple):
        # TODO：样本点进行增强（960， 64），加两个for循环

        for j in range(len(x)):
            for k in range(x.shape[1]):
                x1 = np.random.normal(loc=sigam, scale=mu, size=(960,64))
                # ix = i*int(x.shape[1]) +k
                signal[j,k,i, :,:] = x[j,k,:,:] + x1
                labels[j,k,i] = y[j,k]

    return signal,labels

def crossval_gen(X, y,  batch_size,nsample):
    """
    Generator that produces training batches in an infinite loop by
    randomly selecting them from the training data, normalizing them,
    and adding a little noise
    """

    while True:
        # X = X.reshape((X.shape+(1,)))
        Xout = np.zeros((batch_size,nsample,64))
        yout = np.zeros((batch_size))

        for i in range(batch_size):
            # randomly choose subject and trial
            subject = np.random.choice(len(X))
            trial = np.random.randint(0, X.shape[1])
            # Xout[i,:] = X[subject, trial, :,:]
            augment = np.random.randint(0, X.shape[2])
            x = X[subject, trial, augment, :, :]
            mu = x.mean(0).reshape(1, x.shape[-1])
            sigma = np.maximum(x.std(0).reshape(1, x.shape[-1]), 1e-10)
            #

            add_noise = NOISE_LEVEL * np.random.randn(X.shape[-2], X.shape[-1])

            Xout[i, :, :] = (x - mu) / sigma + add_noise

            # yout[i, :] = y[subject, trial, :]

            yout[i] = y[subject, trial, augment]
        # yout = yout.reshape((-1))
        yout1 = to_categorical(yout, 4)
        Xout1 = Xout.reshape((-1,nsample,64,1))
        yield Xout1, yout1


def crossval_test(X, y):
    """
    Prepares a test set of (X, y) with the subjects included in test_idx.

    flatten: if True output shape is (N, seg_length, N_channels), otherwise
    output shape is (N_subjects, N_trials, seg_length, N_channels) for
    per-subject validation.
    fix_offset: Set the offset of the segment from the start of a trial to
    a selected value. Otherwise samples start at the cue.
    """
    ntrials = X.shape[1]
    preshape = (len(X) * ntrials,)
    Xout = np.zeros(preshape + X.shape[-2:])
    yout = np.zeros(preshape)
    for i in range(len(X)):
        for j in range(X.shape[1]):
            trial = X[i, j, :, :]
            # normalize based on per-channel mean
            x = trial
            mu = x.mean(0).reshape((1,) + x.shape[1:])
            sigma = np.maximum(x.std(0).reshape((1,) + x.shape[1:]), 1e-10)

            out = (x - mu) / sigma
            ix = i * ntrials + j
            Xout[ix, :, :] = out
            yout[ix] = y[i, j]
    Xout = Xout.reshape(Xout.shape+(1,))
    yout = to_categorical(yout,4)

    return Xout, yout


def crossval_test_fi(X, y):
    """
    Prepares a test set of (X, y) with the subjects included in test_idx.

    flatten: if True output shape is (N, seg_length, N_channels), otherwise
    output shape is (N_subjects, N_trials, seg_length, N_channels) for
    per-subject validation.
    fix_offset: Set the offset of the segment from the start of a trial to
    a selected value. Otherwise samples start at the cue.
    """
    ntrials = X.shape[1]
    preshape = (len(X) * ntrials,)
    Xout = np.zeros(preshape + X.shape[-2:])
    yout = np.zeros(preshape)
    for i in range(len(X)):
        for j in range(X.shape[1]):
            trial = X[i, j, :, :]
            # normalize based on per-channel mean
            x = trial
            mu = x.mean(0).reshape((1,) + x.shape[1:])
            sigma = np.maximum(x.std(0).reshape((1,) + x.shape[1:]), 1e-10)

            out = (x - mu) / sigma
            ix = i * ntrials + j
            Xout[ix, :, :] = out
            yout[ix] = y[i, j]
    Xout = Xout.reshape(Xout.shape+(1,))
    yout = to_categorical(yout,4)

    return Xout, yout