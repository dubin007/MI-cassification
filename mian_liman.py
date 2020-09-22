import nndata
import numpy as np
import riemannian
from keras.utils.np_utils import *
import  tensorflow as tf
import  os
import pyriemann
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import riemannian

def data_split(X, y, augmulitple,train_idx, val_idx):
    x = np.zeros((5, len(train_idx)*84*augmulitple, 64, 64))
    y1 = np.zeros((5, len(train_idx)*84*augmulitple))
    x_val = np.zeros((5, len(val_idx)*84, 64, 64))
    y1_val = np.zeros((5, len(val_idx)*84))
    for j in range(len(X)):
        x[j, :], y1[j, :], x_val[j, :], y1_val[j, :] = nndata.signal_propcess(X[j], y[j], train_idx, val_idx,augmulitple)
        print("yes")
    x_train = x.transpose(0,1,2,3)
    # x_val = x_val.transpose(1,2,3,0)
    y_train = y1[0]
    y_val = y1_val[0]

    return x_train, y_train, x_val,y_val

def eval_network(label_val, pred_val):
    plt.hist(label_val.flatten(), bins=1000)
    #plt.show()

    unique, counts = np.unique(label_val, return_counts=True)
    print("Labels: ", unique, counts)
    print(label_val)
    unique, counts = np.unique(pred_val, return_counts=True)
    print("Predicted: ", unique, counts)
    print(pred_val)

    conf_mat = confusion_matrix(label_val, pred_val)
    print(conf_mat)
    tru_pos, prec_i, recall_i = [], [], []
    for i in range(conf_mat.shape[0]):
        tru_pos.append(conf_mat[i, i])
        prec_i.append(conf_mat[i, i]/np.sum(conf_mat[:, i]).astype(float))
        recall_i.append(conf_mat[i, i]/np.sum(conf_mat[i, :]).astype(float))

    accuracy_val = np.sum(tru_pos).astype(float) / (np.sum(conf_mat)).astype(float)
    print("accuracy: {}".format(accuracy_val))

    precision_tot = np.sum(prec_i)/conf_mat.shape[0]
    print("total precision: {}".format(precision_tot))

    precision_cc = np.sum(prec_i[1:]) / (conf_mat.shape[0]-1)
    print("control class precision: {}".format(precision_cc))

    recall_tot = np.sum(recall_i) / conf_mat.shape[0]
    print("total recall: {}".format(recall_tot))

    recall_cc = np.sum(recall_i[1:]) / (conf_mat.shape[0] - 1)
    print("control class recall: {}".format(recall_cc))

    print("# # # # # # # # # # # # # # # # # # # # # # #")
    print(" ")
    print("# # # # # # # # # # # # # # # # # # # # # # #")

    return accuracy_val


def main():
    electrodes = list(range(64))
    augmulitple = 5
    epochs = 150
    batch_size = 200
    # nclasses = [2, 3, 4]
    nclasses = [4]
    # splits = list(range(5))
    splits = [0]

    # results = np.zeros((len(nclasses), len(splits), 4, epochs ))
    for j, nclasses in enumerate(nclasses):
        try:
            del X, y
        except:
            pass
        X, y = nndata.load_raw_data(electrodes=electrodes, num_classes=nclasses)
        # X, y = nndata.n_classfilter(X, y)
        for i in enumerate(splits):
            idx = list(range(X.shape[1]))
            train_idx, val_idx, test_idx = nndata.split_idx(idx, a=6, b=2)
            x_train, y_train, x_val, y_val = data_split(X, y, augmulitple, train_idx, val_idx)

            for i in range(len(X)):
                fgda = pyriemann.tangentspace.FGDA()
                mdm = pyriemann.classification.MDM()
                clf = Pipeline([('FGDA', fgda), ('MDM', mdm)])
                print('Data Train Length: {}'.format(len(x_train[i])))
                #
                # b = np.linalg.eigvals(x_train[i])
                # if np.all(b > 0):
                #     print("is positive")
                # else:
                #     print("not positive")
                #
                # b1 = np.linalg.eigvals(x_val[i])
                # if np.all(b1 > 0):
                #     print("is positive")
                # else:
                #     print("not positive")

                clf.fit_transform(x_train[i], y)
                print("training data: ")
                pred_train = clf.predict(x_val[i])
                a = eval_network(y_val, pred_train)
                print("train time:")
                print(i)

                print("i can do it")
#
# #
# x1 = np.random.randint(1,5,(7,840,64,960))
# x = riemannian.Signals_Covariance(x1)
# x = x.reshape(-1,64,64)
# y= np.random.randint(0,4,(7*840))
# x2 = np.random.randint(1,5,(2,84,64,960))
# # batch_size = 128
# x_val = riemannian.Signals_Covariance(x2)
# x_val = x_val.reshape(-1,64,64)
# y_val = np.random.randint(0,4,(2*84))
# #
# fgda = pyriemann.tangentspace.FGDA()
# mdm = pyriemann.classification.MDM()
# clf = Pipeline([('FGDA', fgda), ('MDM', mdm)])
# print('Data Train Length: {}'.format(len(x)))
# clf.fit_transform(x, y)
# print("training data: ")
# pred_train = clf.predict(x_val)
# eval_network(y_val, pred_train)







if __name__ == '__main__':
    main()
