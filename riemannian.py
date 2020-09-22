# import numpy as np
#
# import pyriemann
# import time
# import matplotlib.pyplot as plt
# from sklearn.pipeline import Pipeline
# from sklearn import covariance
# from pyriemann.utils.mean import mean_covariance, mean_riemann
# from pyriemann.utils.distance import distance
# import pandas
#
#
#
# def Signals_Covariance(signals,core_test,mean_all=True):
#
#     signals = signals.reshape((-1,)+(signals.shape[-2:]))
#     signals = np.transpose(signals, axes=[0, 2,1])
#     x_out = pyriemann.estimation.Covariances().fit_transform(signals)
#     if mean_all:
#         core = mean_covariance(x_out, metric='riemann')
#         # core = training_data_cov_means(X,y,num_classes=4)
#         core = core ** (-1 / 2)
#         signal = core * x_out * core
#         signal1 = np.log(signal)
#         return signal1,core
#     else:
#         core = core_test ** (-1 / 2)
#         signal = core * x_out * core
#         signal1 = np.log(signal)
#         return signal1
#
# # def TangentSpaceMapping(X):
# #
# #     core = mean_covariance(X, metric='riemann')
# #     # core = training_data_cov_means(X,y,num_classes=4)
# #     core = core ** (-1/2)
# #     signal = core * X * core
# #     signal1 = np.log(signal)
# #     return  signal1
#
#
# #输入数据和标签，输出各标签对应的黎曼均值点以及各标签的个数
# # def training_data_cov_means(X, y, num_classes=4):
# #
# #     X = X.reshape((-1,)+ X.shape[-2:])
# #     # b1 = np.linalg.eigvals(X)
# #     # if np.all(b1 > 0):
# #     #     print("is positive")
# #     # else:
# #     #     print("not positive")
# #
# #     mean_all = mean_covariance(X, metric='riemann')
# #
# #     # todo 可以考虑按照所有的被试的样本，根据每个类别就按黎曼中心点
# #
# #     return mean_all
#
