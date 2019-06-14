import numpy as np
import math as math
import pandas as pd
# dataset = np.array([[3,5,1,4,1],
#        [4,4,3,5,3],
#        [3,4,4,4,4],
#        [3,3,5,2,1],
#        [3,4,5,4,3],
#     [3,4,5,4,5],
#  [3,4,5,4,5],
#   [1,1,1,3,1]
# ])
# dataset = pd.read_excel('data222.xls',encoding='gbk')

# dataset = pd.read_csv('data222.csv',encoding='gbk')
# dataset = dataset.drop("Unnamed: 0",axis=1).drop("no",axis=1)
# def corr(data):
#     return np.corrcoef(dataset)
#
# dataset_corr = corr(dataset)#Pearson积矩相关系数#PPMCC Pearson‘s r #数据标准化？
# aaa = pd.DataFrame(dataset_corr)
# aaa.to_csv('wanted_dataset_corr.csv')
# def kmo(dataset_corr):
#     corr_inv = np.linalg.inv(dataset_corr)#矩阵求逆
#     nrow_inv_corr, ncol_inv_corr = dataset_corr.shape
#     A = np.ones((nrow_inv_corr,ncol_inv_corr))
#     for i in range(0,nrow_inv_corr,1):
#         for j in range(i,ncol_inv_corr,1):
#             A[i,j] = -(corr_inv[i,j])/(math.sqrt(corr_inv[i,i]*corr_inv[j,j]))
#             A[j,i] = A[i,j]
#     dataset_corr = np.asarray(dataset_corr)
#     kmo_num = np.sum(np.square(dataset_corr)) - np.sum(np.square(np.diagonal(A)))
#     kmo_denom = kmo_num + np.sum(np.square(A)) - np.sum(np.square(np.diagonal(A)))
#     kmo_value = kmo_num / kmo_denom
#     return kmo_value
# #dataset_corr.to_excel("d.xls")
# print(kmo(dataset_corr))

import numpy as np
import math as math

dataset = pd.read_csv('data222.csv', encoding='gbk')
dataset = dataset.drop("Unnamed: 0", axis=1)


def corr(data):
    return np.corrcoef(dataset)


dataset_corr = corr(dataset)
tru = pd.read_csv('true.csv', encoding='gbk')


def kmo(dataset_corr, tr):
    corr_inv = tr
    nrow_inv_corr, ncol_inv_corr = dataset_corr.shape
    A = np.ones((nrow_inv_corr, ncol_inv_corr))
    for i in range(0, nrow_inv_corr, 1):
        for j in range(i, ncol_inv_corr, 1):
            A[i, j] = -(corr_inv.iloc[i, j]) / (math.sqrt(corr_inv.iloc[i, i] * corr_inv.iloc[j, j]))
            A[j, i] = A[i, j]
    dataset_corr = np.asarray(dataset_corr)
    kmo_num = np.sum(np.square(dataset_corr)) - np.sum(np.square(np.diagonal(A)))
    kmo_denom = kmo_num + np.sum(np.square(A)) - np.sum(np.square(np.diagonal(A)))
    kmo_value = kmo_num / kmo_denom
    return kmo_value


print(kmo(dataset_corr, tru)) # kmo test
