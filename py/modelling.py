import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as nlg

#第一步：读数据
mydata = pd.read_csv('data222.csv',encoding="gb2312")
print("所读取的数据：")
print(mydata)
# 第二步：去除无用数据
mydata=mydata.drop(['no','Unnamed: 0'],axis=1)
#第三步：计算相关矩阵R
R=mydata.corr()   #求相关性矩阵的方法
print("样本相关性矩阵：")
print(R)
#第四步：求R的特征值和标准化特征值向量
#求R的特征值和特征向量
eig_value, eigvector = nlg.eig(R)
eig = pd.DataFrame()
eig['names'] = mydata.columns
eig['eig_value'] = eig_value
#特征值从大到小排序
eig.sort_values('eig_value', ascending=False, inplace=True)
print("特征值：")
print(eig_value)
#print("特征向量：")
#print(eigvector)
#第五步：寻找公共因子个数m
print("公因子个数：")
for m in range(1, 14):
    # 前m个特征值的比重大于85%的标准
    if eig['eig_value'][:m].sum() / eig['eig_value'].sum() >= 0.85:
        print(m)
        break
# 第六步：求因子模型的因子载荷阵
A = np.zeros((14,m))
A[:,0] = math.sqrt(eig_value[0]) * eigvector[:,0]
A[:,1] = math.sqrt(eig_value[1]) * eigvector[:,1]
A[:,2] = math.sqrt(eig_value[2]) * eigvector[:,2]
A[:,3] = math.sqrt(eig_value[2]) * eigvector[:,3]
a = pd.DataFrame(A)
a.columns = ['factor1', 'factor2', 'factor3','factor4']
print("因子载荷矩阵（成分矩阵）：")
print(a)
#第七步：求共同度以及特殊因子方差
h=np.zeros(14)
D=np.mat(np.eye(14))
b=np.mat(np.zeros((4,14)))
for i in range(14):
    b=A[i,:]*A[i,:].T   #.T 转置
    h[i]=b[0]
    D[i,i] = 1-b[0]
print("共同度(每个因子对公共因子的依赖程度)：")
print(h)
print("特殊因子方差：")
print(pd.DataFrame(D))
#第八步：求累计方差贡献率
m=np.zeros(4)
for i in range(4):
    c=A[:,i].T *A[:,i]
    m[i]=c[0]
print("贡献度（每个公共因子对所有因子的影响：")
print(m)