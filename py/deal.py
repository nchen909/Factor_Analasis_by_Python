#不可一气呵成跑完！中间需要对表格调整 分块执行代码

import numpy as np
from numpy import *
import pandas as pd

df = pd.read_csv('data.csv',encoding='gbk')

#数据清洗 先用EPS平台对所需数据调整后导入，故不存在错误数据、多余数据与重复数据，故只简化表格与缺失值处理
df=df.dropna(how="all")
df=df.drop([0])#delete year
#for i in range(df.shape[0]):
#由于所分析问题不针对具体地区，找出缺失值大于1的行并删除
todel=[]
for i in range(df.shape[0]):
    sum = 0
    for j in range(df.shape[1]):
        if pd.isnull(df.iloc[i,j]):
            sum+=1
        if sum>=2:
            todel.append(i)
            break
df=df.drop(todel)

#拉格朗日乘子法作缺失值处理
from scipy.interpolate import lagrange
def ploy(s,n,k=6):
    y=s[list(range(n-k,n))+list(range(n+1,n+1+k))]#取数
    y=y[y.notnull()]
    return lagrange(y.index,list(y))(n)
for i in df.columns:
    for j in range(len(df)):
        if (df[i].isnull())[j]:
            df[i][j]=ploy(df[i],j)
df.to_excel('data222.xls')

#利用KMO检验与Bartlett检验判断因子分析法是否合适
import numpy as np
import math as math

dataset = pd.read_csv('data222.csv', encoding='gbk')
dataset = dataset.drop(['no','Unnamed: 0'],axis=1)


def corr(data):
    return np.corrcoef(dataset)


dataset_corr = corr(dataset)#Pearson's r Pearson积矩相关系数#数据标准化
tru = pd.read_csv('true.csv', encoding='gbk')#由于精度问题求逆需要在matlab中求完导入


def kmo(dataset_corr, tr):
    corr_inv = tr#这原先用np.linalg.inv求逆 但是由于精度问题导致结果出错 故matlab算完后导入
    nrow_inv_corr, ncol_inv_corr = dataset_corr.shape
    A = np.ones((nrow_inv_corr, ncol_inv_corr))#全1矩阵
    for i in range(0, nrow_inv_corr, 1):
        for j in range(i, ncol_inv_corr, 1):
            A[i, j] = -(corr_inv.iloc[i, j]) / (math.sqrt(corr_inv.iloc[i, i] * corr_inv.iloc[j, j]))
            A[j, i] = A[i, j]
    dataset_corr = np.asarray(dataset_corr)
    kmo_num = np.sum(np.square(dataset_corr)) - np.sum(np.square(np.diagonal(A)))#相关系数阵平方和与对角阵平方和的差
    kmo_denom = kmo_num + np.sum(np.square(A)) - np.sum(np.square(np.diagonal(A)))
    kmo_value = kmo_num / kmo_denom
    return kmo_value


print(kmo(dataset_corr, tru)) # kmo test

dataset = pd.read_excel('data222.xls',encoding='gbk')
dataset = dataset.drop(['no','Unnamed: 0'],axis=1)

def corr(data):
    return np.corrcoef(dataset)

dataset_corr = corr(dataset)

from scipy.stats import bartlett
bartlett(dataset_corr[0],dataset_corr[1],dataset_corr[2],dataset_corr[3],dataset_corr[4],\
dataset_corr[6],dataset_corr[7],dataset_corr[8],dataset_corr[9],dataset_corr[10],dataset_corr[11],dataset_corr[12]\
,dataset_corr[13],dataset_corr[14],dataset_corr[15],dataset_corr[16],dataset_corr[17],dataset_corr[18],dataset_corr[19]\
,dataset_corr[20],dataset_corr[21],dataset_corr[22],dataset_corr[23],dataset_corr[24],dataset_corr[25],dataset_corr[26]\
,dataset_corr[27],dataset_corr[28],dataset_corr[29])#bartlett test

#not use factor_analyzer库 纯按原理写
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as nlg

#读数据
mydata = pd.read_csv('data222.csv',encoding="gb2312")
# 去除无用数据
mydata=mydata.drop(['no','Unnamed: 0'],axis=1)
#计算相关矩阵R
R=mydata.corr()   #求相关性矩阵的方法
print("样本相关性矩阵：")
print(R)
#求R的特征值和标准化特征值向量
eig_value, eigvector = nlg.eig(R)
eig = pd.DataFrame()
eig['names'] = mydata.columns
eig['eig_value'] = eig_value
#特征值从大到小排序
eig.sort_values('eig_value', ascending=False, inplace=True)
print("特征值：")
print(eig_value)
# print("特征向量：")
# print(eigvector)
#寻找公共因子个数m
print("公因子个数：")
for m in range(1, 14):
    # 前m个特征值的比重大于85%的标准
    if eig['eig_value'][:m].sum() / eig['eig_value'].sum() >= 0.85:
        print(m)
        break
# 求因子模型的因子载荷阵
A = np.zeros((14,m))
A[:,0] = math.sqrt(eig_value[0]) * eigvector[:,0]
A[:,1] = math.sqrt(eig_value[1]) * eigvector[:,1]
A[:,2] = math.sqrt(eig_value[2]) * eigvector[:,2]
A[:,3] = math.sqrt(eig_value[2]) * eigvector[:,3]
a = pd.DataFrame(A)
a.columns = ['factor1', 'factor2', 'factor3','factor4']
print("因子载荷矩阵（成分矩阵）：")
print(a)
#求共同度以及特殊因子方差
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
#求累计方差贡献率
m=np.zeros(4)
for i in range(4):
    c=A[:,i].T *A[:,i]
    m[i]=c[0]
print("贡献度（每个公共因子对所有因子的影响：")
print(m)

#use factor_analyzer库
import pandas as pd
import numpy as np
from pandas import DataFrame,Series
from factor_analyzer import FactorAnalyzer
#读数据
data = pd.read_csv('data222.csv',encoding="gb2312")
#去除无用数据
data=data.drop(['no','Unnamed: 0'],axis=1)
#data.head()

fa = FactorAnalyzer()
fa.analyze(data, 4, rotation=None)#固定公共因子个数为4个
print("公因子方差:\n", fa.get_communalities())#公因子方差
print("\n成分矩阵:\n", fa.loadings)#成分矩阵
var = fa.get_factor_variance()#给出贡献率
print("\n特征值,解释的总方差（即贡献率）,累积率:\n", var)

fa_score = fa.get_scores(data)#因子得分
print("\n因子得分:\n",fa_score)#.head()

#将各因子乘上他们的贡献率除以总的贡献率,得到因子得分中间值
a = (fa.get_scores(data)*var.values[1])/var.values[-1][-1]
print("\n",fa.get_scores(data),"\n")
print("\n",var.values[1],"\n")
print("\n",var.values[-1][-1],"\n")
print("\n",a,"\n")
#将各因子得分中间值相加，得到综合得分
a['score'] = a.apply(lambda x: x.sum(), axis=1)
#a.head()
print("\n综合得分:\n",a)

from pyecharts import Geo
import pandas as pd
df = pd.read_csv('ditu.csv',encoding="gb2312")
data = [(df.iloc[i][0], df.iloc[i][1]) for i in range(df.shape[0])]
geo = Geo("幸福指数评分", title_color="#fff",
          title_pos="center", width=1000,
          height=600, background_color='#404a59')
attr, value = geo.cast(data)
geo.add("", attr, value, visual_range=[-1.31,1.71], maptype='china', visual_text_color="#fff",
        is_piecewise=True,symbol_size=15, is_visualmap=True)
geo.render("happiness.html")  # 生成html文件