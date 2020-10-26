# -*- coding: utf-8 -*-

'''
Major Code:
Name of Experiment: Lab1_Breast Cancer Diagnosis Based on Clustering Algorithms,
Author: Jiyue Jiang, Student ID: 17281093, Class ID: Medical Information 1707,
Supervised by Qiang Zhu, Beijing Jiaotong University,
Environment Configuration: Python3.7.7, PyCharm2019.3.3.
'''

# 导入模块
import numpy as np
import pandas as pd
import random
import math
import time
import os
import sys
import warnings
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from sklearn import preprocessing
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn import random_projection
from sklearn import manifold
from sklearn import decomposition

# 解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 去除警告
warnings.filterwarnings('ignore')


### 数据预处理
# 读取威斯康星州乳腺癌数据集
data = pd.read_csv('Wisconsin_Breast_Cancer_DataSet.data', sep=',',
                   names = ["id number", "1", "2", "3", "4", "5", "6", "7", "8", "9", "Class"])
# 展示数据
print("威斯康星州乳腺癌数据集行列数：", data.shape)
print("威斯康星州乳腺癌数据集展示：", data)
# 寻找缺失值
rows=[]
for row in range(data.shape[0]):
    for col in range(data.shape[1]):
        if data.iloc[row].iat[col] == "?":
            rows.append(row)
            break
print("打印缺失值的位置：", rows)
# 处理缺失值
data.drop(rows, axis = 0, inplace = True)
data = data.reset_index(drop = True)
print("缺失值处理后的威斯康星州乳腺癌数据集行列数：", data.shape)
print("缺失值处理后的威斯康星州乳腺癌数据集展示：", data)
# 去除去掉身份信息项目的一列
features = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
target = "Class"
D = np.array(data[features + [target]], dtype = np.int)
print("去掉身份信息后的威斯康星州乳腺癌数据集行列数：", D.shape)
print("去掉身份信息后的威斯康星州乳腺癌数据集展示：", D)


### ReliefF进行特征值提取
# ReliefF辅助函数
def GetRandSamples(D, D1, D2, k):
    r = math.ceil((D.shape[0] - 1) * random.random())
    R = D[r, :]
    d1 = np.zeros(D1.shape[0])
    d2 = np.zeros(D2.shape[0])
    for i in range(D1.shape[0]):
        d1[i] = np.sqrt(np.sum(np.square(R - D1[i, :])))
    for j in range(D2.shape[0]):
        d2[j] = np.sqrt(np.sum(np.square(R - D2[j, :])))
    L1 = np.argsort(d1)
    v1 = np.sort(d1)
    L2 = np.argsort(d2)
    v2 = np.sort(d2)
    if R[R.shape[0] - 1] == 2:
        H = D1[L1[1:k + 1], :]
        M = D2[L2[0:k], :]
    else:
        M = D1[L1[0:k], :]
        H = D2[L2[1:k + 1], :]
    Dh = np.zeros((H.shape[0], H.shape[1]))
    Dm = np.zeros((H.shape[0], H.shape[1]))
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            Dh[i][j] = abs(H[i][j] - R[j]) / 9
            Dm[i][j] = abs(M[i][j] - R[j]) / 9
    return R, Dh, Dm
# reliefF函数
def reliefF(D, m, k):
    row = D.shape[0]
    col = D.shape[1]
    D1 = np.zeros((0, col))
    D2 = np.zeros((0, col))
    for i in range(row):
        if D[i][col - 1] == 2:
            D1 = np.append(D1, [D[i]], axis=0)
        elif D[i][col - 1] == 4:
            D2 = np.append(D2, [D[i]], axis=0)
    W = np.zeros(col - 1)
    for i in range(m):
        R, Dh, Dm = GetRandSamples(D, D1, D2, k)
        for j in range(W.shape[0]):
            W[j] = W[j] - np.sum(Dh[:, j]) / (k * m) + np.sum(Dm[:, j]) / (k * m)
    return W
# 定义参数
#抽样次数
m = 80
#最邻近样本个数
k = 8
#运行次数
N = 20
W = np.zeros((N, D.shape[1] - 1))
for i in range(N):
    W[i]= reliefF(D, m, k)
print("打印权重：", W)
# 绘制权重图
x = [i for i in range(1,W.shape[1]+1)]
for i in range(N):
    plt.xlabel("Attribute Number")
    plt.ylabel("Feature Weight")
    plt.title("the ReliefF algorithm calculates the feature weight of breast cancer data")
    plt.plot(x, W[i])
result = np.zeros(W.shape[1])
for i in range(0,W.shape[1]):
    result[i] = np.sum(W[:,i])/W.shape[0]
print("打印result：", result)
# 平均权重值从小到大的索引值，即属性序号
L=np.argsort(result)+1
print("打印L：", L)
# 绘制特征权重平均值图
plt.xlabel("Attribute Number")
plt.ylabel("Feature Weight")
plt.title("the ReliefF algorithm calculated the average value of feature weight of breast cancer data")
plt.plot(x, result)
# 绘制每一种属性的特征权重变化图
x2 = [i for i in range(1, N + 1)]
name = ["块厚度", "细胞大小均匀性", "细胞形态均匀性", "边缘粘附力",
        "单上皮细胞尺寸", "裸核", "Bland染色质", "正常核仁", "核分裂"]
for i in range(0, W.shape[1]):
    plt.figure(figsize=(10,5))
    plt.xlabel("Number of Calculations")
    plt.ylabel("Feature Weight")
    plt.title(name[i] + "(attribute"+ str(i+1) + ") -- Feature Weight Change")
    plt.plot(x2, W[:, i])
    x_major_locator=MultipleLocator(1)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.show()


### K-Means聚类分析
# 控制部分系数
#从多少行开始进行预测分类
N0 = 0
# 所有数据行数
N1 = D.shape[0]
#只选择需要测试的数据
D = D[N0:N1,:]
print("打印只需要测试数据的行列数：", D.shape)
data1 = D[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]]
print("测试特征数据（去掉身份列和分类列）行列数：", data1.shape)
print("测试特征数据（去掉身份列和分类列）数据展示：", data1)
# 训练K-Means，聚成2类
model = KMeans(n_clusters=2)
model.fit(data1)
# 预测聚类结果
idx = model.predict(data1)
print("聚类数据结果行列数：", idx.shape)
print("聚类数据结果展示：", idx)
t = np.array(list(zip(D[:, 9], idx)))
print("t值行列数：", t.shape)
print("t值展示：", t)
d2 = t[t[:, 1]==1, :][:, 0]
print("提取原数据中属于第一类（1）的数据的最后一列的行列数：", d2.shape)
print("提取原数据中属于第一类（1）的数据的最后一列：", d2)
d4 = t[t[:, 1]==0, :][:, 0]
print("提取原数据中属于第二类（0）的数据的最后一列的行列数", d4.shape)
print("提取原数据中属于第二类（0）的数据的最后一列", d4)
a = np.sum(d2 ==2)
print("2（良性）的个数：", a)
b = a/d2.shape[0]
print("2（良性）的个数占d2的比例：", b)
# 调整部分参数
# 总的正确率
totalSum = 0
#第一类（1）的判断正确率，分类类别中的数据正确性
rate1 = 0
#第二类（0）的判断正确率
rate2 = 0
#说明第一类（1）属于良性
if b>0.5:
    totalSum = totalSum + a
    rate1 = b
    totalSum = totalSum + np.sum(d4 == 4)
    rate2 = np.sum(d4 == 4)/d4.shape[0]
# 说明第一类（1）属于恶性
else:
    totalSum = totalSum +  np.sum(d2 ==4)
    totalSum = totalSum +  np.sum(d4 ==2)
    rate2 =  np.sum(d2 == 4) / d2.shape[0]
    rate1 =  np.sum(d4 == 2) / d4.shape[0]
# 绘制属性值分布图
# 第一类横坐标数组
x1 = [i for i in range(1,np.sum(idx==1) + 1)]
# 第二类横坐标数组
x2 = [i for i in range(np.sum(idx == 1) + 1,
                       np.sum(idx == 1) + np.sum(idx == 0) + 1)]
name = ["块厚度", "细胞大小均匀性", "细胞形态均匀性", "边缘粘附力",
        "单上皮细胞尺寸", "裸核", "Bland染色质", "正常核仁", "核分裂"]
for i in range(data1.shape[1]):
    plt.figure(figsize=(6,5))
    plt.xlabel("Record Number")
    plt.ylabel("Attribute Value")
    plt.title(name[i] + "(attribute"+ str(i+1) + ") -- Value Distribution")
    mix = np.array(list(zip(data1[:, i], idx)))
    mix1 = mix[mix[:, mix.shape[1]-1]==1, :][:, 0]
    mix2 = mix[mix[:, mix.shape[1]-1]==0, :][:, 0]
    plt.scatter(x1, mix1, s = 5, color = "#ff7f00", label = "First Kind")
    plt.scatter(x2, mix2, s = 5, color = "#377eb8", label = "Second Kind")
    y_major_locator=MultipleLocator(1)
    ax=plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)
    plt.legend()
    plt.show()
# 总体的正确率
rate = totalSum/t.shape[0]
print("The overall accuracy is：{:f}".format(rate))
print("The accuracy of the first class is：{:f}".format(rate1))
print("The accuracy of the second class is：{:f}".format(rate2))
# 聚类中心
for i in range(model.cluster_centers_.shape[1]):
    print("attribute" + str(i+1) + "the first kind of clustering center：{0:.0f} the second kind of clustering center：{1:.0f}"
            .format(model.cluster_centers_[0][i],model.cluster_centers_[1][i]))
# 对单个属性的数据进行分类测试
for i in range(data1.shape[1]):
    data2 = D[:, i]
    data2 = data2.reshape(-1, 1)
    model = KMeans(n_clusters = 2)
    model.fit(data2)
    # 预测分类结果
    idx = model.predict(data2)
    t = np.array(list(zip(D[:, 9], idx)))
    d2 = t[t[:,1] == 1, :][:, 0]
    d4 = t[t[:,1] == 0, :][:, 0]
    a = np.sum(d2 == 2)
    b = a/d2.shape[0]
    print("预测结果展示：", b)
    # 调整参数
    # 总的正确率
    totalSum = 0
    # 第一类（1）的判断正确率，分类类别中的数据正确性
    rate1 = 0
    # 第二类（0）的判断正确率
    rate2 = 0
    if b>0.5:
        totalSum = totalSum + a
        rate1 = b
        totalSum = totalSum + np.sum(d4 == 4)
        rate2 = np.sum(d4 == 4)/d4.shape[0]
    else:
        totalSum = totalSum +  np.sum(d2 ==4)
        totalSum = totalSum +  np.sum(d4 ==2)
        rate2 =  np.sum(d2 ==4)/d2.shape[0]
        rate1 =  np.sum(d4 ==2)/d4.shape[0]
    rate = totalSum/t.shape[0]
    print("attribute" + str(i+1) + "The overall accuracy is：{0:.4f} the accuracy of the first class is：{1:.4f} the accuracy of the second class is：{2:.4f}"
            .format(rate, rate1, rate2))
# 绘制属性值分布图
data3 = D[:, 5]
data3 = data3.reshape(-1, 1)
model = KMeans(n_clusters=2)
model.fit(data3)
# 预测分类结果
idx = model.predict(data3)
x1 = [i for i in range(1,np.sum(idx==1) + 1)]
x2 = [i for i in range(np.sum(idx==1) + 1,np.sum(idx==1) + np.sum(idx==0) + 1)]
plt.figure(figsize=(6,5))
plt.xlabel("Record Number")
plt.ylabel("Attribute Value")
plt.title("Distribution of values for attribute 6")
mix = np.array(list(zip(data3, idx)))
mix1 = mix[ mix[:,mix.shape[1]-1]==1,:][:,0]
mix2 = mix[ mix[:,mix.shape[1]-1]==0,:][:,0]
plt.scatter(x1, mix1, s = 5, color = "#ff7f00", label="First Kind")
plt.scatter(x2, mix2, s = 5, color = "#377eb8", label="Second Kind")
y_major_locator=MultipleLocator(1)
ax=plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
plt.legend()
plt.show()
data4 = D[:, [0, 5]]
model = KMeans(n_clusters=2)
model.fit(data4)
idx = model.predict(data4)
t = np.array(list(zip(D[:, 9], idx)))
d2 = t[t[:,1]==1, :][:, 0]
d4 = t[t[:,1]==0, :][:, 0]
a = np.sum(d2 == 2)
b = a/d2.shape[0]
print("b数据展示：", b)
# 总的正确率
totalSum = 0
if b>0.5:
    totalSum = totalSum + a
    totalSum = totalSum + np.sum(d4 == 4)
else:
    totalSum = totalSum +  np.sum(d2 ==4)
    totalSum = totalSum +  np.sum(d4 ==2)
# 总体的正确率
rate = totalSum/t.shape[0]
print("Select attribute 6 and attribute 1, and the overall accuracy is：{:f}".format(rate))
# 选择属性6、1、8、3
data5 = D[:,[0,2,5,7]]
model = KMeans(n_clusters=2)
model.fit(data5)
idx = model.predict(data5)
t = np.array(list(zip(D[:,9], idx)))
d2 = t[t[:,1]==1,:][:,0]
d4 = t[t[:,1]==0,:][:,0]
a = np.sum(d2 == 2)
b = a/d2.shape[0]
print("b数据展示：", b)
# 总的正确率
totalSum = 0
if b>0.5:
    totalSum = totalSum + a
    totalSum = totalSum + np.sum(d4 == 4)
else:
    totalSum = totalSum +  np.sum(d2 ==4)
    totalSum = totalSum +  np.sum(d4 ==2)
# 总体的正确率
rate = totalSum/t.shape[0]
print("Select attributes 6, 1, 8, 3, the overall accuracy is：{:f}".format(rate))
# 选择属性6、1、8、3、2、4
data6 = D[:,[0,1,2,3,5,7]]
model = KMeans(n_clusters=2)
model.fit(data6)
idx = model.predict(data6)
t = np.array(list(zip(D[:,9], idx)))
d2 = t[t[:,1]==1, :][:, 0]
d4 = t[t[:,1]==0, :][:, 0]
a = np.sum(d2 == 2)
b = a/d2.shape[0]
print("b数据展示：", b)
# 总的正确率
totalSum = 0
if b>0.5:
    totalSum = totalSum + a
    totalSum = totalSum + np.sum(d4 == 4)
else:
    totalSum = totalSum +  np.sum(d2 ==4)
    totalSum = totalSum +  np.sum(d4 ==2)
# 总体的正确率
rate = totalSum/t.shape[0]
print("Select attributes 6, 1, 8, 3, the overall accuracy is：{:f}".format(rate))
# 选择属性6、1、8、3、2、4、5、7
data7 = D[:, [0, 1, 2, 3, 4, 5, 6, 7]]
model = KMeans(n_clusters=2)
model.fit(data7)
idx = model.predict(data7)
t = np.array(list(zip(D[:, 9], idx)))
d2 = t[t[:,1]==1, :][:, 0]
d4 = t[t[:,1]==0, :][:, 0]
a = np.sum(d2 == 2)
b = a/d2.shape[0]
print("b数据展示：", b)
# 总的正确率
totalSum = 0
if b>0.5:
    totalSum = totalSum + a
    totalSum = totalSum + np.sum(d4 == 4)
else:
    totalSum = totalSum +  np.sum(d2 ==4)
    totalSum = totalSum +  np.sum(d4 ==2)
# 总体的正确率
rate = totalSum/t.shape[0]
print("Select attributes 6, 1, 8, 3, the overall accuracy is：{:f}".format(rate))
data1 = D[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]]
model = KMeans(n_clusters = 3)
model.fit(data1)
idx = model.predict(data1)
print("idx数据展示", idx)
a = np.sum(idx == 0)
b = np.sum(idx == 1)
c = np.sum(idx == 2)
print("a, b, c展示：", a, b, c)
t = np.array(list(zip(D[:, 9], idx)))
d0 = t[t[:, 1]==0, :][:, 0]
d1 = t[t[:, 1]==1, :][:, 0]
d2 = t[t[:, 1]==2, :][:, 0]
print("d0, d1, d2展示：", d0, d1, d2)
rate0 = (np.sum(d0 == 2) / d0.shape[0]) * 100
rate1 = (np.sum(d1 == 4) / d1.shape[0]) * 100
rate2 = (np.sum(d2 == 4) / d2.shape[0]) * 100
print("第一类共{0}条数据，其中良性占{1:.2f}%".format(a,rate0))
print("第二类共{0}条数据，其中恶性占{1:.2f}%".format(b,rate1))
print("第三类共{0}条数据，其中恶性占{1:.2f}%".format(c,rate2))
data6 = D[:, [0, 1, 2, 3, 5, 7]]
model = KMeans(n_clusters = 3)
model.fit(data6)
idx = model.predict(data6)
t = np.array(list(zip(D[:, 9], idx)))
d0 = t[t[:, 1] == 0, :][:, 0]
d1 = t[t[:, 1] == 1, :][:, 0]
d2 = t[t[:, 1] == 2, :][:, 0]
print("d0, d1, d2展示：", d0, d1, d2)
rate0 = (np.sum(d0 == 2) / d0.shape[0]) * 100
rate1 = (np.sum(d1 == 4) / d1.shape[0]) * 100
rate2 = (np.sum(d2 == 4) / d2.shape[0]) * 100
print("第一类共{0}条数据，其中良性占{1:.2f}%".format(a,rate0))
print("第二类共{0}条数据，其中恶性占{1:.2f}%".format(b,rate1))
print("第三类共{0}条数据，其中恶性占{1:.2f}%".format(c,rate2))


