import numpy as np
from matplotlib import pyplot as plt
from numpy import random


def sigmoid(inX):
    res = np.zeros(inX.shape)
    for i in range(inX.shape[0]):
        if inX[i] >= 0:
            res[i] = 1 / (1 + np.exp(-inX[i]))
        else:
            res[i] = np.exp(inX[i]) / (1 + np.exp(inX[i]))
    return res


# 使用梯度下降法 训练数据 得到最佳参数
def gradDscent(data, label):
    data = np.mat(data)  # ?
    m, n = data.shape
    alpha = 0.001  # 迭代步长/学习率
    maxItNum = 500  # 最大迭代次数
    weights = np.ones((n, 1))  # 初始化参数为全1，是列向量
    for i in range(maxItNum):
        grad = -((label - sigmoid(data * weights)).transpose() * data).transpose()
        weights = weights - alpha * grad
    return weights


a = random.normal(loc=1, scale=2, size=(500, 2))  # 类别为0，蓝色表示
b = random.normal(loc=7, scale=2, size=(500, 2))  # 类别为1，红色表示
e = np.ones([1000, 1])
X = np.hstack(( np.vstack((a, b)), e))
c = np.zeros([500, 1])
d = np.ones([500, 1])
labels = np.vstack((c, d))

w = gradDscent(X, labels)
print(w)

weights = w.A
x = np.arange(-5, 15, 0.1)
y = -(weights[0] * x + weights[2]) / weights[1]
plt.plot(a[:, 0], a[:, 1], 'o', c='blue')
plt.plot(b[:, 0], b[:, 1], 'o', c='red')
plt.plot(x, y, c='black')
plt.show()
