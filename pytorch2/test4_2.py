import torch
from matplotlib import pyplot as plt
import numpy as np


def sigmoid(inX):
    res = 1 / (1 + torch.exp(-inX))
    return res


def train(X, Y, V, G, W, T, Eta):
    #  前向计算
    b = sigmoid(torch.mm(X, V) + G)
    y_out = sigmoid(torch.mm(b, W) + T)  # 计算当前样本的输出
    Y_loss = torch.sum(torch.pow(y_out - Y, 2)) / 2  # 损失函数：均方误差

    gj = - (y_out - Y) * y_out * (1 - y_out)  # 计算输出层神经元的梯度项
    G_W = Eta * torch.transpose(torch.mm(torch.transpose(gj, 0, 1), b), 0, 1)
    G_T = - torch.sum(Eta * gj, 0)
    e = b * (1 - b) * torch.mm(gj, torch.transpose(W, 0, 1))  # 计算隐层神经元的梯度项
    G_V = Eta * torch.transpose(torch.mm(torch.transpose(e, 0, 1), X), 0, 1)
    G_G = - torch.sum(Eta * e, 0)
    return Y_loss, G_V, G_G, G_W, G_T


# 步长
eta = 0.1
# 每次迭代后计算损失函数值，以备画图使用
loss = np.zeros(500)

m, d, q, l = 32, 20, 10, 2  # m样本数；d输入维度（属性数）；q隐层神经元维度；l输出神经元维度

# 输入数据和输出数据
x = torch.randn(m, d)
y = torch.randn(m, l)

# 两层权重，以及初始化
v = torch.randn(d, q)  # 第一层
gama = torch.randn(q)
w = torch.randn(q, l)  # 第二层
theta = torch.randn(l)

# BP学习算法
for t in range(500):
    y_loss, grad_v, grad_gama, grad_w, grad_theta = train(x, y, v, gama, w, theta, eta)
    loss[t] = y_loss

    # 更新连接权和阈值
    w += grad_w
    theta += grad_theta
    v += grad_v
    gama += grad_gama

plt.figure()
plt.plot(loss)
plt.xlabel('Number of iterations')
plt.ylabel('Mean square error loss')
plt.show()
