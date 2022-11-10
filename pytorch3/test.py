# 使用pytorch，编写全连接神经网络，实现对MNIST手写体数据集的识别。
# 要求每一轮训练后给出测试集识别的准确率。
# 训练结束后给出测试集准确率并随机展示16个样本，输出他们的识别结果。

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 20

tranform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

from torch.utils.data import DataLoader

train_data = datasets.MNIST(root="./MNIST",
                            train=True,
                            transform=tranform,
                            download=False)

test_data = datasets.MNIST(root="./MNIST",
                           train=True,
                           transform=tranform,
                           download=False)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)


class Digit(nn.Module):  # 继承父类
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)  # 二维卷积、输入通道，输出通道，5*5 kernel
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(20 * 10 * 10, 500)  # 全连接层，输入通道， 输出通道
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):  # 前馈
        input_size = x.size(0)  # 得到batch_size
        x = self.conv1(x)  # 输入：batch*1*28*28, 输出：batch*10*24*24(28-5+1)
        x = F.relu(x)  # 使表达能力更强大的激活函数, 输出batch*10*24*24
        x = F.max_pool2d(x, 2, 2)  # 最大池化层，输入batch*10*24*24，输出batch*10*12*12

        x = self.conv2(x)  # 输入batch*10*12*12，输出batch*20*10*10
        x = F.relu(x)

        x = x.view(input_size, -1)  # 拉平， 自动计算维度，20*10*10= 2000

        x = self.fc1(x)  # 输入batch*2000,输出batch*500
        x = F.relu(x)

        x = self.fc2(x)  # 输入batch*500 输出batch*10

        output = F.log_softmax(x, dim=1)  # 计算分类后每个数字的概率值

        return output


model = Digit().to(DEVICE)

optimizer = optim.Adam(model.parameters())


def train_model(model, device, train_loader, optimizer, epoch):
    model.train()  # PyTorch提供的训练方法
    for batch_index, (data, label) in enumerate(train_loader):
        # 部署到DEVICE
        data, label = data.to(device), label.to(device)
        # 梯度初始化为0
        optimizer.zero_grad()
        # 训练后的结果
        output = model(data)
        # 计算损失（针对多分类任务交叉熵，二分类用sigmoid）
        loss = F.cross_entropy(output, label)
        # 找到最大概率的下标
        pred = output.argmax(dim=1)
        # 反向传播Backpropagation
        loss.backward()
        # 参数的优化
        optimizer.step()
        if batch_index % 3000 == 0:
            print("Train Epoch : {} \t Loss : {:.6f}".format(epoch, loss.item()))


def test_model(model, device, test_loader):
    # 模型验证
    model.eval()
    # 统计正确率
    correct = 0.0
    # 测试损失
    test_loss = 0.0
    with torch.no_grad():  # 不计算梯度，不反向传播
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            # 测试数据
            output = model(data)
            # 计算测试损失
            test_loss += F.cross_entropy(output, label).item()
            # 找到概率值最大的下标
            pred = output.argmax(dim=1)
            # 累计正确率
            correct += pred.eq(label.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print("Test —— Average loss : {:.4f}, Accuracy : {:.3f}\n".format(test_loss,
                                                                          100.0 * correct / len(test_loader.dataset)))


for epoch in range(1, EPOCHS + 1):
    train_model(model, DEVICE, train_loader, optimizer, epoch)
    test_model(model, DEVICE, test_loader)
