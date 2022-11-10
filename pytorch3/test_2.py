# 使用pytorch，编写全连接神经网络，实现对MNIST手写体数据集的识别。
# 要求每一轮训练后给出测试集识别的准确率。
# 训练结束后给出测试集准确率并随机展示16个样本，输出他们的识别结果。
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image

# 对图片进行多个操作时，可以通过Compose将这些操作拼接起来
transform = T.Compose([T.ToTensor(),  # 图片加载成tensor
                       T.Normalize((0.1307,), (0.3081,))  # 归一化
                       ])
dataset_mnist = datasets.MNIST('data/', download=True, train=False, transform=transform)  # 加载数据
dataloader_mnist = DataLoader(dataset_mnist, shuffle=True, batch_size=16)  # 图片分批
dataiter_mnist = iter(dataloader_mnist)  # 可迭代对象
# images_data, labels = next(dataiter_mnist)
# img = make_grid(images_data, 4)  # 4*4拼接图片
# save_image(img, 'a.png')  # 将tensor保存为图片
# Image.open('a.png')

# 定义一个卷积神经网络
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # 卷积：提取特征
        self.pool = nn.MaxPool2d(2, 2)  # 池化：下采样
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 全连接层
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):  # 前馈
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)  # 拉平，自动计算维度
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

# 定义一个损失函数和优化器
import torch.optim as optim


criterion = nn.CrossEntropyLoss()  # 交叉熵
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # SGD优化 lr：学习率

# 训练
for epoch in range(3):  # 所有训练数据轮两轮
    running_loss = 0.0
    for i, data in enumerate(dataloader_mnist, 0):
        inputs, labels = data
        optimizer.zero_grad()  # 梯度初始化为0
        # 前向+后向+优化
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # print('[%d,%d] loss:%.3f' %
        #           (epoch + 1, i + 1, running_loss))
        running_loss = 0.0

    # 测试
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in dataloader_mnist:
            outputs = net(data)
            _, predicted = torch.max(outputs.data, 1)  # 下划线表示每行概率值中最大的，1表示输出所在行最大的index
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images:%d %%' % (100 * correct / total))
print('Finished Training')

# 测试
correct = 0
total = 0
with torch.no_grad():
    for data, labels in dataloader_mnist:
        outputs = net(data)
        _, predicted = torch.max(outputs.data, 1)  # 下划线表示每行概率值中最大的，1表示输出所在行最大的index
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images:%d %%' % (100*correct/total))

images_data, labels = next(dataiter_mnist)
img = make_grid(images_data, 4)  # 4*4拼接图片
save_image(img, 'a.png')  # 将tensor保存为图片
outputs = net(images_data)
_, predicted = torch.max(outputs.data, 1)
print(predicted)

