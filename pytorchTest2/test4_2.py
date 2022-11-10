# 使用pytorch，编写卷积神经网络（自选结构），实现对dogcat图像数据集的识别。
# 要求训练过程中给出训练集、测试集的损失函数、识别准确率。
# 模型训练结束后给出测试集识别准确率并随机展示16个样本，输出他们的识别结果。

import torch
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
# 载入图片
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image

transform = T.Compose([T.Resize(224),  # 保持长宽比不变，最短边为224
                       T.CenterCrop(224),  # 从图片中间切224*224
                       T.ToTensor(),  # 将图片转为Tensor，归一化
                       T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
                       ])

import torch
import os
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DogCat(torch.utils.data.Dataset):
    def __init__(self, root, tuansforms=None):
        root_cat = root + '/cats'  # 猫的文件夹路径
        root_dog = root + '/dogs'  # 狗的文件夹路径
        imgs_cat = os.listdir(root_cat)  # 读取文件夹下所以文件名
        imgs_dog = os.listdir(root_dog)
        self.imgs = [os.path.join(root_cat, img) for img in imgs_cat] + [os.path.join(root_dog, img) for img in
                                                                         imgs_dog]  # 完整文件名
        self.transforms = transform

    def __getitem__(self, index):
        img_path = self.imgs[index]  # 获得图片路径
        label = 1 if 'dog' in img_path.split('/')[-1] else 0  # 分类：如果为狗，值为1；为猫值为0
        pil_img = Image.open(img_path)  # 读取图片
        if self.transforms:
            data = self.transforms(pil_img)  # 图片归一化
        return data, label

    def __len__(self):
        return len(self.imgs)  # 图像个数即为长度


test_dataset = DogCat('./dog&cat/test_set0/test_set', tuansforms=transform)  # 调用
train_dataset = DogCat('./dog&cat/training_set0/training_set', tuansforms=transform)  # 调用

from torch.utils.data import DataLoader

test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=0, drop_last=False)  # 批量处理：每16张图片一批
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0,
                              drop_last=False)  # 批量处理：每16张图片一批
test_dataiter = iter(test_dataloader)
train_dataiter = iter(train_dataloader)

# 定义一个卷积神经网络
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()  # 16*3*224*224

        # self.conv1 = nn.Conv2d(3, 96, 11, 4, padding=0)
        # self.pool = nn.MaxPool2d(3, 2)
        # self.conv2 = nn.Conv2d(96, 256, 5, 1, padding=2)
        # self.conv3 = nn.Conv2d(256, 384, 3, 1, padding=1)
        # self.conv4 = nn.Conv2d(384, 384, 3, 1, padding=1)
        # self.conv5 = nn.Conv2d(384, 256, 3, 1, padding=0)
        #
        # self.fc1 = nn.Linear(4 * 4 * 256, 500)
        # self.fc2 = nn.Linear(500, 100)
        # self.fc3 = nn.Linear(100, 2)

        self.conv1 = nn.Conv2d(3, 6, 5)  # 卷积：提取特征
        self.pool = nn.MaxPool2d(2, 2)  # 池化：下采样
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)  # 全连接层
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):  # 前馈
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # x = self.pool(F.relu(self.conv5(x)))
        # x = x.view(x.size(0), -1)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)  # 拉平，自动计算维度
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net().to(DEVICE)

# 定义一个损失函数和优化器
import torch.optim as optim

criterion = nn.CrossEntropyLoss()  # 交叉熵
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # SGD优化 lr：学习率

# 训练
for epoch in range(1):  # 所有训练数据轮两轮
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()  # 梯度初始化为0
        # 前向+后向+优化
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d,%5d] loss:%.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

            # 测试
            correct = 0
            total = 0
            with torch.no_grad():
                for data, labels in test_dataloader:
                    data, labels = data.to(DEVICE), labels.to(DEVICE)
                    outputs = net(data)
                    _, predicted = torch.max(outputs.data, 1)  # 下划线表示每行概率值中最大的，1表示输出所在行最大的index
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the test images:%d %%' % (100 * correct / total))

    # # 测试
    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for data, labels in dataloader_mnist:
    #         outputs = net(data)
    #         _, predicted = torch.max(outputs.data, 1)  # 下划线表示每行概率值中最大的，1表示输出所在行最大的index
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #
    # print('Accuracy of the network on the test images:%d %%' % (100 * correct / total))
print('Finished Training')

# 测试
correct = 0
total = 0
with torch.no_grad():
    for data, labels in test_dataloader:
        data, labels = data.to(DEVICE), labels.to(DEVICE)
        outputs = net(data)
        _, predicted = torch.max(outputs.data, 1)  # 下划线表示每行概率值中最大的，1表示输出所在行最大的index
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images:%d %%' % (100 * correct / total))

images_data, labels = next(test_dataiter)
img = make_grid(images_data, 4)  # 4*4拼接图片
save_image(img, 'd.png')  # 将tensor保存为图片
outputs = net(images_data)
_, predicted = torch.max(outputs.data, 1)
index = []
out_pre = []
for label in labels:
    if label == 1:
        index.append('dog')
    else:
        index.append('cat')
for pre in predicted:
    if pre == 1:
        out_pre.append('dog')
    else:
        out_pre.append('cat')
print("labels:", index)
print("predicted:", out_pre)
