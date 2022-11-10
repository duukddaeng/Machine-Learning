# 使用pytorch，继承nn.Module，编写二维立体卷积层，
# 要求将输入通道、输出通道即滤波器个数、卷积核尺寸、补零个数、滑动步长作为参数。
# 并验证所编写卷积层的效果（输入任意3通道图片，
# 并初始化卷积层参数为[−1,−1,−1；−1,  8, −1；−1,−1,−1]，
# 输出单通道的特征图并以图片形式展示，例如下图）。


import numpy as np
import torch
from PIL import Image
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.autograd import Variable
# 载入图片
from torchvision.utils import save_image

image = Image.open('lena.jpg')
# 图像转换为Tensor
x = torch.from_numpy(np.array(image, dtype=np.float32))
x = x.unsqueeze(0)
y = x.transpose(2, 3)
y = y.transpose(1, 2)


# 二维立体卷积层
class Net(nn.Module):

    def __init__(self, in_num, out_num, k_size, p_num, stride_num):
        nn.Module.__init__(self)
        # self.filter1 = torch.tensor([[-1, -1, -1],
        #                              [-1, 8, -1],
        #                              [-1, -1, -1]], dtype=torch.float32)
        # self.filter1 = self.filter1.unsqueeze(0)
        self.filter1 = torch.tensor([-1, -1, -1, -1, 8, -1, -1, -1, -1], dtype=torch.float32)  # 卷积核1
        self.filter1 = np.tile(self.filter1, (1, 3))
        self.filter1 = self.filter1.reshape((3, 3, 3))  # 设置成卷积网络需要的格式
        self.filter1 = Variable(torch.from_numpy(self.filter1))
        self.filter1 = self.filter1.unsqueeze(0)
        self.conv2d = nn.Conv2d(in_num, out_num, kernel_size=k_size, padding=p_num, stride=stride_num)
        self.conv2d.weight.data = self.filter1.data

    def forward(self, x):
        output = self.conv2d(x)
        # min = int(torch.min(output))
        # output -= min
        # max = math.ceil(torch.max(output))
        # output = output/max*255
        return output


net = Net(3, 1, 3, 1, 1)
out = net(y)
save_image(out, 'a.png')
