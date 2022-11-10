# 使用pytorch，继承nn.Module，编写二维立体卷积层，
# 要求将输入通道、输出通道即滤波器个数、卷积核尺寸、补零个数、滑动步长作为参数。
# 并验证所编写卷积层的效果（输入任意3通道图片，
# 并初始化卷积层参数为[−1,−1,−1；−1,  8, −1；−1,−1,−1]，
# 输出单通道的特征图并以图片形式展示，例如下图）。


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


dataset = DogCat('./dog&cat/test_set/test_set', tuansforms=transform)  # 调用

from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0, drop_last=False)  # 批量处理：每16张图片一批


# 二维立体卷积层
class Net(nn.Module):

    def __init__(self, in_num, out_num, k_size, p_num, stride_num):
        nn.Module.__init__(self)
        # self.filter1 = torch.tensor([[-1, -1, -1],
        #                              [-1, 8, -1],
        #                              [-1, -1, -1]], dtype=torch.float32)
        # self.filter1 = self.filter1.unsqueeze(0)
        self.filter1 = torch.tensor([-1, -1, -1, -1, 8, -1, -1, -1, -1], dtype=torch.float32)  # 卷积核1
        self.filter1 = np.tile(self.filter1, (3, 3))
        self.filter1 = self.filter1.reshape((3, 3, 3, 3))  # 设置成卷积网络需要的格式
        self.filter1 = Variable(torch.from_numpy(self.filter1))
        self.bias1 = torch.ones([3])
        self.conv2d = nn.Conv2d(in_num, out_num, kernel_size=k_size, padding=p_num, stride=stride_num)
        self.conv2d.weight.data = self.filter1.data
        self.conv2d.bias.data = self.bias1.data

    def forward(self, x):
        output = self.conv2d(x)
        # min = int(torch.min(output))
        # output -= min
        # max = math.ceil(torch.max(output))
        # output = output/max*255
        return output


net = Net(3, 1, 3, 1, 1)
for batch_data in dataloader:  # 卷积
    imags_data, labels = batch_data
    img = make_grid(imags_data, 4)
    save_image(img, 'a1.jpg')
    img1 = net(imags_data)  # 作用在图片上
    img1 = make_grid(img1, 4)
    save_image(img1, 'b1.jpg')

