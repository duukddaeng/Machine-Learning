# 第二题：
# 利用pytorch数据处理工具包中的dataset、dataloader等工具批量的读取和处理dog&cat数据集。
# 对读取的数据按照batch_size=16进行批量的卷积（利用for循环处理dataloader的每一个batch数据），
# 尝试不同卷积核（自选），并显示效果。随机的抽取一个batch的图片展示，将16个图片以及处理效果分别利用make_grid展示出来。
from torch.autograd import Variable
from torchvision import transforms as T

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
import torch.nn.functional as F

filter1 = torch.tensor([1, 2, 1, 0, 0, 0, -1, -2, -1], dtype=torch.float32)  # 卷积核1
filter1 = np.tile(filter1, (3, 3))
filter1 = filter1.reshape((3, 3, 3, 3))  # 设置成卷积网络需要的格式
weight1 = Variable(torch.from_numpy(filter1))  # 给卷积核赋值

filter2 = torch.tensor([-1, 0, 1, -2, 0, 2, -1, 0, 2], dtype=torch.float32)  # 卷积核2
filter2 = np.tile(filter2, (3, 3))
filter2 = filter2.reshape((3, 3, 3, 3))  # 设置成卷积网络需要的格式
weight2 = Variable(torch.from_numpy(filter2))  # 给卷积核赋值

from PIL import Image
from torchvision.utils import make_grid, save_image

for batch_data in dataloader:  # 卷积
    imags_data, labels = batch_data
    img = make_grid(imags_data, 4)
    save_image(img, 'a.jpg')
    img1 = F.conv2d(imags_data, weight1)  # 作用在图片上
    img1 = make_grid(img1, 4)
    save_image(img1, 'b.jpg')
    img2 = F.conv2d(imags_data, weight2)  # 作用在图片上
    img2 = make_grid(img2, 4)
<<<<<<< HEAD
    save_image(img2, 'c.jpg')
=======
    save_image(img2, 'c.jpg')
>>>>>>> 421255f3b4bfdb15774b2ceb8bb9915a2785340e
