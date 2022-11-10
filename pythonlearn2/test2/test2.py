# 第一题：
# 1）读取lena图片，并显示；
# 2）取图像的第0通道，并显示；
# 3）通过公式计算灰度图，Gray=0.299*R+0.587*G+0.144*B;并显示；
# 4）将图片中的人脸位置，行列100至200遮挡，并显示；
# 5）将图片中的人脸位置，行列100至200用猫脸遮挡，并显示；
# 6)  将两幅图横着以及竖着拼接，并显示。

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

lena = mpimg.imread('lena.jpg')  # 读取lena图片
imgplot = plt.imshow(lena)  # 显示
plt.show()

lena0 = lena[:, :, 0]  # 取图像的第0通道
imgplot = plt.imshow(lena0, cmap="Reds")  # 显示
plt.show()

lena_gray = 0.299 * lena[:, :, 0] + 0.587 * lena[:, :, 1] + 0.144 * lena[:, :, 2]  # 计算灰度图
imgplot = plt.imshow(lena_gray, cmap="gray")  # 显示
plt.show()

# lena_black = np.zeros([100, 100, 3])
lena_black = lena
lena_black[100:200, 100:200, :] = 0  # 遮挡行列100-200
imgplot = plt.imshow(lena_black)
plt.imsave("lena_black.jpg", lena_black)
lena_black = mpimg.imread('lena_black.jpg')
plt.show()

cat = mpimg.imread('cat.jpg')  # 读取cat图片
cat = cat[::12, ::19, :]  # 缩小图片行列
lena_cat = lena
lena_cat[100:200, 100:200, :] = cat[:100, :100, :]  # 用猫脸进行替换
imgplot = plt.imshow(lena_cat)
plt.imsave("lena_cat.jpg", lena_cat)
lena_cat = mpimg.imread('lena_cat.jpg')
plt.show()

lena_x = np.hstack((lena_black, lena_cat))  # 横向拼接
imgplot = plt.imshow(lena_x)
plt.show()

lena_y = np.vstack((lena_black, lena_cat))  # 纵向拼接
imgplot = plt.imshow(lena_y)
plt.show()