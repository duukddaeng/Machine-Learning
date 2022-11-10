#  题1：计算1+2+3+…+100的值
sum = 0
for i in range(1, 101, 1):
    sum += i
print(sum)
# i = 1
# sum = 0
# while i <= 100:
#     sum += i
#     i += 1
# print(sum)

#  题2：求1~100之间能被7整除，但不能同时被5整除的所有整数
b_list = []
for i in range(1, 101, 1):
    if i % 7 == 0 and i % 5 != 0:
        b_list.append(i)
print(b_list)

#  题3：构建序列a_list=['a', 'b', 'mpilgrim', 'z', 'example']，输出所有元素，将元素'z'变为元素'c'。
a_list = ['a', 'b', 'mpilgrim', 'z', 'example']
print(a_list)
i = a_list.index('z')
a_list[i] = 'c'
print(a_list)

#  题4：编写程序生成一个含有20个随机数的列表，要求所有元素不相同，并且每个元素的值介于1到100之间。
import random

c_list = []
i = 0
while i < 20:
    random_num = random.randint(1, 100)
    c_list.append(random_num)
    if c_list.count(random_num) > 1:
        del c_list[i]
    else:
        i += 1
print(c_list)