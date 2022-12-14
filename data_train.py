# -*- coding: utf-8 -*-

# PyTorch 0.4.1, https://pytorch.org/docs/stable/index.html

# =============================================================================
#  @article{zhang2017beyond,
#    title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
#    author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
#    journal={IEEE Transactions on Image Processing},
#    year={2017},
#    volume={26},
#    number={7},
#    pages={3142-3155},
#  }
# by Kai Zhang (08/2018)
# cskaizhang@gmail.com
# https://github.com/cszn
# modified on the code from https://github.com/SaoYan/DnCNN-PyTorch
# =============================================================================

# run this to train the model

# =============================================================================
# For batch normalization layer, momentum should be a value from [0.1, 1] rather than the default 0.1.
# The Gaussian noise output helps to stablize the batch normalization, thus a large momentum (e.g., 0.95) is preferred.
# =============================================================================

import argparse  # python的参数解析argparse模块
import re  # Python正则表达式re模块

# datetime模块提供了简单和复杂的方式用于操纵日期和时间的类。
import os, glob, datetime, time  # 文件名操作模块glob
import numpy as np  # NumPy(Numerical Python) 是 Python 语言的一个扩展程序库，支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。
import torch  # 包 torch 包含了多维张量的数据结构以及基于其上的多种数学操作。
# torch.nn，nn就是neural network的缩写，这是一个专门为深度学习而设计的模块。torch.nn的核心数据结构是Module，这是一个抽象的概念，
# 既可以表示神经网络中的某个层（layer），也可以表示一个包含很多层的神经网络。在实际使用中，最常见的做法是继承nn.Module，撰写自己的网络/层。
import torch.nn as nn

from torch.nn.modules.loss import _Loss  # 没一个损失函数作为一个类，不继承自_Loss类，而_Loss类又继承自Module类
import torch.nn.init as init  # 初始化
from torch.utils.data import DataLoader  # DataLoader类的作用就是实现数据以什么方式输入到什么网络中。
import torch.optim as optim  # torch.optim是一个实现了各种优化算法的库。
from torch.optim.lr_scheduler import \
    MultiStepLR  # torch.optim.lr_scheduler提供了几种方法来根据迭代的数量来调整学习率  #按需调整学习率 MultiStepLR按设定的间隔调整学习率。
import data_generator as dg  # 导入处理数据文件，命名dg,数据生成器
from data_generator import DenoisingDataset  # 从data_generator文件导入 DenoisingDataset类
from models import DnCNN
from models import IRCNN
from models import FDnCNN
from models import ZRZDnCNN

# 创建一个解析器
# 使用 argparse 的第一步是创建一个 ArgumentParser 对象
parser = argparse.ArgumentParser(description='PyTorch DnCNN')
# 参加参数
# 给一个 ArgumentParser 添加程序参数信息是通过调用 add_argument() 方法完成的。通常，这些调用指定 ArgumentParser 如何获取命令行字符串并将其转换为对象。
# 模型  字符串   默认为DnCNN
parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
# 批量大小  整型   默认大小128
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
# 训练数据   字符串  默认 data/Train400  路径
parser.add_argument('--train_data', default='data/train400', type=str, help='path of train data')
# 噪声水平  整型  默认25
parser.add_argument('--sigma', default=25, type=int, help='noise level')
# epoch 整型  默认180
parser.add_argument('--epoch', default=50, type=int, help='number of train epoches')
# 学习率  float 0.001  adam优化算法
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers") #17
# 解析参数
# ArgumentParser 通过 parse_args() 方法解析参数。它将检查命令行，把每个参数转换为适当的类型然后调用相应的操作。
args = parser.parse_args()

# rgparse解析命令行参数来传递参数
batch_size = args.batch_size  # batch_size=128
cuda = torch.cuda.is_available()  # torch.cuda.is_available() cuda是否可用；
n_epoch = args.epoch  # n_epoch= 180
sigma = args.sigma  # sigma  = 25
# os.path.join()：  将多个路径组合后返回args.model = DNCNN  str(sigma)=25
# 组合之后的路径为models/DNCNN_sigma25
save_dir = os.path.join('logs', args.model + '_' + 'sigma' + str(sigma))

# 判断路径是否存在 如果不存在 则新建路径
# os.mkdir()创建路径中的最后一级目录，即：只创建DNCNN_sigma25目录，而如果之前的目录不存在并且也需要创建的话，就会报错。
# os.makedirs()创建多层目录，即：models,DNCNN_sigma25如果都不存在的话，会自动创建，
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


# 声明一个类，并继承自nn.Module
# class DnCNN(nn.Module):
#     # 定义构造函数
#     # 构建网络最开始写一个class，然后def _init_（输入的量），然后super（DnCNN，self）.__init__()这三句是程式化的存在，
#     # 初始化
#     def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
#         ##初始化方法使用父类的方法即可，super这里指的就是nn.Module这个基类，第一个参数是自己创建的类名
#         super(DnCNN, self).__init__()
#         # 定义自己的网络
#
#         kernel_size = 3  # 卷积核的大小  3*3
#         padding = 1  ##padding表示的是在图片周围填充0的多少，padding=0表示不填充，padding=1四周都填充1维
#         layers = []
#         # 四个参数 输入的通道  输出的通道  卷积核大小  padding
#         # 构建一个输入通道为channels，输出通道为64，卷积核大小为3*3,四周进行1个像素点的零填充的conv1层  #bias如果bias=True，添加偏置
#         layers.append(
#             nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
#                       bias=True))
#         ##增加网络的非线性——激活函数nn.ReLU(True)  在卷积层（或BN层）之后，池化层之前，添加激活函数
#         layers.append(nn.ReLU(inplace=True))
#         for _ in range(depth - 2):
#             ##构建卷积层
#             layers.append(
#                 nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
#                           bias=False))
#             # 加快收敛速度一一批标准化层nn.BatchNorm2d()  输入通道为64的BN层 与卷积层输出通道数64对应
#             # eps为保证数值稳定性（分母不能趋近或取0）,给分母加上的值。默认为1e-4
#             # momentum： 动态均值和动态方差所使用的动量。默认为0.1
#             layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
#             # 增加网络的非线性——激活函数nn.ReLU(True)  在卷积层（或BN层）之后，池化层之前，添加激活函数
#             layers.append(nn.ReLU(inplace=True))
#         # 构建卷积层
#         layers.append(
#             nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding,
#                       bias=False))
#         # 利用nn.Sequential()按顺序构建网络
#         self.dncnn = nn.Sequential(*layers)
#         self._initialize_weights()  # 调用初始化权重函数
#
#     # 定义自己的前向传播函数
#     def forward(self, x):
#         y = x
#         out = self.dncnn(x)
#         return y - out
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             ## 使用isinstance来判断m属于什么类型【卷积操作】
#             if isinstance(m, nn.Conv2d):
#                 # 正交初始化（Orthogonal Initialization）主要用以解决深度网络下的梯度消失、梯度爆炸问题
#                 init.orthogonal_(m.weight)
#                 print('init weight')
#                 if m.bias is not None:
#                     # init.constant_常数初始化
#                     init.constant_(m.bias, 0)
#             ## 使用isinstance来判断m属于什么类型【批量归一化操作】
#             elif isinstance(m, nn.BatchNorm2d):
#                 # init.constant_常数初始化
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)


# 定义损失函数类
class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """

    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    # MSELoss  计算input和target之差的平方
    # reduce(bool)- 返回值是否为标量，默认为True size_average(bool)- 当reduce=True时有效。为True时，返回的loss为平均值；为False时，返回的各样本的loss之和。
    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)


# 看的不是特别懂，返回值要么最大的，要么0
def findLastCheckpoint(save_dir):
    # 返回所有匹配的文件路径列表。它只有一个参数pathname，定义了文件路径匹配规则，这里可以是绝对路径，也可以是相对路径。
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            # re.findall  的简单用法（返回string中所有与pattern相匹配的全部字串，返回形式为数组）
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))  # append() 方法用于在列表末尾添加新的对象。
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


# strftime()方法使用日期，时间或日期时间对象返回表示日期和时间的字符串
def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


if __name__ == '__main__':
    # model selection
    print('===> Building model')
    # model = DnCNN(channels=1)
    model = ZRZDnCNN(channels=1, num_of_layers=args.num_of_layers)
    initial_epoch = findLastCheckpoint(save_dir=save_dir)  # load the last model matconvnet
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        # model.load_state_dict(torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch)))
        # 加载模型
        model = torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch))
    # 训练模型时会在前面加上
    model.train()
    # criterion = nn.MSELoss(reduction = 'sum')  # PyTorch 0.4.1  #返回的各样本的loss之和
    criterion = sum_squared_error()
    if cuda:
        model = model.cuda()
        # device_ids = [0]
        # model = nn.DataParallel(model, device_ids=device_ids).cuda()
        # criterion = criterion.cuda()
    ## Optimizer  采用Adam算法优化，模型参数  学习率
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # milestones为一个数组，如 [50,70]. gamma为0.1 倍数。如果learning rate开始为0.01 ，则当epoch为50时变为0.001，epoch 为70 时变为0.0001。当last_epoch=-1,设定为初始lr
    scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.1)  # learning rates
    for epoch in range(initial_epoch, n_epoch):  # n——epoch = 180

        scheduler.step(epoch)  # step to the learning rate in this epcoh
        xs = dg.datagenerator(data_dir=args.train_data)  # 调用数据生成器函数
        xs = xs.astype('float32') / 255.0  # 对数据进行处理，位于【0 1】
        # torch.from_numpy将numpy.ndarray 转换为pytorch的 Tensor。  transpose多维数组转置
        xs = torch.from_numpy(xs.transpose((0, 3, 1, 2)))  # tensor of the clean patches, N X C X H X W
        # 加噪声函数
        DDataset = DenoisingDataset(xs)
        # dataset：（数据类型 dataset）
        # num_workers：工作者数量，默认是0。使用多少个子进程来导入数据
        # drop_last：丢弃最后数据，默认为False。设置了 batch_size 的数目后，最后一批数据未必是设置的数目，有可能会小些。这时你是否需要丢弃这批数据。
        # shuffle洗牌。默认设置为False。在每次迭代训练时是否将数据洗牌，默认设置是False。将输入数据的顺序打乱，是为了使数据更有独立性，但如果数据是有序列特征的，就不要设置成True了。
        DLoader = DataLoader(dataset=DDataset, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True)
        epoch_loss = 0  # 初始化
        start_time = time.time()  # time.time() 返回当前时间的时间戳
        print("epoch:", epoch, "\n")
        for n_count, batch_yx in enumerate(DLoader):  # enumerate() 函数用于将一个可遍历的数据对象
            optimizer.zero_grad()  # optimizer.zero_grad()意思是把梯度置零，也就是把loss关于weight的导数变成0.
            if cuda:
                batch_x, batch_y = batch_yx[1].cuda(), batch_yx[0].cuda()
            loss = criterion(model(batch_y), batch_x)  # 计算损失值
            epoch_loss += loss.item()  # 对损失值求和
            loss.backward()  # 反向传播
            optimizer.step()  # adam优化
            # 每十张输出epoch  n_count  xs.size(0)//batch_size  loss.item()/batch_size)
            # 不清楚xs.size(0)//batch_size是什么意思。1862定值
            # print("epoch:",epoch,"\n")
            if n_count % 10 == 0:
                print('%4d %4d / %4d loss = %2.4f' % (
                epoch + 1, n_count, xs.size(0) // batch_size, loss.item() / batch_size))
        elapsed_time = time.time() - start_time  # 当前时间-开始时间

        log('epcoh = %4d , loss = %4.4f , time = %4.2f s' % (epoch + 1, epoch_loss / n_count, elapsed_time))
        # numpy.savetxt(fname,X):第一个参数为文件名，第二个参数为需要存的数组（一维或者二维）第三个参数是保存的数据格式
        # hstack 和 vstack这两个函数分别用于在水平方向和竖直方向增加数据
        np.savetxt('train_result.txt', np.hstack((epoch + 1, epoch_loss / n_count, elapsed_time)), fmt='%2.4f')
        # torch.save(model.state_dict(), os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))
        # 保存模型
        torch.save(model.state_dict(), os.path.join(save_dir, 'model_%03d.pth' % (epoch + 1)))
    torch.save(model.state_dict(), os.path.join(save_dir, 'net.pth' ))