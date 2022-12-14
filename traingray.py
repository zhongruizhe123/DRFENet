import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import  time 
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import ADNet
from models import BTBUNetwithoutDEB
from models import DnCNN
from models import ECNDNet
from models import BTBUNetwithoutRDB
from models import IRCNN
from models import BTBUNet
from models import FFDNet
from models import BRDNet
from models import BRDNet1
from models import BRDNet2
from models import BTBUNetwithoutconcat
from torch.nn.modules.loss import _Loss
from dataset import prepare_data, Dataset
from utils import *
import glob
import re  # Python正则表达式re模块
import matplotlib.pyplot as plt
from skimage import io
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=65, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
'''
parser.add_argument("--clip",type=float,default=0.005,help='Clipping Gradients. Default=0.4') #tcw201809131446tcw
parser.add_argument("--momentum",default=0.9,type='float',help = 'Momentum, Default:0.9') #tcw201809131447tcw
parser.add_argument("--weight-decay","-wd",default=1e-3,type=float,help='Weight decay, Default:1e-4') #tcw20180913347tcw
'''
opt = parser.parse_args()
class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)

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



def main():
    # Load dataset
    t1 = time.perf_counter()
    save_dir = opt.outf + 'sigma' + str(opt.noiseL) + '_' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    # net = ADNet(channels=3, num_of_layers=opt.num_of_layers)
    # net = BTBUNet(channels=1,num_of_layers=opt.num_of_layers)
    # net = DnCNN(channels=1)
    # net = ECNDNet(channels=1)
    # net = BTBUNet(channels=3, num_of_layers=17)
    # net = FFDNet(is_gray = "gray")
    # net = IRCNN()
    # net = BTBUNetwithoutDEB(channels=1)
    # net = BTBUNetwithoutRDB(channels=1)
    # net = BRDNet(channels=1)
    # net = VDN(1, slope=0.2, wf=64, dep_U=4)
    # net = BRDNet1(channels=1)
    net = BRDNet2(channels=1)
    # net = RIDNET()
    # print("Hello World")
    # net = UNetD(1)
    # print("Hello World")
    net.apply(weights_init_kaiming)
    # model = net
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    save_dir1 = "logssigma25_2021-04-03-01-29-20/model_4.pth"   #设立断点保存文件夹
    # save_dir2 = "logssigma25_2021-06-02-02-52-01"
    save_dir2 = '0'
#=================================================================
    initial_epoch = findLastCheckpoint(save_dir=save_dir2)  # load the last model matconvnet
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        model.load_state_dict(torch.load(os.path.join(save_dir2, 'model_%d.pth' % initial_epoch)))
        # 加载模型
        # model = torch.load(os.path.join(save_dir2, 'model_%d.pth' % initial_epoch))



#=================================================================


#=================================================================

    model = model.train()
    criterion = nn.MSELoss(size_average=False)
    # criterion = nn.MSELoss(size_average=False)
    # Move to GPU
    # cuda = torch.cuda.is_available()  # torch.cuda.is_available() cuda是否可用；
    # if cuda:
    #     model = net.cuda()
    # device_ids = [0]
    # model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda()
    # Optimizer
    filename1 = save_dir + 'result.txt'
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    noiseL_B=[0,55] # ingnored when opt.mode=='S'
    psnr_list = []
    num_of_image = 0
    for epoch in range(initial_epoch,opt.epochs):
        if epoch <= opt.milestone:
            current_lr = opt.lr
        if epoch > 30 and  epoch <=40:
            current_lr  =  opt.lr/10.
        if epoch > 40  and epoch <=50:
            current_lr = opt.lr/100.
        if epoch >50   and epoch <=58:
            current_lr = opt.lr/1000.
        if epoch > 58 and epoch <= 65:
            current_lr = opt.lr / 10000.
 
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        for i, data in enumerate(loader_train, 0):
            # training step
            model.train()
            img_train = data
            print(img_train.size())
            if opt.mode == 'S':
                 noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL/255.)
            if opt.mode == 'B':
                noise = torch.zeros(img_train.size())
                stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
                for n in range(noise.size()[0]):
                    sizeN = noise[0,:,:,:].size()
                    noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/255.) 
            imgn_train = img_train + noise      #加上噪声
            num_of_image += 1
            new_path = r'result\train'
            io.imsave(new_path + r'/{}_INoisy.png'.format(num_of_image), imgn_train[0][0])
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda()) 
            noise = Variable(noise.cuda())  
            out_train = model(imgn_train)
            loss =  criterion(out_train, img_train) / (imgn_train.size()[0]*2)
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
            model.eval()
            out_train = torch.clamp(model(imgn_train), 0., 1.) 
            psnr_train = batch_PSNR(out_train, img_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                (epoch+1, i+1, len(loader_train), loss.item(), psnr_train))
        model.eval() 
        # validate
        psnr_val = 0
        for k in range(len(dataset_val)):
            img_val = torch.unsqueeze(dataset_val[k], 0)
            torch.manual_seed(0) #set the seed 
            noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL/255.)
            imgn_val = img_val + noise
            img_val, imgn_val = Variable(img_val.cuda()), Variable(imgn_val.cuda(),requires_grad=False)
            out_val = torch.clamp(model(imgn_val), 0., 1.)
            psnr_val += batch_PSNR(out_val, img_val, 1.)
        psnr_val /= len(dataset_val)
        psnr_val1 = str(psnr_val) 
        psnr_list.append(psnr_val1)
        f = open(filename1, 'a')
        # for line in psnr_val1:
        f.write("epoch:%d   "%(epoch+1)+psnr_val1 + '\n')
        f.close()
        print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))
        model_name = 'model'+ '_' + str(epoch+1) + '.pth' 
        torch.save(model.state_dict(), os.path.join(save_dir, model_name)) 
    filename = save_dir + 'psnr.txt' 
    f = open(filename,'w') 
    for line in psnr_list:  
        f.write(line+'\n') 
    f.close()
    t2 = time.perf_counter()
    t = t2-t1
    print(t)

if __name__ == "__main__":
    if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path='data', patch_size=50, stride=40, aug_times=1) 
        if opt.mode == 'B':
            prepare_data(data_path='data', patch_size=50, stride=10, aug_times=2)
    main()
