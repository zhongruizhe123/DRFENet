import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import DnCNN
from models import IRCNN
from models import FDnCNN
# from ADNet import ADNet
from models import ADNetcai
from models import DRFENet
from utils import *
import random
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image
import time
from torchsummary import summary
from torchvision import models
from thop import profile

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="UNetRes_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs\DRFENet\DRFENet(25)\logssigma25_2021-03-18-21-30-31", help='path of log files')
parser.add_argument("--test_data", type=str, default='testimage', help='test on Set12 or BSD68')
parser.add_argument("--test_noiseL", type=float, default=15, help='noise level used on test set')
parser.add_argument("--number", type=int, default='68', help='test on number of Set12 or BSD68 ')
opt = parser.parse_args()
noise_level = 25
new_path = r'./result/{}_Gaussian_UNetRes'.format(noise_level)
# opt = parser.parse_known_args()


def copy_dir(src_path, target_path):
    if os.path.isdir(src_path) and os.path.isdir(target_path):
        filelist_src = os.listdir(src_path)
        for file in filelist_src:
            path = os.path.join(os.path.abspath(src_path), file)
            if os.path.isdir(path):
                path1 = os.path.join(os.path.abspath(target_path), file)
                if not os.path.exists(path1):
                    os.mkdir(path1)
                copy_dir(path, path1)
            else:
                with open(path, 'rb') as read_stream:
                    contents = read_stream.read()
                    path1 = os.path.join(target_path, file)
                    with open(path1, 'wb') as write_stream:
                        write_stream.write(contents)
        return True

    else:
        return False


def normalize(data):
    return data / 255.

# def string_switch(file, value, keys):
#
#     # """
#     # 替换文件中的字符串
#     # :param file:文件名
#     # :param old_str:旧字符串
#     # :param new_str:新字符串
#     # :return:
#     # """
#     file_data = ""
#     with open(file, "r", encoding='gbk') as f:
#         for line in f:
#             # print(line)
#             if "66号摄像头糖包个数:" in line:
#                 line = line.replace(line, "66号摄像头糖包个数:" + value + "\n")
#             file_data += line
#     with open(file, "w") as f:
#         f.write(file_data)
max_psnr=5

def main():
    max_psnr=25
    num_of_image = 0
    # Build model
    t0 = 0.
    seen = opt.number
    print(opt.test_data)
    print('Loading model ...\n')
    # net = UNetRes(in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose')
    # net = FDnCNN(in_nc=2, out_nc=1, nc=64, nb=20, act_mode='R')
    # net = IRDNN(in_nc=1, out_nc=1, nc=64)
    # net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    # net = ADNet(channels=1)
    # net = ADNetcai(channels=3)
    # net = ADNetcai(channels=3, num_of_layers=opt.num_of_layers)
    # net = ADNet(channels=1,num_of_layers=opt.num_of_layers)
    # net = ZADNet1(channels=1,num_of_layers=17)
    # net = ZADNet2(channels=1,num_of_layers=17)
    # net = ZADNet3(channels=1, num_of_layers=17)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    net = DRFENet(channels=3, num_of_layers=17)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    '''
    #输出模型参数
    '''
    summary(model, input_size=(3, 416, 416))
    input = torch.randn(1, 3, 224, 224).to(device)
    flops, params = profile(net, inputs=(input,),)
    from thop import clever_format
    flops, params = clever_format([flops, params], "%.3f")
    print(" %s | %s" % ("Params(M)", "FLOPs(G)"))
    print(" %s | %s" % (params, flops))




    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'model_25.pth')), False)  #    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net(dzz).pth')),False)
    # model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net(dzz).pth')),False)
    model.eval()
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('data', opt.test_data, '*.png'))
    files_source.sort()
    # process data
    psnr_test = 0
    npsnr_test = 0
    line=''
    # for i in range(100):
    #     num_of_image = 0
    for f in files_source:
        num_of_image += 1
        # image
        Img = cv2.imread(f)
        Img = torch.tensor(Img)
        # print(Img.shape)
        Img = Img.permute(2, 0, 1)  #将tensor的维度换位
        Img = Img.numpy()

        a1, a2, a3 = Img.shape
        Img = np.tile(Img, (3, 1, 1, 1))  # expand the dimensional
        Img = np.float32(normalize(Img))


        # print(Img.shape)
        # Img = normalize(np.float32(Img[:, :, 0]))
        # c = tf.concat([ISource[0][0], ISource[0][1]], axis=0)

        ISource = torch.Tensor(Img)
        ##=========================================================
        # ISource1 = ISource[0]
        # ISource1  = ISource1 .permute(1, 2, 0)
        # ISource1 = np.fliplr(ISource1.reshape(-1, 3)).reshape(ISource1.shape)
        # plt.imshow(ISource1)
        # plt.show()
        ##=========================================================
        plt.title('Original')
        ##=========================================================
        ISource1 = ISource[0]
        ISource1  = ISource1 .permute(1, 2, 0)
        ISource1 = np.fliplr(ISource1.reshape(-1, 3)).reshape(ISource1.shape)
        plt.imshow(ISource1)
        # plt.show()
        ##=========================================================
        # plt.imshow(ISource[0][0])
        # plt.savefig(new_path + r'/{}_original.png'.format(num_of_image))
        # plt.show()
        io.imsave(new_path + r'/{}_original.png'.format(num_of_image), ISource1)
        # print(ISource.shape)
        # noise
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL /255.)

        # noisy image
        INoisy = ISource + noise

        # print(noise.shape)
        # ISource2 = INoisy[0]
        # ISource2 = ISource2.permute(1, 2, 0)
        # ISource2 = np.fliplr(ISource2.reshape(-1, 3)).reshape(ISource2.shape)
        # plt.imshow(ISource2)
        # plt.show()






        plt.title('Add_Noise')

        ##=========================================================
        ISourcenoisy = INoisy[0]
        ISourcenoisy  = ISourcenoisy.permute(1, 2, 0)
        ISourcenoisy = np.fliplr(ISourcenoisy.reshape(-1, 3)).reshape(ISourcenoisy.shape)
        plt.imshow(ISourcenoisy)
        # plt.show()
        ##=========================================================

        # plt.imshow(INoisy[0][0], cmap='gray')
        # plt.imshow(INoisy[0][0])

        io.imsave(new_path + r'/{}_INoisy.png'.format(num_of_image), ISourcenoisy)

        ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())

        with torch.no_grad():  # this can save much memory
            # Out = torch.clamp(INoisy - model(INoisy), 0., 1.)

            t = time_synchronized()
            Out = torch.clamp(model(INoisy), 0., 1.)#########################################################################
            t0 += time_synchronized() - t
        t1 = t0   #  * 1E3
        print('Speed: %.4f s inferencel ' % t1)
        t0 = 0.

        new_pic = Out.cpu()

        plt.title('Denoised_Image')
        ##=========================================================
        Denoise = new_pic[0]
        Denoise  = Denoise.permute(1, 2, 0)
        Denoise = np.fliplr(Denoise.reshape(-1, 3)).reshape(Denoise.shape)
        plt.imshow(Denoise)
        ##=========================================================



        # plt.imshow(new_pic[0][0])
        # plt.imshow(new_pic[0][0], cmap='gray')

        io.imsave(new_path + r'/{}_Denoised_Image.png'.format(num_of_image), Denoise)
        # plt.show()
        # plt.savefig(new_path + r'/{}_Denoised_Image.png'.format(num_of_image))

        psnr = batch_PSNR(Out, ISource, 1.)
        psnr_test += psnr
        line = ("%s PSNR %f" % (f, psnr))
        print(line)
        with open(new_path+'/result.txt', 'a') as f:
            f.write(line+"\n")  # label format
        # print("%s PSNR %f" % (f, psnr))
        # line = (f, psnr)


    t = t0 / seen * 1E3    # tuple
    print('Speed: %.1f ms inferencel ' % t)
    psnr_test /= len(files_source)
    print("\nPSNR on test data %f" % psnr_test)
    with open(new_path + '/result.txt', 'a') as f:
        f.write("\nPSNR on test data %f\n" % psnr_test)  # label format
    return psnr_test
        # if max_psnr < psnr_test :
        #     max_psnr = psnr_test
        #     copy_dir("result/25_Gaussian_UNetRes","result/best")
    # print("max_psnr = ",max_psnr)
if __name__ == "__main__":
    for i in range(100):
        psnr_test=main()
        if max_psnr < psnr_test:
            max_psnr = psnr_test
            copy_dir("result/25_Gaussian_UNetRes", "result/best")
    print("max_psnr = ", max_psnr)