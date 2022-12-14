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
from models import ADNet
from models import ADNetcai
from models import DRFENet
from models import BRDNet
from utils import *
import random
import matplotlib.pyplot as plt
from skimage import io
from thop import profile
import time
from models import BRDNet1
from torchsummary import summary

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="UNetRes_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="G:\gray\DnCNN-PyTorch-master\logs\DRFENet\DRFENet(25)\logssigma25_2021-03-18-21-30-31", help='path of log files')
parser.add_argument("--test_data", type=str, default='Set12', help='test on Set12 or BSD68')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
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


max_psnr=5

def main():
    max_psnr=25
    t0 = 0.
    num_of_image = 0
    # Build model
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
    # net = BRDNet(channels=1)
    net = DRFENet(channels=1)
    # net = BRDNet1(channels=1)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    '''
    #输出模型参数
    
    summary(model, input_size=(1, 416, 416))
    input = torch.randn(1, 1, 224, 224).to(device)
    flops, params = profile(net, inputs=(input,),)
    from thop import clever_format
    flops, params = clever_format([flops, params], "%.3f")
    print(" %s | %s" % ("Params(M)", "FLOPs(G)"))
    print(" %s | %s" % (params , flops))
    '''


    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'model_50.pth')), False)  #    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net(dzz).pth')),False)
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
    fname = r"G:\gray\DnCNN-PyTorch-master\result\train\1.jpg"
    for f in files_source:
        num_of_image += 1
        # image
        # Img = cv2.imread(f)
        # cv2.imwrite(fname , Img, [cv2.IMWRITE_JPEG_QUALITY, 50])
        Img = cv2.imread(fname)
        Img = normalize(np.float32(Img[:, :, 0]))
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        ISource = torch.Tensor(Img)

        plt.title('Original')
        plt.imshow(ISource[0][0], cmap='gray')
        # plt.savefig(new_path + r'/{}_original.png'.format(num_of_image))
        # plt.show()

        # img111 = cv2.imread(ISource[0][0], 1)
        #
        # cv2.imwrite(r"G:\gray\DnCNN-PyTorch-master\result\train\1.jpg", img111, [cv2.IMWRITE_JPEG_QUALITY, 50])

        io.imsave(new_path + r'/{}_original.png'.format(num_of_image), ISource[0][0])

        # noise
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL / 255.)

        # noisy image
        INoisy = ISource + noise

        plt.title('Add_Noise')
        plt.imshow(INoisy[0][0], cmap='gray')
        # plt.savefig(new_path + r'/{}_INoisy.png'.format(num_of_image))
        # plt.show()
        io.imsave(new_path + r'/{}_INoisy.png'.format(num_of_image), INoisy[0][0])

        ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())

        with torch.no_grad():  # this can save much memory
            t = time_synchronized()
            # Out = torch.clamp(INoisy - model(INoisy), 0., 1.)

            Out = torch.clamp(model(INoisy), 0.,1.)  #########################################################################
            t0 += time_synchronized() - t
        t1 = t0  # * 1E3
        print('Speed: %.4f s inferencel ' % t1)
        t0 = 0.




        new_pic = Out.cpu()

        plt.title('Denoised_Image')
        # plt.imshow(new_pic[0][0])
        plt.imshow(new_pic[0][0], cmap='gray')
        io.imsave(new_path + r'/{}_Denoised_Image.png'.format(num_of_image), new_pic[0][0])
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