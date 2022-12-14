import torch
import torch.nn as nn
import basicblock as B
import numpy as np
import math

class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        # layers.append(B.ResBlock(features, features, kernel_size=kernel_size, stride=1, padding=padding, bias=False, mode='CRC', negative_slope=0.2))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        # layers.append(B.ResBlock(features, features, kernel_size=kernel_size, stride=1, padding=padding, bias=False, mode='CRC', negative_slope=0.2))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))             #原来out_channels为channels



        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        y = x
        out = self.dncnn(x)
        out2=y-out
        return out2


class ZRZDnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(ZRZDnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        # layers.append(DnCNN(channels=1))
        layers.append(B.ZRZResBlock2())
        layers.append(B.ZRZResBlock2())
        layers.append(nn.Conv2d(in_channels=64, out_channels=channels, kernel_size=kernel_size, padding=padding,bias=True))
        self.ZRZdncnn = nn.Sequential(*layers)
    def forward(self, x):
        out = self.ZRZdncnn(x)
        return out + x
# --------------------------------------------
# IRCNN denoiser
# --------------------------------------------
class IRCNN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64):
        """
        # ------------------------------------
        denoiser of IRCNN
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super(IRCNN, self).__init__()
        L =[]
        L.append(nn.Conv2d(in_channels=in_nc, out_channels=nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=2, dilation=2, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=3, dilation=3, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=4, dilation=4, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=3, dilation=3, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=2, dilation=2, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=out_nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))
        self.model = B.sequential(*L)

    def forward(self, x):
        n = self.model(x)
        return x-n


# --------------------------------------------
# FDnCNN
# --------------------------------------------
# Compared with DnCNN, FDnCNN has three modifications:
# 1) add noise level map as input
# 2) remove residual learning and BN
# 3) train with L1 loss
# may need more training time, but will not reduce the final PSNR too much.
# --------------------------------------------
class FDnCNN(nn.Module):
    def __init__(self, in_nc=2, out_nc=1, nc=64, nb=20, act_mode='R'):
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        """
        super(FDnCNN, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True

        m_head = B.conv(in_nc, nc, mode='C'+act_mode[-1], bias=bias)
        m_body = [B.conv(nc, nc, mode='C'+act_mode, bias=bias) for _ in range(nb-2)]
        m_tail = B.conv(nc, out_nc, mode='C', bias=bias)

        self.model = B.sequential(m_head, *m_body, m_tail)

    def forward(self, x):
        x = self.model(x)
        return x

#----------------------------------------------------------------------------------分割adnet-------------------------------------------------------------------------------



class Conv_BN_Relu_first(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding,groups,bias):
        super(Conv_BN_Relu_first,self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        groups =1
        self.conv = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(features)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))

class Conv_BN_Relu_other(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding,groups,bias):
        super(Conv_BN_Relu_other,self).__init__()
        kernel_size = 3
        padding = 1
        features = out_channels
        groups =1
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=kernel_size, padding=padding,groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(features)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))


class Conv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding,groups,bais):
        super(Conv,self).__init__()
        kernel_size = 3
        padding = 1
        features = 1
        groups =1
        self.conv = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,groups=groups, bias=False)
    def forward(self,x):
        return self.conv(x)

class Self_Attn(nn.Module):
    def __init__(self,in_dim):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim,out_channels=in_dim//8,kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim,out_channels=in_dim//8,kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim,out_channels=in_dim,kernel_size=1)
        self.gamma=nn.Parameter(torch.zeros(1))
        self.softmax=nn.Softmax(dim=-1)
    def forward(self,x):
        m_batchsize, C, width,height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1)
        proj_key = self.key_conv(x).view(m_batchsize,-1,width*height)
        print (proj_query.size())
        print (proj_key.size())
        print ('5')
        energy = torch.bmm(proj_query,proj_key)
        print ('6')
        #print energy.size()
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height)
        print ('1')
        out = torch.bmm(proj_value,attention.permute(0,2,1))
        print ('2')
        out = out.view(m_batchsize,C,width,height)
        out = self.gamma*out + x
        return out, attention

class ADNet(nn.Module):
    def __init__(self, channels, num_of_layers=15):
        super(ADNet, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        groups =1
        layers = []
        kernel_size1 = 1
        '''
        #self.gamma = nn.Parameter(torch.zeros(1))
        '''
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.first_res = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,groups=groups ,bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
        self.conv_res = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,groups=groups ,bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,groups=groups ,bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,groups=groups ,bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,groups=groups ,bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,groups=groups ,bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )


        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels=channels,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias=False,dilation=2),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_3 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_4 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_5 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias=False,dilation=2),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_6 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_7 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_8 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_9 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias=False,dilation=2),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_10 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_11 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_12 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias=False,dilation=2),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_13 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_14 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_15 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_16 = nn.Conv2d(in_channels=features,out_channels=1,kernel_size=kernel_size,padding=1,groups=groups,bias=False)
        self.conv3 = nn.Conv2d(in_channels=2,out_channels=1,kernel_size=1,stride=1,padding=0,groups=1,bias=True)
        self.ReLU = nn.ReLU(inplace=True)
        self.Tanh= nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
                clip_b = 0.025
                w = m.weight.data.shape[0]
                for j in range(w):
                    if m.weight.data[j] >= 0 and m.weight.data[j] < clip_b:
                        m.weight.data[j] = clip_b
                    elif m.weight.data[j] > -clip_b and m.weight.data[j] < 0:
                        m.weight.data[j] = -clip_b
                m.running_var.fill_(0.01)
    # def _make_layers(self, block,features, kernel_size, num_of_layers, padding=1, groups=1, bias=False):
	#     layers = []
    #     for _ in range(num_of_layers):
    #         layers.append(block(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, groups=groups, bias=bias))
    #     return nn.Sequential(*layers)
    def forward(self, x):
        input = x
        x1 = self.conv1_1(x)
        x1 = self.conv1_2(x1)
        x1 = self.conv1_3(x1)
        x1 = self.conv1_4(x1)
        x1 = self.conv1_5(x1)
        x1 = self.conv1_6(x1)
        x1 = self.conv1_7(x1)
        x1t = self.conv1_8(x1)
        x1 = self.conv1_9(x1t)
        x1 = self.conv1_10(x1)
        x1 = self.conv1_11(x1)
        x1 = self.conv1_12(x1)
        x1 = self.conv1_13(x1)
        x1 = self.conv1_14(x1)
        x1 = self.conv1_15(x1)
        x1 = self.conv1_16(x1)
        out = torch.cat([x,x1],1)
        out= self.Tanh(out)
        out = self.conv3(out)
        out = out*x1
        out2 = x - out
        return out2


class ADNetcai(nn.Module):
    def __init__(self, channels, num_of_layers=15):
        super(ADNetcai, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        groups = 1
        layers = []
        kernel_size1 = 1
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_4 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_5 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_6 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_7 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_8 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_9 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_10 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_11 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_12 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_13 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_14 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_15 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_16 = nn.Conv2d(in_channels=features, out_channels=3, kernel_size=kernel_size, padding=1,
                                  groups=groups, bias=False)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.ReLU = nn.ReLU(inplace=True)
        self.Tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
                clip_b = 0.025
                w = m.weight.data.shape[0]
                for j in range(w):
                    if m.weight.data[j] >= 0 and m.weight.data[j] < clip_b:
                        m.weight.data[j] = clip_b
                    elif m.weight.data[j] > -clip_b and m.weight.data[j] < 0:
                        m.weight.data[j] = -clip_b
                m.running_var.fill_(0.01)
        # def _make_layers(self, block,features, kernel_size, num_of_layers, padding=1, groups=1, bias=False):
        # layers = []
        #     for _ in range(num_of_layers):
        #         layers.append(block(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, groups=groups, bias=bias))
        #     return nn.Sequential(*layers)

    def forward(self, x):
        input = x
        x1 = self.conv1_1(x)
        x1 = self.conv1_2(x1)
        x1 = self.conv1_3(x1)
        x1 = self.conv1_4(x1)
        x1 = self.conv1_5(x1)
        x1 = self.conv1_6(x1)
        x1 = self.conv1_7(x1)
        x1t = self.conv1_8(x1)
        x1 = self.conv1_9(x1t)
        x1 = self.conv1_10(x1)
        x1 = self.conv1_11(x1)
        x1 = self.conv1_12(x1)
        x1 = self.conv1_13(x1)
        x1 = self.conv1_14(x1)
        x1 = self.conv1_15(x1)
        x1 = self.conv1_16(x1)
        out = torch.cat([x, x1], 1)
        out = self.Tanh(out)
        out = self.conv3(out)
        out = out * x1
        out2 = x - out
        return out2
'''
对ADNet修改第二
把之前换成DNCNN
'''






class DRFENet(nn.Module):
    def __init__(self, channels, num_of_layers=15):
        super(DRFENet, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        groups = 1
        layers = []
        kernel_size1 = 1
        '''
        #self.gamma = nn.Parameter(torch.zeros(1))
        '''
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.first_res = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
        self.second_res = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
        self.third_res = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
        self.fourth_res = nn.Sequential(
            nn.Conv2d(in_channels=features * 2, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
        self.fifth_res = nn.Sequential(
            nn.Conv2d(in_channels=features * 2, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_4 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_5 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_6 = nn.Sequential(
            nn.Conv2d(in_channels=features*2, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_7 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_8 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_9 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_10 = nn.Sequential(
            nn.Conv2d(in_channels=features*2, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_11 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_12 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_13 = nn.Sequential(
            nn.Conv2d(in_channels=features*2, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_14 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_15 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_simple = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1,
                                      groups=groups, bias=False)
        self.conv1_16 = nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=1,
                                  groups=groups, bias=False)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=features*2, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv3 = nn.Conv2d(in_channels=channels*2, out_channels=channels, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.ReLU = nn.ReLU(inplace=True)
        self.Tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
                clip_b = 0.025
                w = m.weight.data.shape[0]
                for j in range(w):
                    if m.weight.data[j] >= 0 and m.weight.data[j] < clip_b:
                        m.weight.data[j] = clip_b
                    elif m.weight.data[j] > -clip_b and m.weight.data[j] < 0:
                        m.weight.data[j] = -clip_b
                m.running_var.fill_(0.01)
        # def _make_layers(self, block,features, kernel_size, num_of_layers, padding=1, groups=1, bias=False):
        #     layers = []
        #     for _ in range(num_of_layers):
        #         layers.append(block(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, groups=groups, bias=bias))
        #     return nn.Sequential(*layers)

    def forward(self, x):
        input = x
        x11 = self.first_res(input)  # 第一层
        x22 = self.second_res(x11)  # 第二层
        # x0 = torch.cat([input,x0],1)    #与原图链接
        x33 = self.third_res(x22)  # 第三层
        x333 = x33
        x33 = torch.cat([x11, x33], 1)
        x44 = self.fourth_res(x33)
        x44 = torch.cat([x22, x44], 1)
        x55 = self.fifth_res(x44)
        x55 = torch.cat([x333, x55], 1)
        x55 = self.conv4(x55)
        x1 = self.conv1_1(x55)
        x1 = self.conv1_2(x1)   #空洞卷积
        x1 = self.conv1_3(x1)
        x1 = self.conv1_4(x1)
        # x66=x1
        # x1 = torch.cat([x1,x55], 1)
        x1 = self.conv1_5(x1)   #空洞卷积
        x66=x1
        x1 = torch.cat([x1,x55], 1)
        x1 = self.conv1_6(x1)
        x1 = self.conv1_7(x1)
        x1t = self.conv1_8(x1)
        # x66=x1
        # x1 = torch.cat([x1,x55], 1)
        x1 = self.conv1_9(x1t)  #空洞卷积
        x77=x1
        x1 = torch.cat([x1,x66], 1)
        x1 = self.conv1_10(x1)
        x1 = self.conv1_11(x1)
        x1 = self.conv1_12(x1)  #空洞卷积
        # x88=x1
        x1 = torch.cat([x1,x77], 1)
        x1 = self.conv1_13(x1)
        x1 = self.conv1_14(x1)
        x1 = self.conv1_15(x1)
        x1 = self.conv1_16(x1)
        out = torch.cat([x, x1], 1)
        out = self.Tanh(out)
        out = self.conv3(out)
        out = out * x1
        out2 = x - out
        return out2




class  ECNDNet(nn.Module):
    def __init__(self, channels, num_of_layers=15):
        super(ECNDNet, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        groups =1
        layers = []
        kernel_size1 = 1
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels=channels,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias=False,dilation=2),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_3 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_4 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_5 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias=False,dilation=2),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_6 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_7 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_8 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_9 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias=False,dilation=2),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_10 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_11 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_12 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias=False,dilation=2),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_13 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_14 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_15 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_16 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=1,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.ReLU(inplace=True))
        self.conv3 = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=kernel_size,stride=1,padding=1,groups=1,bias=True)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
        #         clip_b = 0.025
        #         w = m.weight.data.shape[0]
        #         for j in range(w):
        #             if m.weight.data[j] >= 0 and m.weight.data[j] < clip_b:
        #                 m.weight.data[j] = clip_b
        #             elif m.weight.data[j] > -clip_b and m.weight.data[j] < 0:
        #                 m.weight.data[j] = -clip_b
        #         m.running_var.fill_(0.01)

    def forward(self, x):
        input = x
        x1 = self.conv1_1(x)
        x1 = self.conv1_2(x1)
        x1 = self.conv1_3(x1)
        x1 = self.conv1_4(x1)
        x1 = self.conv1_5(x1)
        x1 = self.conv1_6(x1)
        x1 = self.conv1_7(x1)
        x1t = self.conv1_8(x1)
        x1 = self.conv1_9(x1t)
        x1 = self.conv1_10(x1)
        x1 = self.conv1_11(x1)
        x1 = self.conv1_12(x1)
        x1 = self.conv1_13(x1)
        x1 = self.conv1_14(x1)
        x1 = self.conv1_15(x1)
        x1 = self.conv1_16(x1)
        out = self.conv3(x1)
        out1 = x - out
        return out1


class FFDNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=15, act_mode='R'):
        """
        # ------------------------------------
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        # ------------------------------------
        """
        super(FFDNet, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True
        sf = 2

        self.m_down = B.PixelUnShuffle(upscale_factor=sf)

        m_head = B.conv(in_nc * sf * sf + 1, nc, mode='C' + act_mode[-1], bias=bias)
        m_body = [B.conv(nc, nc, mode='C' + act_mode, bias=bias) for _ in range(nb - 2)]
        m_tail = B.conv(nc, out_nc * sf * sf, mode='C', bias=bias)

        self.model = B.sequential(m_head, *m_body, m_tail)

        self.m_up = nn.PixelShuffle(upscale_factor=sf)

    def forward(self, x, sigma):
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h / 2) * 2 - h)
        paddingRight = int(np.ceil(w / 2) * 2 - w)
        x = torch.nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x = self.m_down(x)
        # m = torch.ones(sigma.size()[0], sigma.size()[1], x.size()[-2], x.size()[-1]).type_as(x).mul(sigma)
        m = sigma.repeat(1, 1, x.size()[-2], x.size()[-1])
        x = torch.cat((x, m), 1)
        x = self.model(x)
        x = self.m_up(x)

        x = x[..., :h, :w]
        return x



###########################################NBNet###################################################
class BRDNet(nn.Module):
    def __init__(self, channels, num_of_layers=15):
        super(BRDNet, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        groups = 1
        L = []
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                                bias=False))
        layers.append(nn.BatchNorm2d(features))
        layers.append(nn.ReLU(inplace=True))
        # layers.append(B.ResBlock(features, features, kernel_size=kernel_size, stride=1, padding=padding, bias=False, mode='CRC', negative_slope=0.2))
        for _ in range(15):
            layers.append(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        # layers.append(B.ResBlock(features, features, kernel_size=kernel_size, stride=1, padding=padding, bias=False, mode='CRC', negative_slope=0.2))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding,
                                bias=False))  # 原来out_channels为channels


        L = []
        L.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                                bias=False))
        L.append(nn.BatchNorm2d(features))
        L.append(nn.ReLU(inplace=True))
        for i in range(7):
            L.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2))
            L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                                bias=False))
        L.append(nn.BatchNorm2d(features))
        L.append(nn.ReLU(inplace=True))
        for i in range(6):
            L.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2,
                               groups=groups,bias=False, dilation=2))
            L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                                bias=False))
        L.append(nn.BatchNorm2d(features))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding,
                           bias=False))
        self.BRDNet_first = nn.Sequential(*layers)
        self.BRDNet_second = nn.Sequential(*L)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=channels*2, out_channels=channels, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False))
    def forward(self, x):
        out1 = self.BRDNet_first(x)
        out2 = self.BRDNet_second(x)
        out1 = x - out1
        out2 = x - out2
        out = torch.cat([out1, out2], 1)
        out = self.conv1(out)
        out = x - out
        return out

'''
消融实验
少

'''
class DRFENetwithoutDEB(nn.Module):
    def __init__(self, channels, num_of_layers=15):
        super(DRFENetwithoutDEB, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        groups = 1
        layers = []
        kernel_size1 = 1
        '''
        #self.gamma = nn.Parameter(torch.zeros(1))
        '''
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.first_res = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
        self.second_res = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
        self.third_res = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
        self.fourth_res = nn.Sequential(
            nn.Conv2d(in_channels=features * 2, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
        self.fifth_res = nn.Sequential(
            nn.Conv2d(in_channels=features * 2, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_4 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_5 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_6 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_7 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_8 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_9 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_10 = nn.Sequential(
            nn.Conv2d(in_channels=features*2, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_11 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_12 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_13 = nn.Sequential(
            nn.Conv2d(in_channels=features*2, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_14 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_15 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_simple = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1,
                                      groups=groups, bias=False)
        self.conv1_16 = nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=1,
                                  groups=groups, bias=False)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv3 = nn.Conv2d(in_channels=channels*2, out_channels=channels, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.ReLU = nn.ReLU(inplace=True)
        self.Tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
                clip_b = 0.025
                w = m.weight.data.shape[0]
                for j in range(w):
                    if m.weight.data[j] >= 0 and m.weight.data[j] < clip_b:
                        m.weight.data[j] = clip_b
                    elif m.weight.data[j] > -clip_b and m.weight.data[j] < 0:
                        m.weight.data[j] = -clip_b
                m.running_var.fill_(0.01)
        # def _make_layers(self, block,features, kernel_size, num_of_layers, padding=1, groups=1, bias=False):
        #     layers = []
        #     for _ in range(num_of_layers):
        #         layers.append(block(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, groups=groups, bias=bias))
        #     return nn.Sequential(*layers)

    def forward(self, x):
        input = x
        # x11 = self.first_res(input)  # 第一层
        # x22 = self.second_res(x11)  # 第二层
        # # x0 = torch.cat([input,x0],1)    #与原图链接
        # x33 = self.third_res(x22)  # 第三层
        # x333 = x33
        # x33 = torch.cat([x11, x33], 1)
        # x44 = self.fourth_res(x33)
        # x44 = torch.cat([x22, x44], 1)
        # x55 = self.fifth_res(x44)
        # x55 = torch.cat([x333, x55], 1)
        # x55 = self.conv4(x55)
        x1 = self.conv1_1(x)
        x1 = self.conv1_2(x1)   #空洞卷积
        x1 = self.conv1_3(x1)
        x1 = self.conv1_4(x1)
        # x66=x1
        # x1 = torch.cat([x1,x55], 1)
        x1 = self.conv1_5(x1)   #空洞卷积
        x66=x1
        # x1 = torch.cat([x1,x55], 1)
        x1 = self.conv1_6(x1)
        x1 = self.conv1_7(x1)
        x1t = self.conv1_8(x1)
        # x66=x1
        # x1 = torch.cat([x1,x55], 1)
        x1 = self.conv1_9(x1t)  #空洞卷积
        x77=x1
        x1 = torch.cat([x1,x66], 1)
        x1 = self.conv1_10(x1)
        x1 = self.conv1_11(x1)
        x1 = self.conv1_12(x1)  #空洞卷积
        # x88=x1
        x1 = torch.cat([x1,x77], 1)
        x1 = self.conv1_13(x1)
        x1 = self.conv1_14(x1)
        x1 = self.conv1_15(x1)
        x1 = self.conv1_16(x1)
        out = torch.cat([x, x1], 1)
        out = self.Tanh(out)
        out = self.conv3(out)
        out = out * x1
        out2 = x - out
        return out2

class DRFENetwithoutRDB(nn.Module):
    def __init__(self, channels, num_of_layers=15):
        super(DRFENetwithoutRDB, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        groups = 1
        layers = []
        kernel_size1 = 1
        '''
        #self.gamma = nn.Parameter(torch.zeros(1))
        '''
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.first_res = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
        self.second_res = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
        self.third_res = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
        self.fourth_res = nn.Sequential(
            nn.Conv2d(in_channels=features * 2, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
        self.fifth_res = nn.Sequential(
            nn.Conv2d(in_channels=features * 2, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_4 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_5 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_6 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_7 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_8 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_9 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_10 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_11 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_12 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_13 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_14 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_15 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_simple = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1,
                                      groups=groups, bias=False)
        self.conv1_16 = nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=1,
                                  groups=groups, bias=False)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv3 = nn.Conv2d(in_channels=channels*2, out_channels=channels, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.ReLU = nn.ReLU(inplace=True)
        self.Tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
                clip_b = 0.025
                w = m.weight.data.shape[0]
                for j in range(w):
                    if m.weight.data[j] >= 0 and m.weight.data[j] < clip_b:
                        m.weight.data[j] = clip_b
                    elif m.weight.data[j] > -clip_b and m.weight.data[j] < 0:
                        m.weight.data[j] = -clip_b
                m.running_var.fill_(0.01)
        # def _make_layers(self, block,features, kernel_size, num_of_layers, padding=1, groups=1, bias=False):
        #     layers = []
        #     for _ in range(num_of_layers):
        #         layers.append(block(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, groups=groups, bias=bias))
        #     return nn.Sequential(*layers)

    def forward(self, x):
        input = x
        x11 = self.first_res(input)  # 第一层
        x22 = self.second_res(x11)  # 第二层
        # x0 = torch.cat([input,x0],1)    #与原图链接
        x33 = self.third_res(x22)  # 第三层
        x333 = x33
        x33 = torch.cat([x11, x33], 1)
        x44 = self.fourth_res(x33)
        x44 = torch.cat([x22, x44], 1)
        x55 = self.fifth_res(x44)
        x55 = torch.cat([x333, x55], 1)
        x55 = self.conv4(x55)
        # x1 = self.conv1_1(x55)
        # x1 = self.conv1_2(x1)   #空洞卷积
        # x1 = self.conv1_3(x1)
        # x1 = self.conv1_4(x1)
        # # x66=x1
        # # x1 = torch.cat([x1,x55], 1)
        # x1 = self.conv1_5(x1)   #空洞卷积
        # x66=x1
        # x1 = torch.cat([x1,x55], 1)
        # x1 = self.conv1_6(x1)
        # x1 = self.conv1_7(x1)
        # x1t = self.conv1_8(x1)
        # # x66=x1
        # # x1 = torch.cat([x1,x55], 1)
        # x1 = self.conv1_9(x1t)  #空洞卷积
        # x77=x1
        # x1 = torch.cat([x1,x66], 1)
        # x1 = self.conv1_10(x1)
        # x1 = self.conv1_11(x1)
        # x1 = self.conv1_12(x1)  #空洞卷积
        # # x88=x1
        # x1 = torch.cat([x1,x77], 1)
        # x1 = self.conv1_13(x1)
        x1 = self.conv1_14(x55)
        x1 = self.conv1_15(x1)
        x1 = self.conv1_16(x1)
        out = torch.cat([x, x1], 1)
        out = self.Tanh(out)
        out = self.conv3(out)
        out = out * x1
        out2 = x - out
        return out2





class DRFENetwithoutconcat(nn.Module):
    def __init__(self, channels, num_of_layers=15):
        super(DRFENetwithoutconcat, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        groups = 1
        layers = []
        kernel_size1 = 1
        '''
        #self.gamma = nn.Parameter(torch.zeros(1))
        '''
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.first_res = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
        self.second_res = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
        self.third_res = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
        self.fourth_res = nn.Sequential(
            nn.Conv2d(in_channels=features * 2, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
        self.fifth_res = nn.Sequential(
            nn.Conv2d(in_channels=features * 2, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_4 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_5 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_6 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_7 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_8 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_9 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_10 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_11 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_12 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_13 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_14 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_15 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_simple = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1,
                                      groups=groups, bias=False)
        self.conv1_16 = nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=1,
                                  groups=groups, bias=False)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=features*2, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv3 = nn.Conv2d(in_channels=channels*2, out_channels=channels, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.ReLU = nn.ReLU(inplace=True)
        self.Tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
                clip_b = 0.025
                w = m.weight.data.shape[0]
                for j in range(w):
                    if m.weight.data[j] >= 0 and m.weight.data[j] < clip_b:
                        m.weight.data[j] = clip_b
                    elif m.weight.data[j] > -clip_b and m.weight.data[j] < 0:
                        m.weight.data[j] = -clip_b
                m.running_var.fill_(0.01)
        # def _make_layers(self, block,features, kernel_size, num_of_layers, padding=1, groups=1, bias=False):
        #     layers = []
        #     for _ in range(num_of_layers):
        #         layers.append(block(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, groups=groups, bias=bias))
        #     return nn.Sequential(*layers)

    def forward(self, x):
        input = x
        x11 = self.first_res(input)  # 第一层
        x22 = self.second_res(x11)  # 第二层
        # x0 = torch.cat([input,x0],1)    #与原图链接
        x33 = self.third_res(x22)  # 第三层
        x333 = x33
        x33 = torch.cat([x11, x33], 1)
        x44 = self.fourth_res(x33)
        x44 = torch.cat([x22, x44], 1)
        x55 = self.fifth_res(x44)
        x55 = torch.cat([x333, x55], 1)
        x55 = self.conv4(x55)
        x1 = self.conv1_1(x55)
        x1 = self.conv1_2(x1)   #空洞卷积
        x1 = self.conv1_3(x1)
        x1 = self.conv1_4(x1)
        # x66=x1
        # x1 = torch.cat([x1,x55], 1)
        x1 = self.conv1_5(x1)   #空洞卷积
        x66=x1
        # x1 = torch.cat([x1,x55], 1)
        x1 = self.conv1_6(x1)
        x1 = self.conv1_7(x1)
        x1t = self.conv1_8(x1)
        # x66=x1
        # x1 = torch.cat([x1,x55], 1)
        x1 = self.conv1_9(x1t)  #空洞卷积
        x77=x1
        # x1 = torch.cat([x1,x66], 1)
        x1 = self.conv1_10(x1)
        x1 = self.conv1_11(x1)
        x1 = self.conv1_12(x1)  #空洞卷积
        # x88=x1
        # x1 = torch.cat([x1,x77], 1)
        x1 = self.conv1_13(x1)
        x1 = self.conv1_14(x1)
        x1 = self.conv1_15(x1)
        x1 = self.conv1_16(x1)
        out = torch.cat([x, x1], 1)
        out = self.Tanh(out)
        out = self.conv3(out)
        out = out * x1
        out2 = x - out
        return out2





class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)


    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))

###########################################NBNet###################################################
class BRDNet1(nn.Module):
    def __init__(self, channels, num_of_layers=15):
        super(BRDNet1, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        groups = 1
        L1 = []

        #=====================================================================================================
        self.first_res = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
        self.second_res = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
        self.third_res = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
        self.fourth_res = nn.Sequential(
            nn.Conv2d(in_channels=features * 2, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
        self.fifth_res = nn.Sequential(
            nn.Conv2d(in_channels=features * 2, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_4 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_5 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_6 = nn.Sequential(
            nn.Conv2d(in_channels=features*2, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_7 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_8 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_9 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_10 = nn.Sequential(
            nn.Conv2d(in_channels=features*2, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_11 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_12 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_13 = nn.Sequential(
            nn.Conv2d(in_channels=features*2, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_14 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_15 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_simple = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1,
                                      groups=groups, bias=False)
        self.conv1_16 = nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=1,
                                  groups=groups, bias=False)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=features * 2, out_channels=features, kernel_size=kernel_size, padding=1,
                      groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv3 = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1, stride=1, padding=0,
                               groups=1, bias=True)
        self.ReLU = nn.ReLU(inplace=True)
        self.Tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        # =====================================================================================================



        L1.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                                bias=False))
        L1.append(nn.BatchNorm2d(features))
        L1.append(nn.ReLU(inplace=True))
        for i in range(7):
            L1.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2))
            L1.append(nn.ReLU(inplace=True))
        L1.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                                bias=False))
        L1.append(nn.BatchNorm2d(features))
        L1.append(nn.ReLU(inplace=True))
        for i in range(6):
            L1.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2,
                               groups=groups,bias=False, dilation=2))
            L1.append(nn.ReLU(inplace=True))
        L1.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                                bias=False))
        L1.append(nn.BatchNorm2d(features))
        L1.append(nn.ReLU(inplace=True))
        L1.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding,
                           bias=False))

        self.BRDNet_second_one = nn.Sequential(*L1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=channels*2, out_channels=channels, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False))
    def forward(self, x):

        input = x
        x11 = self.first_res(input)  # 第一层
        x22 = self.second_res(x11)  # 第二层
        # x0 = torch.cat([input,x0],1)    #与原图链接
        x33 = self.third_res(x22)  # 第三层
        x333 = x33
        x33 = torch.cat([x11, x33], 1)
        x44 = self.fourth_res(x33)
        x44 = torch.cat([x22, x44], 1)
        x55 = self.fifth_res(x44)
        x55 = torch.cat([x333, x55], 1)
        x55 = self.conv4(x55)
        x1 = self.conv1_1(x55)
        x1 = self.conv1_2(x1)  # 空洞卷积
        x1 = self.conv1_3(x1)
        x1 = self.conv1_4(x1)
        # x66=x1
        # x1 = torch.cat([x1,x55], 1)
        x1 = self.conv1_5(x1)  # 空洞卷积
        x66 = x1
        x1 = torch.cat([x1, x55], 1)
        x1 = self.conv1_6(x1)
        x1 = self.conv1_7(x1)
        x1t = self.conv1_8(x1)
        # x66=x1
        # x1 = torch.cat([x1,x55], 1)
        x1 = self.conv1_9(x1t)  # 空洞卷积
        x1 = torch.cat([x1, x66], 1)
        x1 = self.conv1_10(x1)
        x1 = self.conv1_11(x1)
        x1 = self.conv1_16(x1)
        out1 = torch.cat([x, x1], 1)
        out1 = self.Tanh(out1)
        out1 = self.conv3(out1)
        out1 = out1 * x1



        out2 = self.BRDNet_second_one(input)

        out1 = x - out1
        out2 = x - out2
        out = torch.cat([out1, out2], 1)
        out = self.conv1(out)
        out = x - out
        return out



#####################################################################################################################
class BRDNet2(nn.Module):
    def __init__(self, channels, num_of_layers=15):
        super(BRDNet2, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        groups = 1
        L1 = []

        # =====================================================================================================
        self.first_res = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
        self.second_res = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
        self.third_res = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
        self.fourth_res = nn.Sequential(
            nn.Conv2d(in_channels=features * 2, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
        self.fifth_res = nn.Sequential(
            nn.Conv2d(in_channels=features * 2, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2,
                      groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1,
                      groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_4 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1,
                      groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_5 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2,
                      groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_6 = nn.Sequential(
            nn.Conv2d(in_channels=features * 2, out_channels=features, kernel_size=kernel_size, padding=1,
                      groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_7 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_8 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1,
                      groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_9 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2,
                      groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_10 = nn.Sequential(
            nn.Conv2d(in_channels=features * 2, out_channels=features, kernel_size=kernel_size, padding=1,
                      groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_11 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1,
                      groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_12 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2,
                      groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_13 = nn.Sequential(
            nn.Conv2d(in_channels=features * 2, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_14 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_15 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1,
                      groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_simple = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size,
                                      padding=1,
                                      groups=groups, bias=False)
        self.conv1_16 = nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=1,
                                  groups=groups, bias=False)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=features * 2, out_channels=features, kernel_size=kernel_size, padding=1,
                      groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv3 = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1, stride=1, padding=0,
                               groups=1, bias=True)
        self.ReLU = nn.ReLU(inplace=True)
        self.Tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        # =====================================================================================================

        L1.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                            bias=False))
        L1.append(nn.BatchNorm2d(features))
        L1.append(nn.ReLU(inplace=True))
        for i in range(7):
            L1.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2,
                                groups=groups,
                                bias=False, dilation=2))
            L1.append(nn.ReLU(inplace=True))
        L1.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                            bias=False))
        L1.append(nn.BatchNorm2d(features))
        L1.append(nn.ReLU(inplace=True))
        for i in range(6):
            L1.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2,
                                groups=groups, bias=False, dilation=2))
            L1.append(nn.ReLU(inplace=True))
        L1.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                            bias=False))
        L1.append(nn.BatchNorm2d(features))
        L1.append(nn.ReLU(inplace=True))
        L1.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding,
                            bias=False))

        self.BRDNet_second_one = nn.Sequential(*L1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False))

    def forward(self, x):

        input = x
        x11 = self.first_res(input)  # 第一层
        x22 = self.second_res(x11)  # 第二层
        # x0 = torch.cat([input,x0],1)    #与原图链接
        x33 = self.third_res(x22)  # 第三层
        x333 = x33
        x33 = torch.cat([x11, x33], 1)
        x44 = self.fourth_res(x33)
        x44 = torch.cat([x22, x44], 1)
        x55 = self.fifth_res(x44)
        x55 = torch.cat([x333, x55], 1)
        x55 = self.conv4(x55)
        x1 = self.conv1_1(x55)
        # x1 = self.conv1_2(x1)  # 空洞卷积
        x1 = self.conv1_3(x1)
        x1 = self.conv1_4(x1)
        # x66=x1
        # x1 = torch.cat([x1,x55], 1)
        # x1 = self.conv1_5(x1)  # 空洞卷积
        x66 = x1
        x1 = torch.cat([x1, x55], 1)
        x1 = self.conv1_6(x1)
        x1 = self.conv1_7(x1)
        x1 = self.conv1_8(x1)
        # x66=x1
        # x1 = torch.cat([x1,x55], 1)
        # x1 = self.conv1_9(x1t)  # 空洞卷积
        x1 = torch.cat([x1, x66], 1)
        x1 = self.conv1_10(x1)
        x1 = self.conv1_11(x1)
        x1 = self.conv1_16(x1)
        out1 = torch.cat([x, x1], 1)
        out1 = self.Tanh(out1)
        out1 = self.conv3(out1)
        out1 = out1 * x1

        out2 = self.BRDNet_second_one(input)

        out1 = x - out1
        out2 = x - out2
        out = torch.cat([out1, out2], 1)
        out = self.conv1(out)
        out = x - out
        return out