import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cbam import *


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        # self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            out = F.relu(out, inplace=True)
        return out
    
class DenseConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseConv2d, self).__init__()
        self.dense_conv = ConvLayer(in_channels, out_channels, kernel_size, stride)
    
    def forward(self, x):
        out = self.dense_conv(x)
        out = torch.cat([x, out], 1)
        return out

class DenseBlock(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        super(DenseBlock, self).__init__()
        out_channels_def = 16
        denseblock=[]
        denseblock += [
            DenseConv2d(in_channels, out_channels_def, kernel_size, stride),
            DenseConv2d(in_channels+out_channels_def, out_channels_def, kernel_size, stride),
            DenseConv2d(in_channels+out_channels_def*2, out_channels_def, kernel_size, stride)]
        self.denseblock = nn.Sequential(*denseblock)
    
    def forward(self,x):
        out = self.denseblock(x)
        return out

class DenseFuse_net(torch.nn.Module):
    def __init__(self, input_nc=1, output_nc=1):
        super(DenseFuse_net, self).__init__()
        nb_filter = [16,128,64,32,16]
        kernel_size = 3
        stride = 1

        ##encoder1
        self.conv1 = ConvLayer(input_nc, nb_filter[0], kernel_size, stride)
        self.DB1 = DenseBlock(nb_filter[0], kernel_size, stride)

        ##cbam
        self.cbam = CBAM(nb_filter[1], 16 )

        ##decoder
        self.conv2 = ConvLayer(nb_filter[1], nb_filter[1], kernel_size, stride)
        self.conv3 = ConvLayer(nb_filter[1], nb_filter[2], kernel_size, stride)
        self.conv4 = ConvLayer(nb_filter[2], nb_filter[3], kernel_size, stride)
        self.conv5 = ConvLayer(nb_filter[3], nb_filter[4], kernel_size, stride)
        self.conv6 = ConvLayer(nb_filter[4], output_nc, kernel_size, stride, is_last=True)


    def forward(self, s0, dolp):
        s_out = self.conv1(s0)
        s_out = self.DB1(s_out)

        d_out = self.conv1(dolp)
        d_out = self.DB1(d_out)

        output = torch.cat([s_out, d_out], 1)
        output = self.cbam(output)
        
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)

        return output

    # initialize weights
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m,nn.Linear):
                torch.nn.init.normal_(m.weight.data,0,0.01)
                # m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()
