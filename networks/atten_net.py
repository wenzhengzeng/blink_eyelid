from .resnet import *
import torch.nn as nn
import torch
from .globalNet import globalNet
from .refineNet import refineNet

__all__ = ['CPN18', 'CPN34', 'CPN50', 'CPN101']

class CPN(nn.Module):
    def __init__(self, resnet, output_shape, num_class, small_channel=False, pretrained=True):
        super(CPN, self).__init__()
        channel_settings = [2048, 1024, 512, 256]
        if small_channel:
            channel_settings = [512, 256, 128, 64]
        self.resnet = resnet
        self.global_net = globalNet(channel_settings, output_shape, num_class)
        self.refine_net = refineNet(256, output_shape, num_class)
    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
    def forward(self, x, phase='train'):
        res_out = self.resnet(x)
        global_fms, global_outs = self.global_net(res_out, phase)
        refine_out = self.refine_net(global_fms)

        return global_outs, refine_out

def CPN18(out_size,num_class,pretrained=True):
    res18 = resnet18(pretrained=pretrained)
    model = CPN(res18, output_shape=out_size,num_class=num_class, small_channel=True, pretrained=pretrained)
    return model

def CPN34(out_size,num_class,pretrained=True):
    res34 = resnet34(pretrained=pretrained)
    model = CPN(res34, output_shape=out_size,num_class=num_class, small_channel=True, pretrained=pretrained)
    return model

def CPN50(out_size,num_class,pretrained=True):
    res50 = resnet50(pretrained=pretrained)
    model = CPN(res50, output_shape=out_size,num_class=num_class, small_channel=False, pretrained=pretrained)
    return model

def CPN101(out_size,num_class,pretrained=True):
    res101 = resnet101(pretrained=pretrained)
    model = CPN(res101, output_shape=out_size,num_class=num_class, small_channel=False, pretrained=pretrained)
    return model
