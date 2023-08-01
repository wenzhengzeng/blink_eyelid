# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import math
import numpy as np



class BlinkEyelidNet(nn.Module):
    def __init__(self, cfg):
        super(BlinkEyelidNet, self).__init__()
        self.upsam = nn.UpsamplingBilinear2d(scale_factor=4)
        self.hi_channel = 128
        self.batch_size = cfg.batch_size
        self.time_size = cfg.time_size
        self.conv1 = nn.Conv2d(3, 24, 5, stride=3, bias=False)
        self.pool1 = nn.MaxPool2d(2, 2, padding=0)
        self.bn1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 48, 3, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(48)
        self.conv3 = nn.Conv2d(48, 80, 3, stride=2, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn3 = nn.BatchNorm2d(80)
        self.lstm = nn.LSTM(input_size=80 * 2, hidden_size=self.hi_channel, num_layers=2, dropout=0.5)
        self.fc6 = nn.Linear(self.hi_channel*2, 2,bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, image, height, width, pos, heatmap, phase='train'):
        datatemp = []
        img_height = image.shape[2]
        img_width = image.shape[3]
        heatall = self.upsam(heatmap)
        bbox = np.int64(pos)
        bbox[:, 0] = np.int64(bbox[:, 0] - height * 0.5)
        bbox[:, 1] = np.int64(bbox[:, 1] - width * 0.5)
        image = image.cuda()

        heatall = heatall.cuda()
        for img, bb, heat in zip(image, bbox, heatall):

            heat = torch.sigmoid(heat)
            
            heat = heat.repeat(3, 1, 1)
            img = img * heat
            img_temp = torch.zeros(3, img_height + 100, img_width + 100)  # torch.Size([3, 356, 292])
            img_temp[:, 50:50 + img_height, 50:50 + img_width] = img
            bb[0] = 50 + bb[0]  # x --> 192    
            bb[1] = 50 + bb[1]  # y --> 256

            img_process = img_temp[:, bb[1]:bb[1] + height, bb[0]:bb[0] + width]
            datatemp.append(img_process)

        x = torch.stack(datatemp).cuda()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.pool1(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.pool1(out)
        out = torch.squeeze(out)
        inputs = out.reshape(-1, self.time_size, 80).transpose(1, 0)
        feature = inputs[1:10, :, :]
        differ = feature - inputs[0:9, :, :]
        inputs = torch.cat((feature, differ), 2)
        inputs = torch.nn.functional.normalize(inputs, dim=2)
        outputs, _ = self.lstm(inputs)
        h_state_1 = outputs[-1]
        h_state_2 = outputs[-2] 

        h = torch.cat((h_state_1, h_state_2), 1)

        logits = self.fc6(h)

        return logits, h