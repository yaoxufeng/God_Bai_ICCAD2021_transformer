# coding:utf-8

'''
decoder part
copied the config from https://github.com/fudan-zvg/SETR/blob/main/configs/_base_/models/setr_naive_pup.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import ml_collections


class VisionTransformer_Naive(nn.Module):
    '''
    decoder Naive upsampling
    
    params:
        input: a standard 3D feature map (H/16, W/16, C)
        architecture: 1x1 conv + BN + relu + 1x1 conv,
        bilinearly upsample to the full image resolution
        followed by a classifier on pixel-wise classification
    
    '''
    def __init__(self, config, align_corners=True):
        super(VisionTransformer_Naive, self).__init__()
        self.img_size = config.img_size  # default 768
        self.embed_dim = config.embed_dim  # default 1024
        self.num_classes = config.num_classes
        self.align_corners = align_corners
        
        # two layer conv1x1
        self.conv_0 = nn.Conv2d(self.embed_dim, 256, kernel_size=3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(256, self.num_classes, kernel_size=3, stride=1, padding=1)
        
        # batch norm
        self.bn0 = nn.BatchNorm2d(256)
    
    def _reshape(self, x):
        '''
        reshape (N, HW/256, C) to (N, H/16, W/16, C)
        '''
        # x = x[0]  # directly select index 0
        n, c, hw = x.size(0), x.size(2), x.size(1)
        h = w = int(math.sqrt(hw))
        x = x.transpose(1, 2).reshape(n, c, h, w)
        
        return x
    
    def forward(self, x):
        x = self._reshape(x)  # (1, 1024, 48, 48)
        x = self.conv_0(x)  # (1, 256, 48, 48)

        x = self.bn0(x)
        x = F.relu(x, inplace=True)
        
        # (1, 256, 192, 192)
        x = F.interpolate(
            x, size=x.shape[-1]*4, mode='bilinear', align_corners=self.align_corners)

        x = self.conv_1(x)  # (1, 19, 192, 192)

        # (1, 19, 768, 768])
        x = F.interpolate(
            x, size=self.img_size, mode='bilinear', align_corners=self.align_corners)

        return x

    
class VisionTransformer_PUP(nn.Module):
    '''
    decoder Progressive UPsampling
    '''
    def __init__(self, config, align_corners=True):
        super(VisionTransformer_PUP, self).__init__()
        self.img_size = config.img_size  # default 768
        self.embed_dim = config.embed_dim  # default 1024
        self.num_classes = config.num_classes
        self.align_corners = align_corners

        # two layer conv1x1
        self.conv_0 = nn.Conv2d(self.embed_dim, 256, kernel_size=3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_4 = nn.Conv2d(256, self.num_classes, kernel_size=1, stride=1)
        
        # batch norm
        self.bn0 = nn.BatchNorm2d(256)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(256)

    def _reshape(self, x):
        '''
        reshape (N, HW/256, C) to (N, H/16, W/16, C)
        '''
        # x = x[0]  # directly select index 0
        n, c, hw = x.size(0), x.size(2), x.size(1)
        h = w = int(math.sqrt(hw))
        x = x.transpose(1, 2).reshape(n, c, h, w)
    
        return x
    
    def forward(self, x):
        x = self._reshape(x)
        x = self.conv_0(x)
        x = self.bn0(x)
        x = F.relu(x, inplace=True)
        x = F.interpolate(
            x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)
        x = self.conv_1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = F.interpolate(
            x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)
        x = self.conv_2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = F.interpolate(
            x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)
        x = self.conv_3(x)
        x = self.bn3(x)
        x = F.relu(x, inplace=True)
        x = self.conv_4(x)
        x = F.interpolate(
            x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)
        
        return x
    
    
class VisionTransformer_MLA(nn.Module):
    '''
    decoder Multi-Level feature Aggregation
    累了，先整俩
    '''
    def __init__(self, config, align_corners=True):
        super(VisionTransformer_MLA, self).__init__()
        
        self.img_size = config.img_size  # default 768
        self.embed_dim = config.embed_dim  # default 1024
        self.num_classes = config.num_classes
        self.align_corners = align_corners
        
        self.conv_0 = nn.Conv2d(self.embed_dim, 256, kernel_size=3, stride=1, padding=1)
        self.head0 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(),
                                   nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU())
        self.head1 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(),
                                   nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU())
        self.head2 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(),
                                   nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU())
        self.head3 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(),
                                   nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU())
        
        self.cls = nn.Conv2d(4 * 256, self.num_classes, 3, padding=1)
        
    
    def _reshape(self, x):
        '''
        reshape (N, HW/256, C) to (N, H/16, W/16, C)
        '''
        x = x[0]  # directly select index 0
        n, c, hw = x.size(0), x.size(2), x.size(1)
        h = w = int(math.sqrt(hw))
        x = x.transpose(1, 2).reshape(n, c, h, w)
    
        return x
    
    def forward(self, x):
        x = self._reshape(x)
        x = self.conv_0(x)
        
        x1 = F.interpolate(self.head0(x), 4 * x.shape[-1], mode='bilinear', align_corners=True)
        x2 = F.interpolate(self.head1(x), 4 * x.shape[-1], mode='bilinear', align_corners=True)
        x3 = F.interpolate(self.head2(x), 4 * x.shape[-1], mode='bilinear', align_corners=True)
        x4 = F.interpolate(self.head3(x), 4 * x.shape[-1], mode='bilinear', align_corners=True)
        
        x = torch.cat([x1, x2, x3, x4], dim=1)
    
        x = self.cls(x)
        x = F.interpolate(x, size=self.img_size, mode='bilinear',
                          align_corners=self.align_corners)
        
        return x
    
if __name__ == "__main__":
    
    config = ml_collections.ConfigDict()
    config.img_size = 768
    config.embed_dim = 1024
    config.patch_size = 16
    config.num_classes = 19
    
    model = VisionTransformer_MLA(config)
    model = model.cuda()
    
    # bz = [1, 4, 16]

    img = torch.randn(1, 2304, 1024)
    out = model(img)
    print("out", out.shape)