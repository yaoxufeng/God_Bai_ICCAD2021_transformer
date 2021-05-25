# coding:utf-8

'''
SETR from https://arxiv.org/pdf/2012.15840.pdf
The original source code is from https://github.com/fudan-zvg/SETR
'''

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import ml_collections

from encoder import VisionTransformer_Encoder
from decoder import VisionTransformer_Naive

class SETR_Transformer(nn.Module):
    '''
    setr transformer
    '''
    def __init__(self, encoder_config, decoder_config):
        super(SETR_Transformer, self).__init__()
        self.encoder = VisionTransformer_Encoder(encoder_config)
        self.decoder = VisionTransformer_Naive(decoder_config)
        
    def forward(self, x):
        x = self.encoder(x)[0]  # choose index 0 as default index
        # print("x_encoder", x[0].shape)
        x = self.decoder(x)
        
        return x
    
    
def encoder_config():
    config = ml_collections.ConfigDict()
    config.img_size = 768
    config.patch_size = 16
    config.in_chans = 3
    config.embed_dim = 1024
    config.depth = 24
    config.num_heads = 16
    config.num_classes = 19
    config.mlp_ratio = 4
    return config


def decoder_config():
    config = ml_collections.ConfigDict()
    config.img_size = 768
    config.embed_dim = 1024
    config.patch_size = 16
    config.num_classes = 19
    return config
    

if __name__ == "__main__":
    encoder_config = encoder_config()
    decoder_config = decoder_config()
    
    model = SETR_Transformer(encoder_config, decoder_config)
    model = model.cuda()

    # test image
    
    bz_lst = [1, 4, 16]
    
    for bz in bz_lst:
    
        image = torch.randn(bz, 3, 768, 768)  # default 224x224
        image = image.cuda()
        
        decoder_input = torch.randn(bz, 2304, 1024)  # default decode input
        decoder_input = decoder_input.cuda()
    
        # ========================= encoder time =========================
        # start_time = time.time()
        #
        # for _ in range(1):
        #     out1 = model.encoder(image)  # total encoder time
        #
        # end_time = time.time() - start_time
        # print("bz {} the whole encoder runing time is {}".format(bz, end_time))

        # ========================= decoder time =========================
        start_time = time.time()

        for _ in range(1000):
            out2 = model.decoder(decoder_input)  # total decoder inference time

        end_time = time.time() - start_time
        print("bz {} the whole decoder runing time is {}".format(bz, end_time))