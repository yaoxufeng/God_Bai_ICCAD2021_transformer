# coding:utf-8

'''
the encoder part is a pure Vit transformer
copied the config from the official https://github.com/fudan-zvg/SETR/blob/main/configs/_base_/models/setr_naive_pup.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import ml_collections
import time
from functools import partial
from itertools import repeat


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)  # 1024, 4096
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)  # 4096, 1024
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)  # (1, 2304, 4096)
        x = self.act(x)  # (1, 2304, 4096)
        x = self.drop(x)
        x = self.fc2(x)  # (1, 2304, 1024)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape  # B: 1, N: 2304, C: 1024
        
        # q: (1, 16, 2304, 64); k: (1, 16, 2304, 64); v: (1, 16, 2304, 64);
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                      C // self.num_heads).permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # 1, 16, 2304, 2304
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # 1, 2304, 1024
        x = self.proj(x)   # 1, 2304, 1024
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)  # 1024 * 4 = 4096
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,  # in_features: 1024, hidden_features: 4096
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # (1, 2304, 1024)
        x = x + self.mlp(self.norm2(x))  # (1, 2304, 1024)

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = tuple(repeat(img_size, 2))
        patch_size = tuple(repeat(patch_size, 2))
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape  # (1 3 768 768)
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x)  # (1, 1024, 48, 48)
        return x


class VisionTransformer_Encoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, config, qkv_bias=True, qk_scale=None, drop_rate=0.1, attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_cfg=None,
                 pos_embed_interp=False, random_init=False, align_corners=False, **kwargs):
        super(VisionTransformer_Encoder, self).__init__(**kwargs)
        
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.in_chans = config.in_chans
        self.embed_dim = config.embed_dim
        self.depth = config.depth
        self.num_heads = config.num_heads
        self.num_classes = config.num_classes
        self.mlp_ratio = config.mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.hybrid_backbone = hybrid_backbone
        self.norm_layer = norm_layer
        self.norm_cfg = norm_cfg
        self.pos_embed_interp = pos_embed_interp
        self.random_init = random_init
        self.align_corners = align_corners

        self.num_stages = self.depth
        self.out_indices = tuple(range(self.num_stages))

        self.patch_embed = PatchEmbed(
            img_size=self.img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=self.embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(
            1, self.num_patches, self.embed_dim))
        self.pos_drop = nn.Dropout(p=self.drop_rate)

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate,
                                                self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                drop=self.drop_rate, attn_drop=self.attn_drop_rate, drop_path=dpr[i], norm_layer=self.norm_layer)
            for i in range(self.depth)])


    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # (1, 1024, 48, 48)

        x = x.flatten(2).transpose(1, 2)  # (1, 2304, 1024)

        x = x + self.pos_embed

        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)  # (1, 2304, 1024)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
    

if __name__ == "__main__":
    
    config = ml_collections.ConfigDict()
    config.img_size = 768
    config.patch_size = 16
    config.in_chans = 3
    config.embed_dim = 1024
    config.depth = 24
    config.num_heads = 16
    config.num_classes = 19
    config.mlp_ratio = 4
    
    model = VisionTransformer_Encoder(config)
    model = model.cuda()
    
    bz_lst = [16]
    
    for bz in bz_lst:
    
        # torch.cuda.empty_cache()  # release gpu memory

        img = torch.randn(bz, 3, 768, 768)  # origin input image
        img = img.cuda()
        
        transformer_input = torch.randn(bz, 2304, 1024)  # embedding 2304 = 768 * 768 / (16 * 16), 1024 is the embedding dim
        transformer_input = transformer_input.cuda()
        
        # ========================= patch embedding time =======================
        
        # start_time = time.time()
        #
        # for _ in range(1000):
        #     out1 = model.patch_embed(img)  # total patch emebdding time
        #
        # out_time = time.time() - start_time
        # print("bz {} the total patch embedding  inference time is {}".format(bz, out_time))

        # ========================= encoder-multi-head-attention inference time =======================

        start_time = time.time()

        for _ in range(1):

            # total encoder-block-attention
            for i, blk in enumerate(model.blocks):
                out2 = blk.attn(transformer_input)

        out_time = time.time() - start_time
        print("bz {} the total encoder-multi-head-attention inference time is {}".format(bz, out_time))

        # ========================= encoder mlp inference time =======================

        # start_time = time.time()
        #
        # for _ in range(100):
        #
        #     # total encoder-block-attention
        #     for i, blk in enumerate(model.blocks):
        #         out2 = blk.mlp(transformer_input)
        #
        # out_time = time.time() - start_time
        # print("bz {} the total encoder mlp inference time is {}".format(bz, out_time))