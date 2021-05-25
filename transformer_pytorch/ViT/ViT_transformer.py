# coding:utf-8

'''
ViT transformer part
'''

import copy
import logging
import math
import time
import ml_collections

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

logger = logging.getLogger(__name__)


def np2th(weights, conv=False):
	"""Possibly convert HWIO to OIHW."""
	if conv:
		weights = weights.transpose([3, 2, 0, 1])
	return torch.from_numpy(weights)


def swish(x):
	return x * torch.sigmoid(x)


# the default ac function is gelu
ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


def get_b16_config():
	"""Returns the ViT-B/16 configuration."""
	config = ml_collections.ConfigDict()
	config.patches = ml_collections.ConfigDict({'size': (16, 16)})
	config.hidden_size = 768
	config.transformer = ml_collections.ConfigDict()
	config.transformer.mlp_dim = 3072
	config.transformer.num_heads = 12
	config.transformer.num_layers = 12
	config.transformer.attention_dropout_rate = 0.0
	config.transformer.dropout_rate = 0.1
	config.classifier = 'token'
	config.representation_size = None
	return config


class Attention(nn.Module):
	'''Attention layer'''
	
	def __init__(self, config, vis):
		super(Attention, self).__init__()
		self.vis = vis
		self.num_attention_heads = config.transformer["num_heads"]  # 12
		self.attention_head_size = int(config.hidden_size / self.num_attention_heads)  # 64 = 768 / 12
		self.all_head_size = self.num_attention_heads * self.attention_head_size  # 768 = 12 * 64
	
		self.query = Linear(config.hidden_size, self.all_head_size)  # (768 768)
		self.key = Linear(config.hidden_size, self.all_head_size)  # (768, 768)
		self.value = Linear(config.hidden_size, self.all_head_size)  # (768, 768)
		
		self.out = Linear(config.hidden_size, config.hidden_size)  # (768, 768)
		self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
		self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])
		
		self.softmax = Softmax(dim=-1)
	
	def transpose_for_scores(self, x):
		new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
		x = x.view(*new_x_shape)  # (1, 197, 12, 64)
		return x.permute(0, 2, 1, 3)  # (1, 12, 197, 64)
	
	def forward(self, hidden_states):
		mixed_query_layer = self.query(hidden_states)  # (1, 768, 768)
		mixed_key_layer = self.key(hidden_states)  # (1, 768, 768)
		mixed_value_layer = self.value(hidden_states)  # (1, 768, 768)
		
		query_layer = self.transpose_for_scores(mixed_query_layer)  # (1, 12, 197, 64)
		key_layer = self.transpose_for_scores(mixed_key_layer)  # (1, 12, 197, 64)
		value_layer = self.transpose_for_scores(mixed_value_layer)  # (1, 12, 197, 64)
		
		attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (1, 12, 197, 197)
		attention_scores = attention_scores / math.sqrt(self.attention_head_size)
		attention_probs = self.softmax(attention_scores)
		weights = attention_probs if self.vis else None
		attention_probs = self.attn_dropout(attention_probs)
		
		context_layer = torch.matmul(attention_probs, value_layer)  # (1, 12, 197, 64)
		context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
		new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
		context_layer = context_layer.view(*new_context_layer_shape)  # (1, 197, 768)
		attention_output = self.out(context_layer)  # (1, 197, 768)
		attention_output = self.proj_dropout(attention_output)
		
		return attention_output, weights


class Mlp(nn.Module):
	def __init__(self, config):
		super(Mlp, self).__init__()
		self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])  #  (768, 3072)
		self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)  # (3072, 768)
		self.act_fn = ACT2FN["gelu"]
		self.dropout = Dropout(config.transformer["dropout_rate"])
		
		self._init_weights()
	
	def _init_weights(self):
		nn.init.xavier_uniform_(self.fc1.weight)
		nn.init.xavier_uniform_(self.fc2.weight)
		nn.init.normal_(self.fc1.bias, std=1e-6)
		nn.init.normal_(self.fc2.bias, std=1e-6)
	
	def forward(self, x):
		x = self.fc1(x)
		x = self.act_fn(x)
		x = self.dropout(x)
		x = self.fc2(x)
		x = self.dropout(x)
		
		return x


class Embeddings(nn.Module):
	"""Construct the embeddings from patch, position embeddings."""
	
	def __init__(self, config, img_size, in_channels=3):
		super(Embeddings, self).__init__()
		img_size = _pair(img_size)  # (224, 224)
		patch_size = _pair(config.patches["size"])  # (16, 16)
		n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])  # 196 = (224*224) / (16*16)
		
		self.patch_embeddings = Conv2d(in_channels=in_channels,  # 3
		                               out_channels=config.hidden_size,  # 768, embedding feature
		                               kernel_size=patch_size,  # (16, 16)
		                               stride=patch_size)
		
		# (1, 196+1, 768)
		self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches + 1, config.hidden_size))
		
		# (1, 1, 768)
		self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
		
		self.dropout = Dropout(config.transformer["dropout_rate"])
	
	def forward(self, x):
		B = x.shape[0]  # 1 batch size x: (1, 3, 224, 224)
		cls_tokens = self.cls_token.expand(B, -1, -1)  # (1, 1, 768)
		
		x = self.patch_embeddings(x)  # (1, 768, 14, 14)  14 = 224 / 16
		x = x.flatten(2)  # (1, 768, 196)
		x = x.transpose(-1, -2)  # (1, 196, 768)
		x = torch.cat((cls_tokens, x), dim=1)  # (1, 197, 768)
		
		embeddings = x + self.position_embeddings  # (1, 197, 768)
		embeddings = self.dropout(embeddings)
		
		return embeddings


class Block(nn.Module):
	def __init__(self, config, vis):
		super(Block, self).__init__()
		self.hidden_size = config.hidden_size
		self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
		self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
		self.ffn = Mlp(config)
		self.attn = Attention(config, vis)
	
	def forward(self, x):
		h = x  #  (1, 197, 768)
		x = self.attention_norm(x)  # (1, 197, 768)
		x, weights = self.attn(x)  # (1, 197, 768)
		x = x + h  # (1, 197, 768)

		h = x
		x = self.ffn_norm(x)  # (1, 197, 768)

		x = self.ffn(x)  # (1, 197, 768)

		x = x + h
		return x, weights


class Encoder(nn.Module):
	'''transformer encoder'''
	def __init__(self, config, vis):
		super(Encoder, self).__init__()
		self.vis = vis
		self.layer = nn.ModuleList()
		self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
		for _ in range(config.transformer["num_layers"]):
			layer = Block(config, vis)
			self.layer.append(copy.deepcopy(layer))

	def forward(self, hidden_states):
		attn_weights = []
		for layer_block in self.layer:
			hidden_states, weights = layer_block(hidden_states)
			if self.vis:
				attn_weights.append(weights)
		encoded = self.encoder_norm(hidden_states)
		return encoded, attn_weights


class Transformer(nn.Module):
	'''transformer part'''
	def __init__(self, config, img_size, vis):
		super(Transformer, self).__init__()
		self.embeddings = Embeddings(config, img_size=img_size)
		self.encoder = Encoder(config, vis)
	
	
	def forward(self, input_ids):
		embedding_output = self.embeddings(input_ids)  # (1, 197, 768)
		encoded, attn_weights = self.encoder(embedding_output)  # ecoded: (1, 197, 768)
		return encoded, attn_weights


class VisionTransformer(nn.Module):
	'''the whole ViT'''
	
	def __init__(self, config, img_size=224, num_classes=1000, zero_head=False, vis=False):
		super(VisionTransformer, self).__init__()
		self.num_classes = num_classes
		self.zero_head = zero_head
		self.classifier = config.classifier
		
		self.transformer = Transformer(config, img_size, vis)  # transformer
		self.head = Linear(config.hidden_size, num_classes)  # classification head
	
	def forward(self, x, labels=None):
		x, attn_weights = self.transformer(x)
		logits = self.head(x[:, 0])
		
		if labels is not None:
			loss_fct = CrossEntropyLoss()
			loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
			return loss
		else:
			return logits, attn_weights


if __name__ == "__main__":
	configs = get_b16_config()  # get configs
	print(configs)
	model = VisionTransformer(configs)  # init model
	model = model.cuda()
	# print("===============model================\n {}".format(model))  # print model
	
	# test image
	
	bz_lst = [1, 4, 16]
	
	for bz in bz_lst:
		# bz = 8
		image = torch.randn(bz, 3, 224, 224)  # default 224x224
		image = image.cuda()
		
		embeddings = torch.randn(bz, 197, 768)  # default embedding
		embeddings = embeddings.cuda()
		
		# calculate time
		
		# ============= patch embedding inference time ==================
		start_time = time.time()
		
		for _ in range(1000):
			out2 = model.transformer.embeddings(image)  # transformer embedding inference time

		end_time = time.time() - start_time
		# print("out is {}".format(out0[0].shape))
		print("bz {} the whole patch embedding runing time is {}".format(bz, end_time))
		
		# ============= encoder-multi-head-attention inference time ==================
		start_time = time.time()
		
		for _ in range(1000):
			for encoder_layer in model.transformer.encoder.layer:
				out5 = encoder_layer.attn(embeddings)

		end_time = time.time() - start_time
		# print("out is {}".format(out0[0].shape))
		print("bz {} the whole encoder-multi-head-attention runing time is {}".format(bz, end_time))
		
		# ============= encoder-encoder-mlp inference time ==================
		start_time = time.time()
		
		for _ in range(1000):
			
			for encoder_layer in model.transformer.encoder.layer:
				out6 = encoder_layer.ffn(embeddings)
		
		end_time = time.time() - start_time
		# print("out is {}".format(out0[0].shape))
		print("bz {} the whole encoder-encoder-mlp inference time runing time is {}".format(bz, end_time))
		
		# ============= classification inference time ==================
		start_time = time.time()
		
		for _ in range(1000):
			out7 = model.head(embeddings)  # classfication head
		
		end_time = time.time() - start_time
		# print("out is {}".format(out0[0].shape))
		print("bz {} the whole classifcation runing time is {}".format(bz, end_time))