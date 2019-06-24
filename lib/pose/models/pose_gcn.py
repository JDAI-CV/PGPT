# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Modifed based on Bin Xiao's code: https://github.com/microsoft/human-pose-estimation.pytorch
# ------------------------------------------------------------------------------


import os
import logging

import torch
import torch.nn as nn
from torch.nn import Parameter
import math
import torch.nn.functional as F
from .graph_simple import GraphConvolution


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PoseResNet(nn.Module):

	def __init__(self, block, layers, cfg, flag=2, **kwargs):
		super(PoseResNet, self).__init__()
		self.feature_flag = True
		self.inplanes = 64
		self.flag = flag
		self.num_classes = 3594
		self.input_height, self.input_width = 384, 288
		extra = cfg.MODEL.EXTRA
		self.deconv_with_bias = extra.DECONV_WITH_BIAS
		self.indices = torch.LongTensor([0,1,2,5,6,7,8,9,10,11,12,13,14,15,16])
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
							   bias=False)
		self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

		# used for deconv layers
		self.deconv_layers = self._make_deconv_layer(
			extra.NUM_DECONV_LAYERS,
			extra.NUM_DECONV_FILTERS,
			extra.NUM_DECONV_KERNELS,
		)

		self.final_layer = nn.Conv2d(
			in_channels=extra.NUM_DECONV_FILTERS[-1],
			out_channels=cfg.MODEL.NUM_JOINTS,
			kernel_size=extra.FINAL_CONV_KERNEL,
			stride=1,
			padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
		)
		for p in self.parameters():
			p.requires_grad = False

		self.embedding_layer = nn.Sequential(self._make_embedding_layer(block, 64, 1, stride=1),
												nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
											)
												
		self.pose_conv = nn.Sequential(nn.Conv2d(17, 256, kernel_size=3, stride=2, padding=1, bias=False),
										nn.BatchNorm2d(256),
										nn.ReLU(inplace=True),
										nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
										nn.BatchNorm2d(256),
										nn.ReLU(inplace=True),
										nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
										nn.BatchNorm2d(256),
										nn.ReLU(inplace=True))
		self.graph_layer1 = GraphConvolution(in_features=9*256, out_features=9*256)
		self.graph_layer2 = GraphConvolution(in_features=9*256, out_features=9*256)
		self.fc_feature = nn.Linear(256 * int(self.input_height/32) * int(self.input_width/32), 2048)
		self.fc_feature_align = nn.Linear(15 * 9 * 256, 2048)
		self.classification = nn.Linear(2048, self.num_classes)
		#self.angleLinear = AngleLinear(512, self.num_classes)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)
		
	def _make_embedding_layer(self, block, planes, blocks, stride=1):
		downsample = nn.Sequential(
			nn.Conv2d(2048, planes * block.expansion,
						 kernel_size=1, stride=stride, bias=False),
			nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
		)

		layers = []
		layers.append(block(2048, planes, stride, downsample))
		inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(inplanes, planes))

		return nn.Sequential(*layers)
	
	def set_bn_fix(self):
		def set_bn_eval(m):
			classname = m.__class__.__name__
			if classname.find('BatchNorm') != -1:
			  m.eval()
		self.bn1.apply(set_bn_eval)
		self.layer1.apply(set_bn_eval)
		self.layer2.apply(set_bn_eval)
		self.layer3.apply(set_bn_eval)
		self.layer4.apply(set_bn_eval)
		self.deconv_layers.apply(set_bn_eval)
		self.final_layer.apply(set_bn_eval)	

	def _get_deconv_cfg(self, deconv_kernel, index):
		if deconv_kernel == 4:
			padding = 1
			output_padding = 0
		elif deconv_kernel == 3:
			padding = 1
			output_padding = 1
		elif deconv_kernel == 2:
			padding = 0
			output_padding = 0

		return deconv_kernel, padding, output_padding

	def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
		assert num_layers == len(num_filters), \
			'ERROR: num_deconv_layers is different len(num_deconv_filters)'
		assert num_layers == len(num_kernels), \
			'ERROR: num_deconv_layers is different len(num_deconv_filters)'

		layers = []
		for i in range(num_layers):
			kernel, padding, output_padding = \
				self._get_deconv_cfg(num_kernels[i], i)

			planes = num_filters[i]
			layers.append(
				nn.ConvTranspose2d(
					in_channels=self.inplanes,
					out_channels=planes,
					kernel_size=kernel,
					stride=2,
					padding=padding,
					output_padding=output_padding,
					bias=self.deconv_with_bias))
			layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
			layers.append(nn.ReLU(inplace=True))
			self.inplanes = planes

		return nn.Sequential(*layers)

	# flag 0: only pose, 
	# flag 1: only embedding 
	# flag 2: pose and embedding
	def forward(self, x, adj, flag=0):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		if flag == 0:
			x = self.deconv_layers(x)
			x = self.final_layer(x)
			return x
		elif flag ==1:
			self.indices = self.indices.type(torch.cuda.LongTensor)
			hp = self.deconv_layers(x)
			hp_o = self.final_layer(hp)
			hp = F.upsample(hp_o, size=[12,9], mode='bilinear')
			hp = torch.index_select(hp,1,self.indices)
			x = self.embedding_layer(x)
			feature = torch.mul(hp.unsqueeze(dim=2), x.unsqueeze(dim=1))

			# Generate the mask and align the  feature
			align_feature = torch.zeros(hp.size(0), hp.size(1),9*256)
			for b in range(len(hp)):
				for i in range(len(hp[0])):
					mask = torch.zeros(hp.size(2),hp.size(3))
					temp = hp[b][i]
					y_values, ys = temp.max(dim=0) 
					x_values, xc = y_values.max(dim=0)  # xc = x cordinate
					y = ys[xc]
					mask[y][xc] = 1
					for n in range(0,2):
						y_l = y + n
						y_s = y - n
						xc_l = xc + n
						xc_s = xc - n
						if y_l >= hp.size(2):
							y_l = hp.size(2)-1
						if y_s < 0:
							y_s = 0
						if xc_l >= hp.size(3):
							xc_l = hp.size(3)-1
						if xc_s < 0:
							xc_s = 0
						
						mask[y_l][xc_l] = 1
						mask[y_s][xc_s] = 1
						mask[y][xc_l] = 1
						mask[y][xc_s] = 1
						mask[y_l][xc] = 1
						mask[y_s][xc] = 1
						mask[y_s][xc_l] = 1
						mask[y_l][xc_s] = 1
					mask = mask.type(torch.cuda.ByteTensor)
					temp_feature = torch.masked_select(feature[b][i], mask)
					cnt = mask.sum()
					if cnt < 9:
						mean = torch.mean(temp_feature).repeat(256)
						while cnt < 9:
							temp_feature = torch.cat((temp_feature, mean), dim=0)
							cnt = cnt + 1
					align_feature[b][i] = temp_feature
			align_feature = align_feature.type(torch.cuda.FloatTensor)
			x_part = F.dropout(F.relu(self.graph_layer1(align_feature,adj)))
			x_part = self.graph_layer2(x_part,adj)
			x_part = self.fc_feature_align(x_part.view(x_part.size(0), -1))

			x = x.view(x.size(0), -1)
			x = self.fc_feature(x)

			x = 0.1 * x + 0.9 * x_part
			if not self.feature_flag:
				x = self.classification(x)
			return x
		else:
			x1 = self.deconv_layers(x)
			x1 = self.final_layer(x1)
			
			x_pose = self.pose_conv(x1)
			x2 = self.embedding_layer(x)
			x2 = x2 * x_pose
			x2 = x2.view(x2.size(0),-1)
			x2 = self.fc_feature(x2)
			if not self.feature_flag:
				x2 = self.classification(x2)
			return x1, x2

	def init_weights(self, pretrained=''):
		if os.path.isfile(pretrained):
			logger.info('=> init deconv weights from normal distribution')
			for name, m in self.deconv_layers.named_modules():
				if isinstance(m, nn.ConvTranspose2d):
					logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
					logger.info('=> init {}.bias as 0'.format(name))
					nn.init.normal_(m.weight, std=0.001)
					if self.deconv_with_bias:
						nn.init.constant_(m.bias, 0)
				elif isinstance(m, nn.BatchNorm2d):
					logger.info('=> init {}.weight as 1'.format(name))
					logger.info('=> init {}.bias as 0'.format(name))
					nn.init.constant_(m.weight, 1)
					nn.init.constant_(m.bias, 0)
			
			logger.info('=> init final layer weights from normal distribution')
			for m in self.final_layer.modules():
				if isinstance(m, nn.Conv2d):
					# nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
					logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
					logger.info('=> init {}.bias as 0'.format(name))
					nn.init.normal_(m.weight, std=0.001)
					nn.init.constant_(m.bias, 0)
					
			logger.info('=> init embedding weights from normal distribution')
			for name, m in self.embedding_layer.named_modules():
				if isinstance(m, nn.Conv2d):
					# nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
					logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
					logger.info('=> init {}.bias as 0'.format(name))
					nn.init.normal_(m.weight, std=0.001)
					#nn.init.constant_(m.bias, 0)
						
				elif isinstance(m, nn.BatchNorm2d):
					logger.info('=> init {}.weight as 1'.format(name))
					logger.info('=> init {}.bias as 0'.format(name))
					nn.init.constant_(m.weight, 1)
					nn.init.constant_(m.bias, 0)
					
			logger.info('=> init pose_conv weights from normal distribution')		
			for name, m in self.pose_conv.named_modules():
				if isinstance(m, nn.Conv2d):
					# nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
					logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
					logger.info('=> init {}.bias as 0'.format(name))
					nn.init.normal_(m.weight, std=0.001)
					#nn.init.constant_(m.bias, 0)
						
				elif isinstance(m, nn.BatchNorm2d):
					logger.info('=> init {}.weight as 1'.format(name))
					logger.info('=> init {}.bias as 0'.format(name))
					nn.init.constant_(m.weight, 1)
					nn.init.constant_(m.bias, 0)
			
			pretrained_state_dict = torch.load(pretrained)
			logger.info('=> loading pretrained model {}'.format(pretrained))
			self.load_state_dict(pretrained_state_dict, strict=False)
		else:
			logger.error('=> imagenet pretrained model dose not exist')
			logger.error('=> please download it first')
			raise ValueError('imagenet pretrained model does not exist')


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_pose_net(cfg, is_train, flag=1, **kwargs):
    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS

    block_class, layers = resnet_spec[num_layers]

    model = PoseResNet(block_class, layers, cfg, flag, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model
