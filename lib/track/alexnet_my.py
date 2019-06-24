import torch
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import models
from torch.autograd import Variable
from torch import nn


from .pose_simple import CPM
import cv2 
from PIL import Image


class SiameseAlexNet(nn.Module):
	def __init__(self, gpu_id, train=True):
		print('Initilizeing alexnet_my~~')
		super(SiameseAlexNet, self).__init__()

		self.PoseNet = CPM(1, initialize=train)

		for p in self.PoseNet.parameters():
			p.requires_grad = False

		self.exemplar_size = (127, 127)
		self.instance_size = (24, 24)

		self.features = nn.Sequential(
			nn.Conv2d(4, 96, 11, 2),
			nn.BatchNorm2d(96),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(3, 2),
			nn.Conv2d(96, 256, 5, 1, groups=2),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(3, 2),
			nn.Conv2d(256, 384, 3, 1),
			nn.BatchNorm2d(384),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 384, 3, 1, groups=2),
			nn.BatchNorm2d(384),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 256, 3, 1, groups=2))						   
		
		self.corr_bias = nn.Parameter(torch.zeros(1))

		if train:
			gt, weight = self._create_gt_mask((config.train_response_sz, config.train_response_sz),mode='train')
			with torch.cuda.device(gpu_id):
				self.train_gt = torch.from_numpy(gt).cuda()
				self.train_weight = torch.from_numpy(weight).cuda()
			gt, weight = self._create_gt_mask((config.response_sz, config.response_sz), mode='valid')
			with torch.cuda.device(gpu_id):
				self.valid_gt = torch.from_numpy(gt).cuda()
				self.valid_weight = torch.from_numpy(weight).cuda()
		self.exemplar = None

	"""
	Initial the models 
	"""
			
	def set_bn_fix(self):
		def set_bn_eval(m):
			classname = m.__class__.__name__
			if classname.find('BatchNorm') != -1:
			  m.eval()
		self.PoseNet.apply(set_bn_eval)
		
	def forward(self, x = (None, None), y = (None,None), feature=None):
		exemplar, instance = x # Tracking input
		exemplar_pose, instance_pose = y # Pose input
		if instance is not None:
			N, C, H, W = instance.shape
		if feature is None:
			if exemplar is not None and instance is not None:
				exemplar_limb, exemplar_joint = self.PoseNet(exemplar_pose)
				instance_limb, instance_joint = self.PoseNet(instance_pose)
				
				example_heatmap = exemplar_joint[:,18:,:,:]
				instance_heatmap = instance_joint[:,18:,:,:]
				
				exemplar_pose_feature = F.upsample(example_heatmap, size= (127, 127), mode='bilinear')
				instance_pose_feature = F.upsample(instance_heatmap, size= (W, W), mode='bilinear')
				
				exemplar = torch.cat((exemplar, exemplar_pose_feature), 1)
				instance = torch.cat((instance, instance_pose_feature), 1)
				
				exemplar = self.features(exemplar)
				instance = self.features(instance)
			   
				score_map = []
				N, C, H, W = instance.shape

				if N > 1:
					for i in range(N):
						score = F.conv2d(instance[i:i+1], exemplar[i:i+1]) * config.response_scale + self.corr_bias
						score_map.append(score)
					return torch.cat(score_map, dim=0)
				else:
					return F.conv2d(instance, exemplar) * config.response_scale + self.bias
			elif exemplar is not None and instance is None:
				exemplar_limb, exemplar_joint = self.PoseNet(exemplar_pose)
				example_heatmap = exemplar_joint[:,18:,:,:]
				exemplar_pose_feature = F.upsample(example_heatmap, size= (127, 127), mode='bilinear')
				exemplar = torch.cat((exemplar, exemplar_pose_feature), 1)
				exemplar = self.features(exemplar)

				return exemplar
			else:
				instance_limb, instance_joint = self.PoseNet(instance_pose)
				instance_heatmap = instance_joint[:,18:,:,:]
				instance_pose_feature = F.upsample(instance_heatmap, size= (W, W), mode='bilinear')
				instance = torch.cat((instance, instance_pose_feature), 1)
				instance = self.features(instance)

				score_map = []
				for i in range(instance.shape[0]):
					score_map.append(F.conv2d(instance[i:i+1], self.exemplar))
				return torch.cat(score_map, dim=0)
		else:
			self.exemplar = feature
			instance_limb, instance_joint = self.PoseNet(instance_pose)
			instance_heatmap = instance_joint[:,18:,:,:]
			instance_pose_feature = F.upsample(instance_heatmap, size= (W, W), mode='bilinear')
			instance = torch.cat((instance, instance_pose_feature), 1)
			instance = self.features(instance)
			
			score_map = []
			for i in range(instance.shape[0]):
				score_map.append(F.conv2d(instance[i:i+1], self.exemplar))
			return torch.cat(score_map, dim=0)      

	def loss(self, pred):
		return F.binary_cross_entropy_with_logits(pred, self.gt)

	def weighted_loss(self, pred):
		if self.training:
			#print(pred.shape,self.train_gt.shape)
			return F.binary_cross_entropy_with_logits(pred, self.train_gt,
					self.train_weight, size_average = False) / config.train_batch_size # normalize the batch_size
		else:
			#print(pred.shape, self.valid_gt.shape, self.valid_weight.shape)
			return F.binary_cross_entropy_with_logits(pred, self.valid_gt,
					self.valid_weight, size_average = False) / config.valid_batch_size # normalize the batch_size

	def _create_gt_mask(self, shape, mode='train'):
		# same for all pairs
		h, w = shape
		y = np.arange(h, dtype=np.float32) - (h-1) / 2.
		x = np.arange(w, dtype=np.float32) - (w-1) / 2.
		y, x = np.meshgrid(y, x)
		dist = np.sqrt(x**2 + y**2)
		mask = np.zeros((h, w))
		mask[dist <= config.radius / config.total_stride] = 1
		mask = mask[np.newaxis, :, :]
		weights = np.ones_like(mask)
		weights[mask == 1] = 0.5 / np.sum(mask == 1)
		weights[mask == 0] = 0.5 / np.sum(mask == 0)
		if mode == 'train':
			mask = np.repeat(mask, config.train_batch_size, axis=0)[:, np.newaxis, :, :]
		elif mode == 'valid':
			mask = np.repeat(mask, config.valid_batch_size, axis=0)[:, np.newaxis, :, :]
		return mask.astype(np.float32), weights.astype(np.float32)
