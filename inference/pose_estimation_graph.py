# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Modifed based on the original code of Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------


import logging
import time
import os
import pprint
import cv2

import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import math

import _init_paths
from pose.core.config import config as cfg
from pose.core.config import update_config
from pose.core.config import update_dir
from pose.core.config import get_model_name
from pose.models.pose_gcn import get_pose_net 

from pose.core.inference import get_max_preds
from pose.utils.transforms import flip_back
from pose.utils.transforms import get_affine_transform
from pose.utils.transforms import transform_preds
from pose.utils.vis import save_debug_images
from pose.utils.utils import create_logger

'''
For posetrack dataset, the keypoints is as follow
"keypoints": {
	0: "nose",
	1: "head_bottom",
	2: "head_top",
	3: "None",
	4: "None",
	5: "left_shoulder",
	6: "right_shoulder",
	7: "left_elbow",
	8: "right_elbow",
	9: "left_wrist",
	10: "right_wrist",
	11: "left_hip",
	12: "right_hip",
	13: "left_knee",
	14: "right_knee",
	15: "left_ankle",
	16: "right_ankle"
}
'''
class PoseNet(object):
	def __init__(self, gpu_id=0, model_path=None):
		self.cfg_file='/PGPT/cfgs/pose_res152.yaml'
		self.flag = 0

		update_config(self.cfg_file)
		print('----------------------------------------')
		print('PoseEstimation: Initilizing pose estimation network...')
		self.gpu_id = gpu_id
		self.pixel_std = 200
		self.image_size = cfg.MODEL.IMAGE_SIZE
		self.image_width = self.image_size[0]
		self.image_height = self.image_size[1]
		self.aspect_ratio = self.image_width * 1.0 /self.image_height
		self.transform = transforms.Compose([transforms.ToTensor(),
											 transforms.Normalize(mean = [0.485,0.456,0.406],
																  std = [0.229,0.224,0.225])
											])		
		self.flip_pairs = [[5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]

		cfg.TEST.FLIP_TEST = True
		cfg.TEST.POST_PROCESS = True
		cfg.TEST.SHIFT_HEATMAP = True
		cfg.TEST.MODEL_FILE = model_path

		torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
		torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

		with torch.cuda.device(self.gpu_id):
			self.model = get_pose_net(cfg, is_train=False, flag = self.flag)
			self._load_model()
			self.model.eval()
	
	def reset_adj_no1(self):
	## Existing points
		adj = torch.zeros([15,15])
		adj[0][1] = 1
		adj[1][0] = 1
		adj[0][2] = 1
		adj[2][0] = 1
		adj[1][2] = 1
		adj[2][1] = 1
		adj[3][4] = 1
		adj[4][3] = 1
		adj[3][5] = 1
		adj[5][3] = 1
		adj[5][7] = 1
		adj[7][5] = 1
		adj[4][6] = 1
		adj[6][4] = 1
		adj[6][8] = 1
		adj[3][9] = 1
		adj[9][3] = 1
		adj[4][10] = 1
		adj[10][4] = 1
		adj[9][10] = 1
		adj[10][9] = 1
		adj[9][11] = 1
		adj[11][13] = 1
		adj[10][12] = 1
		adj[12][10] = 1
		adj[12][14] = 1
		adj[14][12] = 1


		## Set the self-connection
		for i in range(adj.size()[0]):
			for j in range(adj.size()[0]):
				if i == j:
					# print('The i is', i)
					adj[i][j] = 1 - 0.1
				elif adj[i][j] != 1: 
					adj[i][j] = 0.2
		return adj

	
	def _load_model(self,model_file = None):		
		model_file = cfg.TEST.MODEL_FILE if model_file is None else model_file
		print("PoseEstimation: Loading checkpoint from %s" % (model_file))
		checkpoint=torch.load(model_file)
		from collections import OrderedDict
		model_dict = self.model.state_dict()
		new_state_dict = OrderedDict()
		for k,v in checkpoint.items():
			new_name = k[7:] if 'module' in k else k
			new_state_dict[new_name]=v
		model_dict.update(new_state_dict)
		self.model.load_state_dict(model_dict)
		self.model = self.model.cuda()
		print('PoseEstimation: PoseEstimation network has been initilized')	
	
	def detect_pose(self, im, bbox):
		im_in = im
		cords = bbox[0:4]
		center, scale = self.x1y1x2y2_to_cs(cords)
		r=0
		
		trans = get_affine_transform(center[0], scale[0], r, self.image_size)
		input_image = cv2.warpAffine(im_in, trans, (int(self.image_width), int(self.image_height)), flags=cv2.INTER_LINEAR)
		with torch.no_grad():
			input = self.transform(input_image)
			input = input.unsqueeze(0)
			with torch.cuda.device(self.gpu_id):
				adj = self.reset_adj_no1()
				adj = adj.unsqueeze(0)
				adj = adj.type(torch.cuda.FloatTensor)
				input = input.cuda()
				output = self.model(input, adj)
				if cfg.TEST.FLIP_TEST:
					input_flipped = np.flip(input.cpu().numpy(),3).copy()
					input_flipped = torch.from_numpy(input_flipped).cuda()
					output_flipped = self.model(input_flipped, adj)
					output_flipped = flip_back(output_flipped.cpu().numpy(), self.flip_pairs)
					output_flipped = torch.from_numpy(output_flipped.copy()).cuda()
					if cfg.TEST.SHIFT_HEATMAP:
						output_flipped[:, :, :, 1:] = output_flipped.clone()[:, :, :, 0:-1]
					output = (output + output_flipped) *0.5
		
		output = output.cpu().numpy()
		preds, maxvals = self.get_final_preds(output, center, scale)
		preds, maxvals = preds.squeeze(), maxvals.squeeze()
		heatmaps = output.squeeze()

		#For posetrack dataset, the 3th and 4th channel is None
		preds = np.delete(preds, [3,4], axis=0)
		maxvals = np.delete(maxvals, [3,4], axis=0)
		heatmaps = np.delete(heatmaps, [3,4], axis=0)
		return preds, maxvals, heatmaps
		
	def embedding(self, im, bbox):
		im_in = im
		cords = bbox[0:4]
		center, scale = self.x1y1x2y2_to_cs(cords)
		r=0	
		trans = get_affine_transform(center[0], scale[0], r, (288, 384))
		input_image = cv2.warpAffine(im_in, trans, (288, 384), flags=cv2.INTER_LINEAR)
		with torch.no_grad():
			adj = self.reset_adj_no1()
			adj = adj.unsqueeze(0)
			adj = adj.type(torch.cuda.FloatTensor)
			input = self.transform(input_image)
			input = input.unsqueeze(0)
			with torch.cuda.device(self.gpu_id):
				input = input.cuda()
				feature = self.model(input, adj,flag=1)
				if cfg.TEST.FLIP_TEST:
					input_flipped = np.flip(input.cpu().numpy(),3).copy()
					input_flipped = torch.from_numpy(input_flipped).cuda()
					feature_flipped = self.model(input_flipped, adj, flag=1)
					feature = (feature + feature_flipped) *0.5
		
		feature = feature.cpu().numpy().squeeze()
		return feature.tolist()
		
	def detect_pose_secway(self, im, bbox):
		cords = bbox[0:4]
		H,W,C = im.shape
		xmin,ymin,xmax,ymax = [int(i) for i in cords]
		xmax = min(xmax,W)
		ymax = min(ymax,H)
		xmin = max(xmin,0)
		ymin = max(ymin,0)
		w,h = xmax-xmin, ymax-ymin
		if w<0 or h<0:
			return np.zeros([17,2]),np.zeros([17]),np.zeros([17,64,64])
		im_in = im[ymin:ymax,xmin:xmax,:]
		scale_factor = [self.image_width/w, self.image_height/h]
		im_resize = cv2.resize(im_in, (int(self.image_width),int(self.image_height)), interpolation=cv2.INTER_LINEAR)
		with torch.no_grad():
			input = self.transform(im_resize)
			input = input.unsqueeze(0)
			
			with torch.cuda.device(self.gpu_id):
				adj = self.reset_adj_no1()
				adj = adj.unsqueeze(0)
				adj = adj.type(torch.cuda.FloatTensor)
				input = input.cuda()
				output = self.model(input, adj)
				if cfg.TEST.FLIP_TEST:
					input_flipped = np.flip(input.cpu().numpy(),3).copy()
					input_flipped = torch.from_numpy(input_flipped).cuda()
					output_flipped = self.model(input_flipped, adj)
					output_flipped = flip_back(output_flipped.cpu().numpy(), self.flip_pairs)
					output_flipped = torch.from_numpy(output_flipped.copy()).cuda()
					if cfg.TEST.SHIFT_HEATMAP:
						output_flipped[:, :, :, 1:] = output_flipped.clone()[:, :, :, 0:-1]
					output = (output + output_flipped) *0.5
		
			preds, maxvals = get_max_preds(output.clone().cpu().numpy())
			preds, maxvals = preds.squeeze()*4, maxvals.squeeze()
			heatmaps = output.cpu().numpy().squeeze()

			#For posetrack dataset, the 3th and 4th channel is None
			preds = np.delete(preds, [3,4], axis=0)
			preds[:,0] = preds[:,0]/scale_factor[0] + bbox[0]
			preds[:,1] = preds[:,1]/scale_factor[1] + bbox[1]
			maxvals = np.delete(maxvals, [3,4], axis=0)
			heatmaps = np.delete(heatmaps, [3,4], axis=0)
		return preds, maxvals, heatmaps
		
	def x1y1x2y2_to_cs(self, bbox):
		x,y,xmax,ymax = bbox
		w,h = xmax-x, ymax-y
		center = np.zeros((2), dtype=np.float32)
		center[0] = x + w * 0.5
		center[1] = y + h * 0.5

		if w > self.aspect_ratio * h:
			h = w * 1.0 / self.aspect_ratio
		elif w < self.aspect_ratio * h:
			w = h * self.aspect_ratio
		scale = np.array(
			[w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
			dtype=np.float32)
		if center[0] != -1:
			scale = scale * 1.25
		
		return np.expand_dims(center,0), np.expand_dims(scale,0)
	
	def get_final_preds(self, batch_heatmaps, center, scale):
		coords, maxvals = get_max_preds(batch_heatmaps)
		heatmap_height = batch_heatmaps.shape[2]
		heatmap_width = batch_heatmaps.shape[3]

		# post-processing
		if cfg.TEST.POST_PROCESS:
			for n in range(coords.shape[0]):
				for p in range(coords.shape[1]):
					hm = batch_heatmaps[n][p]
					px = int(math.floor(coords[n][p][0] + 0.5))
					py = int(math.floor(coords[n][p][1] + 0.5))
					if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
						diff = np.array([hm[py][px+1] - hm[py][px-1],
										 hm[py+1][px]-hm[py-1][px]])
						coords[n][p] += np.sign(diff) * .25 

		preds = coords.copy()
		for i in range(coords.shape[0]):
			preds[i] = transform_preds(coords[i], center[i], scale[i],
									   [heatmap_width, heatmap_height])
		return preds, maxvals


	
