import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class CPM(nn.Module):
	def __init__(self, depth_dim, num_features_part=128, use_relu=False, dilation=1, initialize=True):
		super(CPM, self).__init__()
		#self.pretrained = os.environ['CPM_PRETRAINED']

		self.depth_dim = depth_dim
		self.use_relu = use_relu
		self.dilation = dilation

		self.conv1_1 = nn.Conv2d(3, out_channels=64, kernel_size=3, stride=1, padding=1)
		self.conv1_2 = nn.Conv2d(64, out_channels=64, kernel_size=3, stride=1, padding=1)
		self.pool1_stage1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

		self.conv2_1 = nn.Conv2d(64, out_channels=128, kernel_size=3, stride=1, padding=1)
		self.conv2_2 = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=1, padding=1)
		self.pool2_stage1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

		self.conv3_1 = nn.Conv2d(128, out_channels=256, kernel_size=3, stride=1, padding=1)
		self.conv3_2 = nn.Conv2d(256, out_channels=256, kernel_size=3, stride=1, padding=1)
		self.conv3_3 = nn.Conv2d(256, out_channels=256, kernel_size=3, stride=1, padding=1)
		self.conv3_4 = nn.Conv2d(256, out_channels=256, kernel_size=3, stride=1, padding=1)
		self.pool3_stage1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

		self.conv4_1 = nn.Conv2d(256, out_channels=512, kernel_size=3, stride=1, padding=1)
		self.conv4_2 = nn.Conv2d(512, out_channels=512, kernel_size=3, stride=1, padding=1)
		self.conv4_3_CPM = nn.Conv2d(512, out_channels=256, kernel_size=3, stride=1, padding=1)
		self.conv4_4_CPM = nn.Conv2d(256, out_channels=128, kernel_size=3, stride=1, padding=1)

		# Stage1
		# limbs
		self.conv5_1_CPM_L1 = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=1, padding=1)
		self.conv5_2_CPM_L1 = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=1, padding=1)
		self.conv5_3_CPM_L1 = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=1, padding=1)
		self.conv5_4_CPM_L1 = nn.Conv2d(128, out_channels=512, kernel_size=1, stride=1, padding=0)
		self.conv5_5_CPM_L1 = nn.Conv2d(512, out_channels=38, kernel_size=1, stride=1, padding=0)
		# joints
		self.conv5_1_CPM_L2 = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=1, padding=1)
		self.conv5_2_CPM_L2 = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=1, padding=1)
		self.conv5_3_CPM_L2 = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=1, padding=1)
		self.conv5_4_CPM_L2 = nn.Conv2d(128, out_channels=512, kernel_size=1, stride=1, padding=0)
		self.conv5_5_CPM_L2 = nn.Conv2d(512, out_channels=19, kernel_size=1, stride=1, padding=0)

		if self.dilation == 1:
			self.concat_stage3_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

		# Stage2
		# limbs
		self.Mconv1_stage2_L1 = nn.Conv2d(185, out_channels=128, kernel_size=7, stride=1, padding=3)
		self.Mconv2_stage2_L1 = nn.Conv2d(128, out_channels=128, kernel_size=7, stride=1, padding=3)
		self.Mconv3_stage2_L1 = nn.Conv2d(128, out_channels=128, kernel_size=7, stride=1, padding=3)
		self.Mconv4_stage2_L1 = nn.Conv2d(128, out_channels=128, kernel_size=7, stride=1, padding=3)
		self.Mconv5_stage2_L1 = nn.Conv2d(128, out_channels=128, kernel_size=7, stride=1, padding=3)
		self.Mconv6_stage2_L1 = nn.Conv2d(128, out_channels=128, kernel_size=1, stride=1, padding=0)
		self.Mconv7_stage2_L1 = nn.Conv2d(128, out_channels=38, kernel_size=1, stride=1, padding=0)
		# joints
		self.Mconv1_stage2_L2 = nn.Conv2d(185, out_channels=128, kernel_size=7, stride=1, padding=3)
		self.Mconv2_stage2_L2 = nn.Conv2d(128, out_channels=128, kernel_size=7, stride=1, padding=3)
		self.Mconv3_stage2_L2 = nn.Conv2d(128, out_channels=128, kernel_size=7, stride=1, padding=3)
		self.Mconv4_stage2_L2 = nn.Conv2d(128, out_channels=128, kernel_size=7, stride=1, padding=3)
		self.Mconv5_stage2_L2 = nn.Conv2d(128, out_channels=128, kernel_size=7, stride=1, padding=3)
		self.Mconv6_stage2_L2 = nn.Conv2d(128, out_channels=128, kernel_size=1, stride=1, padding=0)
		self.Mconv7_stage2_L2 = nn.Conv2d(128, out_channels=19, kernel_size=1, stride=1, padding=0)

		

		if initialize:
			self.init_pretrained()

	def forward(self, inputs):
		# data rescale
		output1 = 0.0039 * inputs

		output1 = F.relu(self.conv1_1(output1))
		output1 = F.relu(self.conv1_2(output1))
		output1 = self.pool1_stage1(output1)

		output1 = F.relu(self.conv2_1(output1))
		output1 = F.relu(self.conv2_2(output1))
		output1 = self.pool2_stage1(output1)

		output1 = F.relu(self.conv3_1(output1))
		output1 = F.relu(self.conv3_2(output1))
		output1 = F.relu(self.conv3_3(output1))
		output1 = F.relu(self.conv3_4(output1))
		output1 = self.pool3_stage1(output1)

		output1 = F.relu(self.conv4_1(output1))
		output1 = F.relu(self.conv4_2(output1))
		output1 = F.relu(self.conv4_3_CPM(output1))
		output1 = F.relu(self.conv4_4_CPM(output1))

		output2_1 = F.relu(self.conv5_1_CPM_L1(output1))
		output2_1 = F.relu(self.conv5_2_CPM_L1(output2_1))
		output2_1 = F.relu(self.conv5_3_CPM_L1(output2_1))
		output2_1 = F.relu(self.conv5_4_CPM_L1(output2_1))
		output2_1 = self.conv5_5_CPM_L1(output2_1)

		output2_2 = F.relu(self.conv5_1_CPM_L2(output1))
		output2_2 = F.relu(self.conv5_2_CPM_L2(output2_2))
		output2_2 = F.relu(self.conv5_3_CPM_L2(output2_2))
		output2_2 = F.relu(self.conv5_4_CPM_L2(output2_2))
		output2_2 = self.conv5_5_CPM_L2(output2_2)

		output2 = torch.cat([output2_1, output2_2, output1], dim=self.depth_dim)

		output3_1 = F.relu(self.Mconv1_stage2_L1(output2))
		output3_1 = F.relu(self.Mconv2_stage2_L1(output3_1))
		output3_1 = F.relu(self.Mconv3_stage2_L1(output3_1))
		output3_1 = F.relu(self.Mconv4_stage2_L1(output3_1))
		output3_1 = F.relu(self.Mconv5_stage2_L1(output3_1))
		output3_1 = F.relu(self.Mconv6_stage2_L1(output3_1))
		output3_1 = self.Mconv7_stage2_L1(output3_1)
		
		
			
		output3_2 = F.relu(self.Mconv1_stage2_L2(output2))
		output3_2 = F.relu(self.Mconv2_stage2_L2(output3_2))
		output3_2 = F.relu(self.Mconv3_stage2_L2(output3_2))
		output3_2 = F.relu(self.Mconv4_stage2_L2(output3_2))
		output3_2 = F.relu(self.Mconv5_stage2_L2(output3_2))
		output3_2 = F.relu(self.Mconv6_stage2_L2(output3_2))
		output3_2 = self.Mconv7_stage2_L2(output3_2)

		
		#print('In CPM The \033[1;35m output3_1 is {}, the output3_2 is {}, the output1 is {}\033[0m'.format(output3_1.shape, output3_2.shape, output1.shape))
		return output3_1, output3_2

	def init_pretrained(self):
		location = '..../cpm.pth'
		state_dict = torch.load(location)
		from collections import OrderedDict
		new_model_dict = OrderedDict()
		model_state_dict = self.state_dict()
		for k,v in state_dict.items():
			if k in model_state_dict:
				new_model_dict[k] = v
		model_state_dict.update(new_model_dict)

		self.load_state_dict(model_state_dict, strict=True)
		print('CPM pretrained model loaded!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
		print('-----------------------------------------------------------')
		
