import glob
import os
import pandas as pd
import argparse
import numpy as np
import cv2
import time
import json

from fire import Fire
from tqdm import tqdm
import random

from pose_estimation_graph import PoseNet
from tracker import SiamFCTracker

from match import Matcher
from model.nms.nms_wrapper import nms

class Track_And_Detect(object):
	effective_track_thresh = 0.5
	effective_detection_thresh = 0.5
	effective_keypoints_thresh = 0.6
	effective_keypoints_number = 8
	iou_match_thresh = 0.5
	nms_thresh = 0.5
	oks_thresh = 0.8
	embedding_match_thresh = 2
	feature_length = 2048
	
	oks_flag = True
	tracker_flag = True
	tracker_update_flag = True
	new_embedding_flag = True
	descrease_tracker_flag = True
	#descrease_tracker_flag = False
	
	def __init__(self, gpu_id=0, track_model=None, pose_model=None, embedding_model=None):
		if self.tracker_flag:
			self.tracker = SiamFCTracker(gpu_id, track_model)
		self.posenet = PoseNet(gpu_id, pose_model)
		
		self.matcher = Matcher()
		print('----------------------------------------')
		print('Flag parameters are set as follow:')
		print('Tracker flag: {}'.format(self.tracker_flag))
		print('Tracker update flag: {}'.format(self.tracker_update_flag))
		print('Decrease tracker flag: {}'.format(self.descrease_tracker_flag))
		print('New embedding(with pose) flag: {}'.format(self.new_embedding_flag))
		print('----------------------------------------')
	
	# bbox must be format of x1y1x2y2
	def update_tracker(self, rgb_frame, bbox, track_id):
		self.tracker.update_data_dict(rgb_frame, bbox, track_id)
		
	def pose_detect(self, im, bbox):
		return self.posenet.detect_pose(im, bbox)
		
	def embedding(self, frame, bbox):
		feature = self.posenet.embedding(frame,bbox)
		return feature


	# initialize the first frame of this video
	def init_tracker(self, frame, bbox_list):
		self.new_id_flag=0
		self.track_id_dict= dict()
		if self.tracker_flag:
			self.tracker.clear_data()
		# conver bgr(opencv) to rgb
		rgb_frame = frame[:,:,::-1]
		bbox_list, keypoint_list = self.oks_filter(bbox_list, frame)
		for bbox in bbox_list:
			self.create_id(frame, rgb_frame, bbox)
		bbox_list=[]
		for id,item in self.track_id_dict.items():
			bbox = item['bbox_and_score'] + [id]
			bbox_list.append(bbox)
		return bbox_list
		
	def oks_filter(self, det_list, frame):
		keypoint_list = []
		for bbox in det_list:
			center, scale = self.posenet.x1y1x2y2_to_cs(bbox[0:4])
			area = np.prod(scale*200,1)
			pred = np.zeros((15,3), dtype=np.float32)
			pose_positions, pose_vals, pose_heatmaps = self.pose_detect(frame, bbox)
			pred[:,0:2] = pose_positions
			pred[:,2] = pose_vals
			score_all,valid_num = 0, 0
			for i in range(15):
				score_i = pose_vals[i]
				if score_i >= 0.2:
					score_all += score_i
					valid_num += 1
			if valid_num!=0:
				new_score = score_all/valid_num *bbox[4]
			else:
				new_score = 0
			keypoint_dict={'score':new_score, 'area':area, 'keypoints':pred}
			keypoint_list.append(keypoint_dict)
		keep = self.matcher.oks_nms(keypoint_list, thresh= self.oks_thresh)
		new_det_list = [det_list[i] for i in keep]
		new_keypoint_list = [keypoint_list[i] for i in keep]
		return new_det_list, new_keypoint_list
		
	def create_id(self, frame, rgb_frame, bbox):
		score = bbox[4]
		bbox = bbox[0:4]
		track_id = self.new_id_flag
		if self.new_embedding_flag:
			feature = self.embedding(frame, bbox)
		else:
			feature = self.embedder.embedding(frame, bbox)
		self.track_id_dict[track_id]={'bbox_and_score':bbox+[score],'feature':feature, 'frame_flag':1, 'exist':True}
		if self.tracker_flag:
			self.update_tracker(rgb_frame, bbox, track_id)
		self.new_id_flag += 1
		
	def update_id(self, frame, rgb_frame, det_bbox, track_id):
		bbox, score = det_bbox[0:4], det_bbox[4]
		if self.new_embedding_flag:
			feature = np.array(self.embedding(frame, bbox))
		else:
			feature = np.array(self.embedder.embedding(frame, bbox))
		
		former_track_dict = self.track_id_dict[track_id]
		former_frame_flag, former_feature = former_track_dict['frame_flag'], np.array(former_track_dict['feature'])
		now_frame_flag = former_frame_flag+1
		now_feature = feature.tolist()
		self.track_id_dict[track_id]={'bbox_and_score':det_bbox,'feature':now_feature, 'frame_flag':now_frame_flag, 'exist':True}
		if self.tracker_flag and self.tracker_update_flag:
			self.update_tracker(rgb_frame, bbox, track_id)
		
	def multi_track(self, frame):
		rgb_frame = frame[:,:,::-1]
		bbox_list = []
		for id in self.track_id_dict:
			if self.track_id_dict[id]['exist'] == False:
				continue
			bbox, score = self.tracker.track_id(rgb_frame, id)
			bbox_list.append(bbox +[score] +[id])
			self.track_id_dict[id]['bbox'] = bbox
		return bbox_list

	def match_detection_tracking_oks_iou_embedding(self, detections, track_list, frame):
		rgb_frame = frame[:,:,::-1]
			
		#print(detections)
		for track in track_list:
			track_score = track[4]
			if track_score >= self.effective_track_thresh:
				if self.descrease_tracker_flag:
					track_score -= 0.35
				detections.append(track[0:4]+[track_score])
		#print(detections)
		if self.oks_flag:
			detections, keypoint_list = self.oks_filter(detections, frame)
		#decrease the tracking score
			
		#get feature for former trackers
		database_bbox_list = []
		for database_id in self.track_id_dict:
			database_id_bbox = self.track_id_dict[database_id]['bbox_and_score']+[database_id]
			database_bbox_list.append(database_id_bbox)
		
		matches, unmatched_detections, unmatched_trackers = self.matcher.associate_detections_to_trackers_iou(detections, database_bbox_list, iou_threshold = self.iou_match_thresh)
		
		#update the matched trackers with detection bbox
		for match in matches:
			det_index, track_index = match
			det_bbox = detections[det_index]
			update_id = database_bbox_list[track_index][5]
			self.update_id(frame, rgb_frame, det_bbox, update_id)
			
		#create new index for unmatched_detections
		det_feature_list =[]
		for new_index in unmatched_detections:
			det_bbox = detections[new_index]
			det_score = det_bbox[4]
			if self.new_embedding_flag:
				det_id_feature = self.embedding(frame, det_bbox) + det_bbox
			else:
				det_id_feature = self.embedder.embedding(frame, det_bbox) + det_bbox
			det_feature_list.append(det_id_feature)
			
		track_feature_list = []
		for delete_index in unmatched_trackers:
			track_bbox = database_bbox_list[delete_index]
			track_score, delete_id = track_bbox[4], track_bbox[5]
			delete_id_feature = self.track_id_dict[delete_id]['feature'] + [delete_id]
			track_feature_list.append(delete_id_feature)
			
		embedding_matches, \
		embedding_unmatched_detections,\
		embedding_unmatched_trackers = self.matcher.associate_detections_to_trackers_embedding(det_feature_list, 
																								track_feature_list,
																								distance_threshold = self.embedding_match_thresh)
		
		#update matched embedding detections and former tracking feature
		for match in embedding_matches:
			det_index, track_index = match
			det_bbox = det_feature_list[det_index][2048:]
			update_id = track_feature_list[track_index][2048]
			self.update_id(frame, rgb_frame, det_bbox, update_id) 
			
		#create new id for unmatched detections
		for new_index in embedding_unmatched_detections:
			det_bbox = det_feature_list[new_index][2048:]
			det_score = det_bbox[4]
			pose_position, pose_value, pose_heatmap = self.pose_detect(frame, det_bbox)
			if det_score >= self.effective_detection_thresh and np.sum(pose_value >= self.effective_keypoints_thresh) >= self.effective_keypoints_number:
				self.create_id(frame, rgb_frame, det_bbox)
			
		#delete unuseful index for unmatched_trackers
		for delete_index in embedding_unmatched_trackers:
			delete_id = track_feature_list[delete_index][2048]
			del self.track_id_dict[delete_id]
			if self.tracker_flag:
				self.tracker.delete_id(delete_id)
			
		bbox_list = []
		for id,item in self.track_id_dict.items():
			if item['exist']==True:
				bbox_list.append(item['bbox_and_score']+[id])
		return bbox_list
		


	
