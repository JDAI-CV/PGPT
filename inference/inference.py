import glob
import os
import pandas as pd
import argparse
import numpy as np
import cv2
import time
import json

from tqdm import tqdm
import random

#from config import Config
import sys
sys.path.append('../lib')
from config import cfg
from config import update_config

from track_and_detect_new import Track_And_Detect

'''
For posetrack dataset, the output keypoints is as follow 
"keypoints": {
	0: "nose",
	1: "head_bottom",
	2: "head_top",
	3: "left_shoulder",
	4: "right_shoulder",
	5: "left_elbow",
	6: "right_elbow",
	7: "left_wrist",
	8: "right_wrist",
	9: "left_hip",
	10: "right_hip",
	11: "left_knee",
	12: "right_knee",
	13: "left_ankle",
	14: "right_ankle"
}
For competition
"keypoints": {
  0: "right_ankle",
  1: "right_knee",
  2: "right_hip",
  3: "left_hip",
  4: "left_knee",
  5: "left_ankle",
  6: "right_wrist",
  7: "right_elbow",
  8: "right_shoulder",
  9: "left_shoulder",
  10: "left_elbow",
  11: "left_wrist",
  12: "neck",
  13: "nose",
  14: "head_top",
}
'''
match_list=[13,12,14,9,8,10,7,11,6,3,2,4,1,5,0]
#config = Config()
def parseArgs():
	parser = argparse.ArgumentParser(description="Evaluation of Pose Estimation and Tracking (PoseTrack)")
	parser.add_argument('--cfg', type=str, required=True) #added by alnguyen
	parser.add_argument("-t", "--detection_thresh",dest = 'det_thresh',required=False, default=0.4, type= float)
	parser.add_argument("-p", "--pos_thresh",dest = 'pose_thresh',required=False, default=0, type= float)
	parser.add_argument("-v", "--vis_flag",dest = 'vis_flag',required=False, default=False, type= bool)
	parser.add_argument('opts', 
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER) #added by alnguyen
						
	args = parser.parse_args()

	return args

class DateEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj,np.float32):
			return float(obj)
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return json.JSONEncoder.default(self, obj)

def track_test():

	args = parseArgs()
	pose_vis_thresh = args.pose_thresh
	detection_score_thresh = args.det_thresh
	vis_flag = args.vis_flag

	update_config(cfg, args)
	gpu_id = cfg.GPU_ID
	json_path = cfg.INPUT.JSON_DETECTION_PATH
	# Change temporially 
	save_dir = cfg.OUTPUT.SAVE_DIR

	gt_json_path = cfg.INPUT.GT_JSON_PATH
	data_folder = cfg.INPUT.DATA_FOLDER
	video_path = cfg.OUTPUT.VIDEO_PATH

	print('----------------------------------------------------------------------------------')
	print('Detection_score_thresh: {}    Vis_flag: {}'.format(detection_score_thresh, vis_flag))
	print('Detection results is set as {}'.format(json_path))
	print('Results will be saved to {}'.format(save_dir))
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	# Load the Detection Results (demo_detection.json)
	with open(json_path,'r') as f:
		bbox_dict = json.load(f)
	
	# Create the Tracker
	track_model=cfg.INPUT.TRACK_MODEL
	pose_model=cfg.INPUT.POSE_MODEL
	embedding_model=cfg.INPUT.EMBEDDING_MODEL
	tracker = Track_And_Detect(gpu_id=gpu_id, track_model=track_model, pose_model=pose_model, embedding_model=embedding_model)
	
	# Load the Ground Truth to get the right video keys (demo_val.json)
	with open(gt_json_path,'r') as f:
		gt_dict = json.load(f)
	

	video_keys = gt_dict.keys()
	pbar = tqdm(range(len(video_keys)))
	for video_name in video_keys: #in demo_val.json
		pbar.update(1)
		frame_dict = bbox_dict[video_name]
		#video_name = video_name.replace('.json','')
		video_json = {'annolist':[]}
		save_path = os.path.join(save_dir, video_name+'.json')
		idx =0
		for frame_name in sorted(frame_dict.keys()):
			start = time.time()
			frame_path = os.path.join(data_folder,frame_name)
			frame = cv2.imread(frame_path)
			bbox_list = frame_dict[frame_name]
			det_list = []
			for bbox in bbox_list:
				if bbox[4] >= detection_score_thresh:
					det_list.append(bbox)
			if idx == 0:
				im_H,im_W,im_C = frame.shape
				if vis_flag:
					fourcc = cv2.cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
					if not os.path.exists(video_path):
						os.makedirs(video_path)
					video_store_name = video_path + '/{}.mp4'
					videoWriter = cv2.VideoWriter(video_store_name.format(video_name+'-pgpt'),fourcc,10,(im_W,im_H))
				final_list = tracker.init_tracker(frame,det_list)
			else:	
				track_list = tracker.multi_track(frame)
				final_list = tracker.match_detection_tracking_oks_iou_embedding(det_list, track_list, frame)
			
			image_dict = dict()
			annorect = []
			for det in final_list:
				point_list = []
				pose_position, pose_value, pose_heatmap = tracker.pose_detect(frame, det)
				for i, pose in enumerate(pose_position):
					score_i = pose_value[i]
					pose_id = match_list[i]
					point_list.append({'id':[pose_id],'x':[pose[0]],'y':[pose[1]],'score':[score_i]})
				point_dict = {'point':point_list}
				xmin,ymin,xmax,ymax,score,track_id = det
				annorect.append({'x1':[xmin],'x2':[xmax],'y1':[ymin],'y2':[ymax],'score':[score],'track_id':[track_id],'annopoints':[point_dict]})
			image_dict['image'] = [{'name':frame_name}]
			image_dict['annorect'] = annorect
			video_json['annolist'].append(image_dict)
			idx += 1
			pbar.set_description('Processing video {}: process {} takes {:.3f} seconds'.format(video_name, frame_name, time.time()-start))
			
		with open(save_path,'w') as f:
			json.dump(video_json, f, cls=DateEncoder)
	pbar.close()


if __name__ == "__main__":
	track_test()
