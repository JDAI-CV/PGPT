import os
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment

class Matcher:
	def __init__(self):
		pass

	def iou(self, bb_test,bb_gt):
		"""
		Computes IOU between two bboxes in the form [x1,y1,x2,y2]
		"""
		xx1 = np.maximum(bb_test[0], bb_gt[0])
		yy1 = np.maximum(bb_test[1], bb_gt[1])
		xx2 = np.minimum(bb_test[2], bb_gt[2])
		yy2 = np.minimum(bb_test[3], bb_gt[3])
		w = np.maximum(0., xx2 - xx1)
		h = np.maximum(0., yy2 - yy1)
		wh = w * h
		o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
		+ (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
		return(o)
		
	def nms(self, dets, thresh):
		"""Pure Python NMS baseline."""
		x1 = dets[:, 0]
		y1 = dets[:, 1]
		x2 = dets[:, 2]
		y2 = dets[:, 3]
		scores = dets[:, 4]

		areas = (x2 - x1 + 1) * (y2 - y1 + 1)
		order = scores.argsort()[::-1]

		keep = []
		while order.size > 0:
			i = order[0]
			keep.append(i)
			xx1 = np.maximum(x1[i], x1[order[1:]])
			yy1 = np.maximum(y1[i], y1[order[1:]])
			xx2 = np.minimum(x2[i], x2[order[1:]])
			yy2 = np.minimum(y2[i], y2[order[1:]])

			w = np.maximum(0.0, xx2 - xx1 + 1)
			h = np.maximum(0.0, yy2 - yy1 + 1)
			inter = w * h
			ovr = inter / (areas[i] + areas[order[1:]] - inter)

			inds = np.where(ovr <= thresh)[0]
			order = order[inds + 1]

		return keep
		
	def oks_iou(self, g, d, a_g, a_d, sigmas=None, in_vis_thre=None):
		if not isinstance(sigmas, np.ndarray):
			#for coco sigmas
			#sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
			#for posetrack sigmas
			sigmas = np.array([.26, .25, .25, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
		vars = (sigmas * 2) ** 2
		xg = g[0::3]
		yg = g[1::3]
		vg = g[2::3]
		ious = np.zeros((d.shape[0]))
		for n_d in range(0, d.shape[0]):
			xd = d[n_d, 0::3]
			yd = d[n_d, 1::3]
			vd = d[n_d, 2::3]
			dx = xd - xg
			dy = yd - yg
			e = (dx ** 2 + dy ** 2) / vars / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2 
			if in_vis_thre is not None:
				ind = list(vg > in_vis_thre) and list(vd > in_vis_thre)
				e = e[ind]
			ious[n_d] = np.sum(np.exp(-e)) / e.shape[0] if e.shape[0] != 0 else 0.0 
		return ious

	def oks_nms(self, kpts_db, thresh, sigmas=None, in_vis_thre=0.2):
		"""
		greedily select boxes with high confidence and overlap with current maximum <= thresh
		rule out overlap >= thresh, overlap = oks
		:param kpts_db
		:param thresh: retain overlap < thresh
		:return: indexes to keep
		"""
		if len(kpts_db) == 0:
			return []

		scores = np.array([kpts_db[i]['score'] for i in range(len(kpts_db))])
		kpts = np.array([kpts_db[i]['keypoints'].flatten() for i in range(len(kpts_db))])
		areas = np.array([kpts_db[i]['area'] for i in range(len(kpts_db))])

		order = scores.argsort()[::-1]

		keep = []
		while order.size > 0:
			i = order[0]
			keep.append(i)

			oks_ovr = self.oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]], sigmas, in_vis_thre)

			inds = np.where(oks_ovr <= thresh)[0]
			order = order[inds + 1]

		return keep
		
	def distance(self, feature_test, feature_gt):
		"""
		Computes distance between two 2048-d feature 
		"""		
		feature_test = np.array(feature_test[0:2048])
		feature_gt = np.array(feature_gt[0:2048])
		inner = np.dot(feature_test,feature_gt)
		a = np.linalg.norm(feature_test)
		b = np.linalg.norm(feature_gt)
		dis = 1-inner/(a*b)
		return dis
	  
	def associate_detections_to_trackers_iou(self, detections, track_list, iou_threshold = 0.5):
		"""
		Assigns detections to tracked object (both represented as bounding boxes)
		Returns 3 lists of matches, unmatched_detections and unmatched_trackers
		"""
		if(len(track_list)==0):
			return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
		iou_matrix = np.zeros((len(detections),len(track_list)),dtype=np.float32)

		for d,det in enumerate(detections):
			for t,trk in enumerate(track_list):
			  iou_matrix[d,t] = self.iou(det,trk)
		matched_indices = linear_assignment(-iou_matrix)

		unmatched_detections = []
		for d,det in enumerate(detections):
			if(d not in matched_indices[:,0]):
			  unmatched_detections.append(d)
		unmatched_track_list = []
		for t,trk in enumerate(track_list):
			if(t not in matched_indices[:,1]):
			  unmatched_track_list.append(t)

		#filter out matched with low IOU
		matches = []
		for m in matched_indices:
			if(iou_matrix[m[0],m[1]]<iou_threshold):
			  unmatched_detections.append(m[0])
			  unmatched_track_list.append(m[1])
			else:
			  matches.append(m.reshape(1,2))
		if(len(matches)==0):
			matches = np.empty((0,2),dtype=int)
		else:
			matches = np.concatenate(matches,axis=0)

		return matches, np.array(unmatched_detections), np.array(unmatched_track_list)
	
	def associate_detections_to_trackers_embedding(self, detections, track_list, distance_threshold = 2):
		"""
		Assigns detections to tracked object (both represented as bounding boxes)
		Returns 3 lists of matches, unmatched_detections and unmatched_trackers
		"""
		if(len(track_list)==0):
			return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
		distance_matrix = np.zeros((len(detections),len(track_list)),dtype=np.float32)

		for d,det in enumerate(detections):
			for t,trk in enumerate(track_list):
			  distance_matrix[d,t] = self.distance(det,trk)
		matched_indices = linear_assignment(distance_matrix)

		unmatched_detections = []
		for d,det in enumerate(detections):
			if(d not in matched_indices[:,0]):
			  unmatched_detections.append(d)
		unmatched_track_list = []
		for t,trk in enumerate(track_list):
			if(t not in matched_indices[:,1]):
			  unmatched_track_list.append(t)

		#filter out matched with high distance
		matches = []
		for m in matched_indices:
			if(distance_matrix[m[0],m[1]] > distance_threshold):
			  unmatched_detections.append(m[0])
			  unmatched_track_list.append(m[1])
			else:
			  matches.append(m.reshape(1,2))
		if(len(matches)==0):
			matches = np.empty((0,2),dtype=int)
		else:
			matches = np.concatenate(matches,axis=0)

		return matches, np.array(unmatched_detections), np.array(unmatched_track_list)
