import json
import os
import argparse
from tqdm import tqdm
from config import Config

config=Config()

def parseArgs():

    parser = argparse.ArgumentParser(description="Convert the pose results")
    parser.add_argument("-n", "--number",dest = 'pose_number',required=False, default=0, type= int)
    parser.add_argument("-t", "--thresh",dest = 'pose_thresh',required=False, default=0.5, type=float)
    parser.add_argument("-d", "--testdir",dest = 'test_dir',required=False, default=None, type=str)
    return parser.parse_args()

args = parseArgs()

if args.test_dir == None:
	test_filedir = config.test_dir
else:
	test_filedir = args.test_dir

json_dir = test_filedir
json_files = os.listdir(json_dir)
pose_vis_thresh = args.pose_thresh
pose_number_thresh = args.pose_number
print('--------------------------------------')
print('PoseVisThresh: {} PoseVisNumber: {}'.format(pose_vis_thresh, pose_number_thresh))


pbar = tqdm(range(len(json_files)))
# save_dir is where we store the convert pose results 
save_dir = '{}_{}_{}/'.format(test_filedir, pose_number_thresh, pose_vis_thresh)

if not os.path.exists(save_dir):
	os.mkdir(save_dir)
for json_name in json_files:
	video_json = {'annolist':[]}
	with open(os.path.join(json_dir,json_name),'r') as f:
		old_annolist = json.load(f)['annolist']
	save_path = os.path.join(save_dir, json_name)
	pbar.set_description('Processing video {}'.format(json_name))
	pbar.update(1)
	for annotation in old_annolist:
		#print(frame_name)
		image_dict = dict()
		old_annorect = annotation['annorect']
		new_annorect = []
		for anno in old_annorect:
			old_point_list = anno['annopoints'][0]['point']
			new_point_list = []
			flag = 0
			score_all = 0
			for i, pose in enumerate(old_point_list):
				score = pose['score'][0]
				score_all += score
				if score >= pose_vis_thresh:
					flag += 1
					new_point_list.append({'id':pose['id'],'x':pose['x'],'y':pose['y'],'score':pose['score']})
			det_score = anno['score'][0]
			
            #final_score = det_score + score_all/15.0
			final_score = det_score

			if flag>=pose_number_thresh:
				new_point_dict = {'point':new_point_list}
				new_annorect.append({'x1':anno['x1'],'x2':anno['x2'],'y1':anno['y1'],'y2':anno['y2'],'score':[final_score],'track_id':anno['track_id'],'annopoints':[new_point_dict]})
		image_dict['image'] = annotation['image']
		image_dict['annorect'] = new_annorect
		video_json['annolist'].append(image_dict)
	with open(save_path,'w') as f:
		json.dump(video_json, f)
pbar.close()
print('result has been saved to {}'.format(save_dir))
