import json
import os
import argparse
from tqdm import tqdm
import cv2
import numpy as np
from cv2_color import Color
import glob
from config import Config

# The match list from the results to the test
match_list=[13,12,14,9,8,10,7,11,6,3,2,4,1,5,0]
color = Color(flag='bgr')

def draw_limb(image, kps, color):
    def draw_line(head, tail):
        if head == [] or tail == []:
            return
        cv2.line(image, head, tail, color, 3)
    limbs = [
        (0, 1),
        (1, 2),
        (3, 4),
        (4, 5),
        (6, 7),
        (7, 8),
        (8, 12),
        (9, 10),
        (10, 11),
        (9, 12),
        (12, 13),
        (13, 14)
    ]
    for h, t in limbs:
        draw_line(kps[h], kps[t])


def demo(image_dir, result_dir, save_dir):
    """
    image_dir: the location where the frames are stored
    result_dir: the results in 2017PT fromat, each video has one file 
    save_dir: the loaction where we store the result videos
    """
    
    #json_files = os.listdir(result_dir)
    json_files = glob.glob(result_dir + "/*.json")
    json_files = [json_file.split('/')[-1] for json_file in json_files]
    
    pbar = tqdm(range(len(json_files)))
    for json_name in json_files:
       
        video_name = json_name.replace('.json','-pgpt')
        
        video_folder = os.path.join(save_dir, video_name)
        
        if not os.path.exists(video_folder):
                os.mkdir(video_folder)
       
        with open(os.path.join(result_dir,json_name),'r') as f:
            old_annolist = json.load(f)['annolist']
        pbar.set_description('Visulizing video {}'.format(video_name))
        color_list = color.get_random_color_list()
        j = 0
        for i,annotation in enumerate(old_annolist):
            color_flag = 0
            frame_name = annotation['image'][0]['name']
            #annotation['image_id'] #10010010103=frame_id
            frame_store_path = os.path.join(video_folder, frame_name.split('/')[-1]) #file
            frame_path = os.path.join(image_dir,frame_name) #file dor read
            frame = cv2.imread(frame_path)
            im_H, im_W, im_C = frame.shape
            if i==0:
                fourcc = cv2.cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                videoWriter = cv2.VideoWriter(save_dir + '{}.mp4'.format(video_name),fourcc,10,(im_W,im_H))
            old_annorect = annotation['annorect']
            for anno in old_annorect:
                if len(anno['annopoints']) == 0:
                    continue
                old_point_list = anno['annopoints'][0]['point']
                xmin, xmax, ymin, ymax, track_id = anno['x1'][0], anno['x2'][0], anno['y1'][0], anno['y2'][0], anno['track_id'][0]
                color_flag = int(track_id) % 16
                kps = [[] for _ in range(15)]
                for pose in old_point_list:
                    pose_id, pose_x, pose_y, = pose['id'][0], pose['x'][0], pose['y'][0] 
                    kps[pose_id] = (int(pose_x), int(pose_y))
                    cv2.circle(frame,(int(pose_x),int(pose_y)), 3 ,color_list[color_flag], -1)
                draw_limb(frame, kps, color_list[color_flag])
                cv2.rectangle(frame, (int(xmin),int(ymin)), (int(xmax),int(ymax)), color_list[color_flag], 3)            
                cv2.putText(frame, 'id:' + str(track_id), (int(xmin),int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 1, color_list[color_flag], 2)
            videoWriter.write(frame)
            cv2.imwrite(frame_store_path, frame)
        #videoWriter.release()
        pbar.update(1)
    pbar.close()

    
if __name__ == '__main__':
    print('Visualizing the results')
    config = Config()
    parser = argparse.ArgumentParser(description="Visualizing the results")
    parser.add_argument("--image_dir", type=str, default=config.root)
    parser.add_argument("--result_dir", type=str, default=config.save_dir)
    parser.add_argument("--save_dir", type=str, default=config.video_path)
	#parser.add_argument('--cfg', type=str, required=True) #added by alnguyen
	#parser.add_argument('opts', 
                        #help='Modify config options using the command-line',
                        #default=None,
                        #nargs=argparse.REMAINDER) #added by alnguyen
						
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    demo(args.image_dir, args.result_dir, args.save_dir)


