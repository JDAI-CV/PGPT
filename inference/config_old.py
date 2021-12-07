
class Config():
    root = '/PGPT'
    
    # save_dir is the loaction where we store the results
    save_dir = root + '/results/demo'
    
    # json_path_detection is the loaction where we store the detection results
    json_path_detection = root + '/results/demo_detection.json'
    
    # gt_json_path is the ground truth of the validiation, all the ground_truth are in one file
    gt_json_path = root + '/data/demo_val.json'
    
    # the data folder of the PoseTrack dataset
    data_folder = root + '/data/demodataset'
    
    # the path of the location where we store the video
    video_path = save_dir
    
    # the path of the track model
    track_model = root + '/models/tracker.pth'
    
    # the path of the pose estimation model
    pose_model = root + '/models/pose_gcn.pth.tar'
    
    # pose config file location
    pose_cfg = root + '/cfgs/pose_res152.yaml'
    
    # the path of the embedding model
    embedding_model = root + '/models/embedding_model.pth'
    
    def __init__(self):
        print('Using the config class at', __file__)
