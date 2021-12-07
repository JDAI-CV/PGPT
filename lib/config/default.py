import os
from yacs.config import CfgNode as CN


_C = CN()

_C.GPU_ID = 0 #ne marche pas pour les autres valeurs (1,2,3)
_C.SYSTEM = CN()
# Number of GPUS to use in the experiment
_C.SYSTEM.NUM_GPUS = 4
# Number of workers for doing things
_C.SYSTEM.NUM_WORKERS = 4

_C.SYSTEM.PGPT_ROOT = '/PGPT'
# pose config file location
_C.SYSTEM.POSE_CONFIG = '/PGPT/cfgs/pose_res152.yaml'

_C.INPUT = CN()
# json from detection and pose, images directory
_C.INPUT.JSON_DETECTION_PATH = '/PGPT/results/demo_detection.json'
# gt_json_path is the ground truth of the validiation, all the ground_truth are in one file
_C.INPUT.GT_JSON_PATH = '/PGPT/data/demo_val.json'
# the data folder of the PoseTrack dataset
_C.INPUT.DATA_FOLDER =  '/PGPT/data/demodataset'
# the path of the track model
_C.INPUT.TRACK_MODEL = '/PGPT/models/tracker.pth'
# the path of the pose estimation model
_C.INPUT.POSE_MODEL = '/PGPT/models/pose_gcn.pth.tar'
_C.INPUT.EMBEDDING_MODEL = '/PGPT/models/embedding_model.pth'

_C.OUTPUT = CN()
# where we store the results
_C.OUTPUT.SAVE_DIR = '/PGPT/results/demo'
# the path of the location where we store the video
_C.OUTPUT.VIDEO_PATH = '/PGPT/results/demo/demo-pgpt.mp4'

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
