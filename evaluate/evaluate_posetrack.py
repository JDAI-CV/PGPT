import json
import os
import sys
import numpy as np
import argparse

import _init_paths
from poseval.py.evaluateAP import evaluateAP
from poseval.py.evaluateTracking import evaluateTracking

from poseval.py import eval_helpers
from poseval.py.eval_helpers import Joint

def parseArgs():

    parser = argparse.ArgumentParser(description="Evaluation of Pose Estimation and Tracking (PoseTrack)")
    parser.add_argument("-g", "--groundTruth",required=False,default='/export/home/cyh/cvpr2019/SiamFC/data/posetrack/annotations_posetrack/val_2017/',
                        type=str,help="Directory containing ground truth annotatations per sequence in json format")
    parser.add_argument("-p", "--predictions",required=False,default='/export/home/zby/SiamFC/data/evaluate_pose_convert/',
                        type=str,help="Directory containing predictions per sequence in json format")
    parser.add_argument("-e", "--evalPoseEstimation",required=False,action="store_true",help="Evaluation of per-frame  multi-person pose estimation using AP metric")
    parser.add_argument("-t", "--evalPoseTracking",required=False,action="store_true",help="Evaluation of video-based  multi-person pose tracking using MOT metrics")
    parser.add_argument("-s","--saveEvalPerSequence",required=False,action="store_true",help="Save evaluation results per sequence",default=False)
    parser.add_argument("-o", "--outputDir",required=False,type=str,help="Output directory to save the results",default="./out0120")
    return parser.parse_args()


def main():

    args = parseArgs()
    print(args)
    argv = ['',args.groundTruth,args.predictions]

    print("Loading data")
    gtFramesAll,prFramesAll = eval_helpers.load_data_dir(argv)

    print("# gt frames  :", len(gtFramesAll))
    print("# pred frames:", len(prFramesAll))

    if (not os.path.exists(args.outputDir)):
        os.makedirs(args.outputDir)
        
    txt = open('result.txt','a')
    if (args.evalPoseEstimation):
        #####################################################
        # evaluate per-frame multi-person pose estimation (AP)

        # compute AP
        print("Evaluation of per-frame multi-person pose estimation")
        apAll,preAll,recAll = evaluateAP(gtFramesAll,prFramesAll,args.outputDir,True,args.saveEvalPerSequence)

        # print AP
        print("Average Precision (AP) metric:")
        eval_helpers.printTable(apAll) 
        print("Mean Precision (Pre) metric:")
        eval_helpers.printTable(preAll)
        print("Mean Recall (Rec) metric:")
        eval_helpers.printTable(recAll)
        total_AP, total_Pre, total_Rec = apAll[15][0] , preAll[15][0], recAll[15][0]
        txt.write('AP:{} Pre:{} Rec:{}\n'.format(total_AP, total_Pre, total_Rec))
    txt.close()

    if (args.evalPoseTracking):
        #####################################################
        # evaluate multi-person pose tracking in video (MOTA)
        
        # compute MOTA
        print("Evaluation of video-based  multi-person pose tracking")    
        metricsAll = evaluateTracking(gtFramesAll,prFramesAll,args.outputDir,True,args.saveEvalPerSequence)

        metrics = np.zeros([Joint().count + 4,1])
        for i in range(Joint().count+1):
            metrics[i,0] = metricsAll['mota'][0,i]
        metrics[Joint().count+1,0] = metricsAll['motp'][0,Joint().count]
        metrics[Joint().count+2,0] = metricsAll['pre'][0,Joint().count]
        metrics[Joint().count+3,0] = metricsAll['rec'][0,Joint().count]

        # print AP
        print("Multiple Object Tracking (MOT) metrics:")
        eval_helpers.printTable(metrics,motHeader=True)

if __name__ == "__main__":
   main()
