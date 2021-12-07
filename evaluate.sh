#! /bin/bash
# predications is where the predication results store
# results is where the convert results store, and -t is the threshold of the pose score
# pose_number is how many number of keypoints we want to see in a person
# pose_threshold is the score we set to filter the keypoints whose score is small 
# ground_truth='/export/home/cyh/mygithub/PoseGCN/data/annotations/val_2017/'
ground_truth='/PGPT/data/demodataset/annotations'
predictions='/PGPT/results/demo'
results='/PGPT/results/evaluate'
pose_number=0
pose_threshold=0.5

echo "Ground Truth is ${ground_truth}"
echo "Start evaluate the results of ${predictions}" 
echo "Store the final resuts in ${results}"
echo "===============================" 

cd ./evaluate
python -u convert_pose.py -d ${predictions} -n ${pose_number} -t ${pose_threshold} 2>&1 | tee -a ../log/${results}_convert_log.txt
python -u evaluate_posetrack.py -s -e -t -g ${ground_truth} -p ${predictions}"_"${pose_number}"_"${pose_threshold}"/" -o ${results} 2>&1 | tee ../log/${results}_evaluate_log.txt

cd ../
