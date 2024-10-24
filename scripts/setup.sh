#!/bin/bash

# SETUP DATA
# create the data dir
mkdir ../data
mkdir ../data/egoschema
mkdir ../data/nextqa
mkdir ../data/intentqa
mkdir ../data/activitynet
# TODO add more datasets here
# link the data
ln -s /PATH/TO/DATA/egoschema/* ../data/egoschema/
ln -s /PATH/TO/DATA/nextqa/* ../data/nextqa/
ln -s /PATH/TO/DATA/intentqa/* ../data/intentqa/
ln -s /PATH/TO/DATA/activitynetqa/* ../data/activitynet/
# TODO add more datasets here

# SETUP GROUNDINGDINO
# create the weights dir in GroundingDINO
# TODO uncomment the following line if you want to use GroundingDINO
# mkdir ../toolbox/GroundingDINO/weights
# link the GroundingDINO dependencies
# TODO uncomment the following line if you want to use GroundingDINO
# ln -s /path/to/GroundingDINO/weights/* ../toolbox/GroundingDINO/weights/

# SETUP LAVILA
# create the required dir
mkdir ../toolbox/lavila_video_captioner/modelzoo/
# link the LaViLa dependencies
# TODO make sure to change the path to the model checkpoints depending on your download location
ln -s /PATH/TO/MODEL_CHECKPOINTS/lavila/modelzoo/* ../toolbox/lavila_video_captioner/modelzoo/

# SETUP UNIVTG
# create the results dir in UnivTG
mkdir ../toolbox/UniVTG/results
mkdir ../toolbox/UniVTG/results/omni
mkdir ../toolbox/UniVTG/results/omni/pretrained
mkdir ../toolbox/UniVTG/results/omni/finetuned
# link the UnivTG dependencies
# TODO make sure to change the path to the model checkpoints depending on your download location
ln -s /PATH/TO/MODEL_CHECKPOINTS/UniVTG/results/omni/pretrained/* ../toolbox/UniVTG/results/omni/pretrained/
ln -s /PATH/TO/MODEL_CHECKPOINTS/UniVTG/results/omni/finetuned/* ../toolbox/UniVTG/results/omni/finetuned/

# create slurm dir
mkdir ../slurm