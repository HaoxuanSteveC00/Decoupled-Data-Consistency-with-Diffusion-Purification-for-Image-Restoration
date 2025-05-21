#!/bin/bash

#SBATCH --partition prod
#SBATCH --job-name=sr
#SBATCH --output=/home/ml/zach/Decoupled-Data-Consistency-with-Diffusion-Purification-for-Image-Restoration/log/sr.out
#SBATCH --error=/home/ml/zach/Decoupled-Data-Consistency-with-Diffusion-Purification-for-Image-Restoration/log/sr.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPUMODEL_A6000


######################
# Begin work section #
######################
conda init
conda activate bmposterior
python dcdp.py --task_config=./task_configurations/super_resolution_config.yaml --purification_config=./purification_configurations/purification_config_super_resolution_imagenet.yaml \
                 --model_config=./model_configurations/model_config_ImageNet.yaml
