#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --job-name=dino
#SBATCH --mail-user=s319439@studenti.polito.it
#SBATCH --mail-type=END
#SBATCH --partition=cuda
#SBATCH --output=/home/nberardo/dino/train.log
#SBATCH --error=/home/nberardo/dino/train.err
###### 1 Load the module
module load nvidia/cudasdk
module load intel/python/3
# 2 Activate the virtual environment
source /home/nberardo/myenv/bin/activate
# 3 Run the python script
python -m torch.distributed.launch --nproc_per_node=4 eval_linear.py \
  --pretrained_weights /home/nberardo/dino/experiments/checkpoint.pth \
  --arch resnet50 --data_path /home/eaiello/ImageNet/
# python -m torch.distributed.launch --nproc_per_node=4 main_dino.py --arch resnet50 --optimizer sgd --lr 0.03 --weight_decay 1e-4 --weight_decay_end 1e-4 --global_crops_scale 0.14 1 --local_crops_scale 0.05 0.14 --data_path /home/eaiello/ImageNet/train --output_dir /home/nberardo/dino/experiments


