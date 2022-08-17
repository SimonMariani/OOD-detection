# This file runs the training, inference and evaluation
source /home/miniconda3/bin/activate uncertainty
/home/miniconda3/envs/uncertainty/bin/python /home/code/train.py
/home/miniconda3/envs/uncertainty/bin/python /home/code/inference.py
/home/miniconda3/envs/uncertainty/bin/python /home/code/evaluation.py