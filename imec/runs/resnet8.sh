#!/bin/tcsh
#$ -S /bin/tcsh
#$ -cwd
#$ -V
#$ -o outputs/resnet8.out
#$ -e outputs/resnet8.err
#$ -m aes
#$ -q batch_gpu.q
#$ -l gpu_name=v100
#$ -N GPU_JOB
#$ -v nada64


###################################################################
# Set these values to configure
###################################################################
module purge

conda activate gpu 

# load  other modules here
# run script -- arguments assume pytorch lightning is used
 
cd /imec/other/csainfra/nada64/DianaTraining

python resnet8.py
#python -m dawn --data_dir=../../data/cifar10