#!/bin/tcsh
#$ -S /bin/tcsh
#$ -cwd
#$ -V
#$ -o example.out
#$ -e example.err
#$ -m aes
###$ -M nada64@imec.be
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
 
#cd /imec/other/csainfra/nada64/DianaTraining

python imagenetresnet18.py