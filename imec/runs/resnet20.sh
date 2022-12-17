#!/bin/tcsh
#$ -S /bin/tcsh
#$ -cwd
#$ -V
#$ -o outputs/resnet20.out
#$ -e outputs/resnet20.err
#$ -q batch_gpu.q 
#$ -l gpu_name=v100
#$ -N QAT_JOB 

module purge

conda activate gpu 


 
cd /imec/other/csainfra/nada64/DianaTraining
# Serialize model 
#python generate_config.py
# Quantize model with selected config (Make sure to edit the file accordingly)
#python quantize_model.py 
# get dataset scale 
#python datasetscale.py
# start training , add -h to see the different options
# begin with flaoting point model training
#python trainer.py  
# train fq_model 
python trainer.py --lr 0.01 --momentum 0.9 --weight_decay 5e-5 --batch_size 256 --num_workers 8 --num_epochs 1 --scale 0.03125 --quant_steps 0 --config_pth ./serialized_models/resnet20.yaml --stage fq --fp_pth ./zoo/cifar10/resnet20/FP_weights.pth --quantized_pth ./zoo/cifar10/resnet20/quantized.pth --checkpoint_dir ./zoo/cifar10/resnet20
# train hw_mapped model 
#python trainer.py 

# python trainer.py --lr 0.01 --weight_decay 5e-4 --batch_size 256 --num_workers 8 --num_epochs 20 --scale 0.03125 --quant_steps 0 --config_pth ./serialized_models/resnet20.yaml --stage fq --fp_pth ./zoo/cifar10/resnet20/FP_weights.pth --quantized_pth ./zoo/cifar10/resnet20/quantized.pth --checkpoint_dir ./z00/cifar10/resnet20