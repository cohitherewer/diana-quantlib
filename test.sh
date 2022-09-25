#!/bin/sh
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!


export PATH=/imec/other/csainfra/nada64/anaconda3/bin:$PATH
source activate gpu 
cd /imec/other/csainfra/nada64/DianaTraining
python serializationtest.py