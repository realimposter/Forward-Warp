#!/bin/bash
work_path=$(dirname $(readlink -f $0))
cd ${work_path}/Forward_Warp/cuda/
# conda activate pytorch
python setup.py install --verbose 2>&1 | tee /var/log/setup.log
cd ../../
python setup.py install
