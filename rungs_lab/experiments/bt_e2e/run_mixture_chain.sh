#!/bin/sh
cd /home/sogang/jaehoon/VBPM/experiments/bt_e2e || exit 1
export PYTHONPATH=/home/sogang/jaehoon/VBPM:/home/sogang/jaehoon/VBPM/external/beat_transformer/code
while pgrep -f "train_r2_em_fps2.py" > /dev/null; do sleep 20; done
/home/sogang/mnt/db_2/anaconda3/envs/chart/bin/python mixture_kernel_probe.py > mixture_kernel_probe.log 2>&1
echo "mixture probe exit: $?" >> mixture_kernel_probe.log
