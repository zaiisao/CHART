#!/bin/bash
# after the dirac run writes its model, run the EVIDENCE-OFF (alpha=0) + tempering sweep
D=/home/sogang/jaehoon/VBPM_reintegration/vbpm_fix/varB_pf
until [ -f $D/res_dirac.json ]; do sleep 30; done
sleep 10
cd /home/sogang/jaehoon/VBPM_reintegration
CUDA_VISIBLE_DEVICES=3 /home/sogang/mnt/db_2/anaconda3/envs/chart/bin/python \
  vbpm_fix/varB_pf/eval_extra.py --mode dirac --ckpt $D/res_dirac_model.pt \
  --alphas 0,0.1,0.3,1.0 --n_eval 30 --K 500 --out $D/res_dirac_extra.json
