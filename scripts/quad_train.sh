python train.py --config-name quadrotor settings.train_mode='dt_dyn'

python train.py --config-name quadrotor settings.train_mode='ct_ctl'

python train.py --config-name quadrotor settings.train_mode='dt_dyn' train_dt_dyn.reach.eps_init=0.2 train_dt_dyn.reach.eps_final=0.1

python train.py --config-name quadrotor settings.train_mode='dt_dyn' train_dt_dyn.reach.eps_init=0.3 train_dt_dyn.reach.eps_final=0.2

python train.py --config-name quadrotor settings.train_mode='dt_dyn' train_dt_dyn.reach.eps_init=0.5 train_dt_dyn.reach.eps_final=0.3

python train.py --config-name quadrotor settings.train_mode='dt_dyn' train_dt_dyn.reach.mode='none'

python train.py --config-name quadrotor settings.train_mode='ct_ctl' train_ct_ctl.reach.mode='none'

python train.py --config-name quadrotor settings.train_mode='ct_ctl' train_ct_ctl.reach.eps_init=0.03 train_ct_ctl.reach.eps_final=0.02 train_ct_ctl.reach.weight=0.0006

python train.py --config-name quadrotor settings.train_mode='ct_ctl' train_ct_ctl.reach.eps_init=0.03 train_ct_ctl.reach.eps_final=0.02 train_ct_ctl.reach.weight=0.0008

python train.py --config-name quadrotor settings.train_mode='ct_ctl' train_ct_ctl.reach.eps_init=0.03 train_ct_ctl.reach.eps_final=0.02 train_ct_ctl.reach.weight=0.001

python train.py --config-name quadrotor settings.train_mode='ct_ctl' train_ct_ctl.reach.eps_init=0.03 train_ct_ctl.reach.eps_final=0.02 train_ct_ctl.reach.weight=0.0012


python train.py --config-name quadrotor settings.train_mode='ct_ctl' train_ct_ctl.reach.eps_init=0.02 train_ct_ctl.reach.eps_final=0.01 train_ct_ctl.reach.weight=0.0006

python train.py --config-name quadrotor settings.train_mode='ct_ctl' train_ct_ctl.reach.eps_init=0.02 train_ct_ctl.reach.eps_final=0.01 train_ct_ctl.reach.weight=0.0008

python train.py --config-name quadrotor settings.train_mode='ct_ctl' train_ct_ctl.reach.eps_init=0.02 train_ct_ctl.reach.eps_final=0.01 train_ct_ctl.reach.weight=0.001

python train.py --config-name quadrotor settings.train_mode='ct_ctl' train_ct_ctl.reach.eps_init=0.02 train_ct_ctl.reach.eps_final=0.01 train_ct_ctl.reach.weight=0.0012