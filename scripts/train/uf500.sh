SEED=123

#pretrain the value function
python train.py --pretrain_mode value --update_epochs 1 --lr_init 1e-3 --lr_final 1e-4 --total_trials 10 --episodes_per_trial 1 --project_name rlnmc_supervised_pretraining_uf500 --problem_config config_500 --train_config config_train_uf500 --save_name pretrained_value --save_run_id value_${SEED} --seed $SEED

#pretrain policy and value
python train.py --pretrain_mode action_and_value --update_epochs 25 --lr_init 1e-3 --lr_final 1e-4 --total_trials 1 --episodes_per_trial 1 --project_name rlnmc_supervised_pretraining_uf500 --problem_config config_500 --train_config config_train_uf500 --save_name pretrain_action_and_val --save_run_id action_and_value_${SEED} --load_name pretrained_value --load_run_id value_${SEED} --load_project_name rlnmc_supervised_pretraining_uf500 --load_trained 1 --seed $SEED

#run reinforcement learning
python train.py --pretrain_mode none --project_name rlnmc_train_uf500 --problem_config config_500 --train_config config_train_uf500 --load_name pretrain_action_and_val --load_run_id action_and_value_${SEED} --load_project_name rlnmc_supervised_pretraining_uf500 --load_trained 1 --save_run_id final_model_${SEED} --seed $SEED