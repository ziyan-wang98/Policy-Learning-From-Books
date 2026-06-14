# EXP_NAME='comment=normob-lr&pi_alg_type=CIQL&strategy=all_replaced&alpha=60.0&target_value=0.02&coef_t=0.5&coef_r=0.5&lr_rescale=0.5&trace_real_num=0&extra_real_traj_num=0&skip_gen_data=False&rollout_num=0&fake_rollout_num=-1&obs_stack_num=4_20240309185736'
EXP_NAME='comment=normob-lr-ent-0.4&pi_alg_type=CIQL&strategy=all_replaced&alpha=60.0&target_value=0.02&coef_t=0.5&coef_r=0.5&lr_rescale=0.5&trace_real_num=0&extra_real_traj_num=0&skip_gen_data=False&rollout_num=0&fake_rollout_num=-1&obs_stack_num=4_20240312162327'
python learning/load_and_eval.py --exp_name $EXP_NAME --eval_times $1
python learning/load_and_eval.py --exp_name $EXP_NAME --eval_times $1
python learning/load_and_eval.py --exp_name $EXP_NAME --eval_times $1
python learning/load_and_eval.py --exp_name $EXP_NAME --eval_times $1

