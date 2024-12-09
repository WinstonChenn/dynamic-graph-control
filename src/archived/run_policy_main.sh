export CUDA_VISIBLE_DEVICES="0"
for seed in 0 
do
    python policy_main.py  --seed $seed \
        --num_nodes 30 --lat_dim 2 --edge_thresh 0.6 \
        --train_init_inf_prop 0.1 --train_inf_thresh 0.3 --train_max_inf_days 10 \
        --train_inf_param 1.0 1.0 --train_sus_param 1.0 1.0 --train_rec_param 1.0 1.0 \
        --num_train 100 --num_val 100 --num_test 100 \
        --model_name SAGE --batch 32 --epochs 1000 --lr 1e-3 \
        --l2 5e-4 --patience 10 --delta 1e-4 \
        --eval_init_inf_prop 0.1 --eval_inf_thresh 0.3 --eval_max_inf_days 10 \
        --eval_inf_param 1.0 1.0 --eval_sus_param 1.0 1.0 --eval_rec_param 1.0 1.0 \
        --num_times 200 --num_int 1
done

