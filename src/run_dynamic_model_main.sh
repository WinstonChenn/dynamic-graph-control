export CUDA_VISIBLE_DEVICES="0"
for seed in 1
do
    python dynamic_model_main.py --seed $seed \
        --num_nodes 30 --lat_dim 2 --edge_thresh 0.6 \
        --init_inf_prop 0.1 --inf_thresh 0.3 --max_inf_days 10 \
        --inf_param 1.0 1.0 --sus_param 1.0 1.0 --rec_param 1.0 1.0 \
        --int_param 1.0 1.0 --intervene_rate 0.1 \
        --num_train 100 --num_val 100 --num_test 100 \
        --model_name SAGELSTM --batch 32 --epochs 1000 --lr 1e-3 \
        --l2 5e-4 --patience 100 --delta 1e-4 --overwrite_model
done
