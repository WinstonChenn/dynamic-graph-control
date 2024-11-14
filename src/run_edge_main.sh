export CUDA_VISIBLE_DEVICES="0"
for seed in 0 
do
    python edge_main.py --num_nodes 30 \
        --lat_dim 2 --edge_thresh 0.6 --int_param 1.0 1.0 \
        --init_inf_prop 0.1 --inf_thresh 0.3 --max_inf_days 10 \
        --inf_param 1.0 1.0 --sus_param 1.0 1.0 --rec_param 1.0 1.0 \
        --num_train 100 --num_val 100 --num_test 100 \
        --lr 1e-2 --epochs 100 --seed $seed --model_name SAGE --overwrite_model
done

