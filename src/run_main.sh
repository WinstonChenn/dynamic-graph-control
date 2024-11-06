
for seed in 0 
do
    python main.py --num_nodes 30 \
        --lat_dim 30 --edge_thresh 0.55 --int_param 1.0 1.0 \
        --init_inf_prop 0.1 --inf_thresh 0.3 --max_inf_days 10 \
        --inf_param 1.0 1.0 --sus_param 1.0 1.0 --rec_param 1.0 1.0 \
        --num_train 100 --num_val 100 --num_test 100 \
        --lr 1e-2 --epochs 200 --seed $seed
done
