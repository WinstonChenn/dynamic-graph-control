# dynamic-graph-control


### Train Dynamic GNN model: 
- run `./run_dynamic_forecast_main`
### Evaluate Forecasting: 
- run `./run_dynamic_forecast_main`
- Plot evaluation results with `./notebooks/plot_forecasting.ipynb`
### Evaluate Policy:
- run `./run_dynamic_forecast_main`
- Plots for historical disease can be found in: `Dynamic-Graph-Control/figures/#nodes=30/latDim=2_edgeThresh=0.6/initProp=0.1_infThresh=0.3_maxDays=10_infParam=[1.0, 1.0]_susParam=[1.0, 1.0]_recParam=[1.0, 1.0]_intParam=[1.0, 1.0]/InterveneRate=0.1/#train=100_#val=100_#test=100/seed=0/Policy/model=SAGELSTM/#epochs=1000_batch=32_lr=0.001_l2=0.0005_patience=100_delta=0.0001/initProp=0.1_infThresh=0.3_maxDays=10_infParam=[1.0, 1.0]_susParam=[1.0, 1.0]_recParam=[1.0, 1.0]_intParam=[1.0, 1.0]`
- Plots for novel disease can be found in: `Dynamic-Graph-Control/figures/#nodes=30/latDim=2_edgeThresh=0.6/initProp=0.1_infThresh=0.3_maxDays=10_infParam=[1.0, 1.0]_susParam=[1.0, 1.0]_recParam=[1.0, 1.0]_intParam=[1.0, 1.0]/InterveneRate=0.1/#train=100_#val=100_#test=100/seed=0/Policy/model=SAGELSTM/#epochs=1000_batch=32_lr=0.001_l2=0.0005_patience=100_delta=0.0001/initProp=0.1_infThresh=0.3_maxDays=5_infParam=[1.0, 1.0]_susParam=[1.0, 1.0]_recParam=[1.0, 1.0]_intParam=[1.0, 1.0]`