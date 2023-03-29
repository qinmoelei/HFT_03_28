CUDA_VISIBLE_DEVICES=0 nohup python RL/trader.py --transcation_cost 0 > original_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python RL/trader.py --transcation_cost 0.001 > original_001.log 2>&1 &