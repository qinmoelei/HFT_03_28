CUDA_VISIBLE_DEVICES=0 nohup python RL/trader_pes.py --transcation_cost 0.001 --seed 12345 > log/pes_001_12345.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python RL/trader_pes.py --transcation_cost 0.0 --seed 12345 > log/pes_0_12345.log 2>&1 &
