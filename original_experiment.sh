CUDA_VISIBLE_DEVICES=0 nohup python RL/trader.py --transcation_cost 0 --seed 12345 > log/original_0_12345.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python RL/trader.py --transcation_cost 0 --seed 23451 > log/original_0_23451.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python RL/trader.py --transcation_cost 0 --seed 34512 > log/original_0_34512.log 2>&1 &


CUDA_VISIBLE_DEVICES=3 nohup python RL/trader.py --transcation_cost 0.001 --seed 12345 > log/original_001_12345.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python RL/trader.py --transcation_cost 0.001 --seed 23451 > log/original_001_23451.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python RL/trader.py --transcation_cost 0.001 --seed 34512 > log/original_001_34512.log 2>&1 &
