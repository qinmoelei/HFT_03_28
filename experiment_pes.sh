# CUDA_VISIBLE_DEVICES=0 nohup python RL/trader_pes.py --transcation_cost 0.001 --seed 12345 > log/pes_001_12345.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python RL/trader_pes.py --transcation_cost 0.001 --seed 23451 > log/pes_001_23451.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python RL/trader_pes.py --transcation_cost 0.001 --seed 34512 > log/pes_001_34512.log 2>&1 &




CUDA_VISIBLE_DEVICES=3 nohup python RL/trader_pes.py --transcation_cost 0.0 --seed 12345 > log/pes_0_12345.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python RL/trader_pes.py --transcation_cost 0.0 --seed 23451 > log/pes_0_23451.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python RL/trader_pes.py --transcation_cost 0.0 --seed 34512 > log/pes_0_34512.log 2>&1 &
