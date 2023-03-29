import ray
from ray.util import ActorPool
import sys
import pandas as pd
import numpy as np

sys.path.append(".")
import torch
import random
import os
import numpy as np
from env.env import Training_Env, tech_indicator_list, transcation_cost, back_time_length, max_holding_number
from functools import partial
from model.net import masked_net1
from multiprocessing import Pool
import time


def seed_torch(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class actor:
    def __init__(self, model, seed, epsilon=0.9) -> None:
        seed_torch(seed)
        self.trader = model
        # if torch.cuda.is_available():
        #     self.device = "cuda"
        # else:
        self.device = "cpu"
        self.epsilon = epsilon
        self.trader = self.trader.to(self.device)

    def act(self, state, info):
        x = torch.unsqueeze(torch.FloatTensor(state).reshape(-1),
                            0).to(self.device)
        previous_action = torch.unsqueeze(
            torch.tensor(info["previous_action"]).long().to(self.device),
            0).to(self.device)
        avaliable_action = torch.unsqueeze(
            info["avaliable_action"].to(self.device), 0).to(self.device)
        holding_length = torch.unsqueeze(
            torch.tensor(info["holding_length"]).float(), 0).to(self.device)

        if np.random.uniform() < self.epsilon:
            actions_value = self.trader.forward(
                x,
                previous_action,
                avaliable_action,
            )
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0]
        else:
            action_choice = []
            for i in range(len(info["avaliable_action"])):
                if info["avaliable_action"][i] == 1:
                    action_choice.append(i)
            action = random.choice(action_choice)
        return action


def collect_experience(id, actor: actor, environment):
    tranjectory = []
    specific_env = environment(random_start=id)
    done = False
    s, info = specific_env.reset()
    while not done:
        action = actor.act(s, info)
        s_, r, done, info_ = specific_env.step(action)
        tranjectory.append((s, info, action, r, s_, info_, done))
        s, info = s_, info_
    optimal_result = np.max(
        specific_env.q_table[0][0][:]) / specific_env.required_money
    final_return_rate = specific_env.final_balance / specific_env.required_money
    indicator = (optimal_result - final_return_rate) * optimal_result * 10000

    return id, tranjectory, indicator


if __name__ == "__main__":
    data = pd.read_feather(
        "/data1/sunshuo/qml/HFT/HFT_03_27/data/BTCTUSD/2023/test.feather")
    start_env = partial(
        Training_Env,
        df=data,
        tech_indicator_list=tech_indicator_list,
        transcation_cost=transcation_cost,
        back_time_length=back_time_length,
        max_holding_number=max_holding_number,
        chunck_length=1000,
    )
    model = masked_net1(66, 11, 32)
    agent = actor(model=model, seed=12345)
    #multi preprocessing

    func = partial(collect_experience, actor=agent, environment=start_env)
    start1 = time.time()
    func(id=0)
    end1 = time.time()
    print("to run one environment, it will take {} seconds".format(end1 -
                                                                   start1))

    start1 = time.time()
    func(id=1000)
    end1 = time.time()
    print("to run one environment, it will take {} seconds".format(end1 -
                                                                   start1))

    ctx = torch.multiprocessing.get_context("spawn")
    pool = ctx.Pool()
    args = [
        0,
        1000,
        2000,
        3000,
        4000,
        5000,
        6000,
        7000,
        8000,
        9000,
        10000,
        11000,
        12000,
        13000,
        14000,
        15000,
        16000,
        17000,
        18000,
        19000,
    ]
    result = pool.map(func, args)
    pool.close()
    pool.join()
    # print(result)
    end2 = time.time()
    print("to run 20 environment, it will take {} seconds".format(end2 - end1))
    for i in range(len(result)):
        id = result[i][0]
        print(id)
        indicator = result[i][2]
        print(indicator)
    #actor pool