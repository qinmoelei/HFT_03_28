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
        del previous_action
        del x
        del avaliable_action
        return action


def collect_experience(id, actor: actor, environment):
    # tranjectory = []
    specific_env = environment(random_start=id)
    done = False
    s, info = specific_env.reset()
    while not done:
        action = actor.act(s, info)
        s_, r, done, info_ = specific_env.step(action)
        # tranjectory.append((s, info, action, r, s_, info_, done))
        s, info = s_, info_
    optimal_result = np.max(
        specific_env.q_table[0][0][:]) / specific_env.required_money
    final_return_rate = specific_env.final_balance / (specific_env.required_money+1e-12)
    indicator = (optimal_result - final_return_rate) * optimal_result * 10000
    return id, indicator, optimal_result, final_return_rate


def collect_multiple_experience(id_list, actor, environment,num_process=10):
    ctx = torch.multiprocessing.get_context("spawn")
    # pool = ctx.Pool(processes=num_process)
    pool = ctx.Pool()
    func = partial(collect_experience, actor=actor, environment=environment)
    result = pool.map(func, id_list)
    pool.close()
    pool.join()
    id_list = []
    # tranjectory_list = []
    indicator_list = []
    optimal_list = []
    final_return_rate_list = []
    for i in range(len(result)):
        id, indicator, optimal_result, final_return_rate = result[
            i]
        id_list.append(id)
        # tranjectory_list.append(tranjectory)
        indicator_list.append(indicator)
        optimal_list.append(optimal_result)
        final_return_rate_list.append(final_return_rate)
    return range(
        len(result)
    ), id_list, indicator_list, optimal_list, final_return_rate_list


if __name__ == "__main__":
    data = pd.read_feather(
        "data/BTCTUSD/2023/train.feather")
    chunk_length=14400
    start_env = partial(
        Training_Env,
        df=data,
        tech_indicator_list=tech_indicator_list,
        transcation_cost=transcation_cost,
        back_time_length=back_time_length,
        max_holding_number=max_holding_number,
        chunck_length=chunk_length,
    )
    model = masked_net1(66, 11, 32)
    agent = actor(model=model, seed=12345)
    #multi preprocessing
    id_list = range(0, len(data), chunk_length)
    print("id_list", id_list)
    index_list, id_list_test, indicator_list , optimal_list, final_return_rate_list= collect_multiple_experience(
        id_list, agent, start_env)
    print(id_list_test)
    print(indicator_list)

    # func = partial(collect_experience, actor=agent, environment=start_env)

    # ctx = torch.multiprocessing.get_context("spawn")
    # pool = ctx.Pool()

    # result = pool.map(func, id_list)
    # pool.close()
    # pool.join()
    # print(result)
