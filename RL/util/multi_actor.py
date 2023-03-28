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
from env.env import Training_Env,tech_indicator_list,transcation_cost,back_time_length,max_holding_number
from functools import partial
from model.net import masked_net1
from multiprocessing import Pool

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
    def __init__(self, model, seed) -> None:
        seed_torch(seed)
        self.trader = model
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

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
            actions_value = self.eval_net.forward(x, previous_action,
                                                  avaliable_action,
                                                  holding_length)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0]
        else:
            action_choice = []
            for i in range(len(info["avaliable_action"])):
                if info["avaliable_action"][i] == 1:
                    action_choice.append(i)
            action = random.choice(action_choice)
        return action


def collect_experience(actor:actor,environment,id):
    tranjectory=[]
    specific_env=environment(random_start=id)
    done=False
    s,info=specific_env.reset()
    while not done:
        action=actor.act(s,info)
        s_, r, done, info_=specific_env.step(action)
        tranjectory.append(s,info,action,r,s_,info_,done)
        s,info=s_,info_
    optimal_result=np.max(specific_env.q_table[0][0][:])/specific_env.required_money
    final_return_rate=specific_env.final_balance/specific_env.required_money
    indicator=(optimal_result-final_return_rate)*optimal_result

    return id,tranjectory,indicator



if __name__ == "__main__":
    data=pd.read_feather("/data1/sunshuo/qml/HFT/HFT_03_27/data/BTCTUSD/2023/test.feather")
    start_env=partial(Training_Env, df=data,
        tech_indicator_list=tech_indicator_list,
        transcation_cost=transcation_cost,
        back_time_length=back_time_length,
        max_holding_number=max_holding_number,
        chunck_length=1000,)
    model=masked_net1(66,11,32)
    agent=actor(model=model,seed=12345)
    #multi preprocessing
    pool = Pool()
    args=[0,7200]
    func=partial(collect_experience,actor=agent,environment=start_env)
    print(func)


    result = pool.map(func, args)
    pool.close()
    pool.join()
    print(result)


    #actor pool