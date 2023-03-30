import sys

sys.path.append(".")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from torch.utils.tensorboard import SummaryWriter
from RL.util.replay_buffer import Multi_step_ReplayBuffer_multi_info
from RL.util.multi_step_decay import get_ada, get_epsilon
import random
from tqdm import tqdm
import argparse
from model.net import *
import numpy as np
import torch
from torch import nn
import yaml
import pandas as pd
from env.env import Testing_env, Training_Env
from RL.util.multi_actor import *
from RL.util.episode_selector import *
#相比最原版增加了优化器正则和clip norm
#调试调整了epoch number
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--buffer_size",
    type=int,
    default=100000,
    help="the number of transcation we store in one memory",
)

parser.add_argument(
    "--q_value_memorize_freq",
    type=int,
    default=100,
    help="the number of step we store one q value",
)
# the replay buffer get cleared get every time the target net get updated
parser.add_argument(
    "--batch_size",
    type=int,
    default=512,
    help="the number of transcation we learn at a time",
)
parser.add_argument(
    "--eval_update_freq",
    type=int,
    default=1000,
    help="the number of step before we do one update",
)

parser.add_argument("--lr", type=float, default=1e-4, help="the learning rate")

parser.add_argument("--epsilon",
                    type=float,
                    default=0.99,
                    help="the learning rate")
parser.add_argument("--update_times",
                    type=int,
                    default=20,
                    help="the update times")
parser.add_argument("--gamma", type=float, default=1, help="the learning rate")

parser.add_argument(
    "--target_freq",
    type=int,
    default=50,
    help=
    "the number of updates before the eval could be as same as the target and clear all the replay buffer",
)
parser.add_argument("--ada",
                    type=float,
                    default=0.1,
                    help="the coffient for auxliary task")
parser.add_argument(
    "--num_sample",
    type=int,
    default=300,
    help="the number of sampling during one epoch",
)

parser.add_argument(
    "--transcation_cost",
    type=float,
    default=0.0,
    help="the transcation cost of not holding the same action as before",
)
# view this as a a task
parser.add_argument("--back_time_length",
                    type=int,
                    default=1,
                    help="the length of the holding period")
"""notice that since it is a informer sctructure problem which involes twice conv on the time series to compress,
therefore back_time_length must be larger than or equal 4"""

parser.add_argument(
    "--result_path",
    type=str,
    default="result/PES",
    help="the path for storing the test result",
)
parser.add_argument(
    "--seed",
    type=int,
    default=12345,
    help="the random seed for training and sample",
)

parser.add_argument(
    "--n_step",
    type=int,
    default=10,
    help="the number of step we have in the td error and replay buffer",
)
parser.add_argument(
    "--chunk_length",
    type=int,
    default=14400,
    help="the chunk length we have in one episode",
)
parser.add_argument(
    "--preserve_bonus",
    type=float,
    default=0,
    help="the bonus we give to the agent if it preserves its action",
)
parser.add_argument(
    "--reward_scale",
    type=float,
    default=1,
    help="the scale factor we put in reward",
)

parser.add_argument(
    "--ada_decay",
    type=int,
    default=1,
    help="whether we have regulizer decay ",
)

parser.add_argument(
    "--ada_decay_coffient",
    type=float,
    default=0.9,
    help="the coffient for decay",
)
parser.add_argument(
    "--epsilon_decay",
    type=int,
    default=1,
    help="whether we have epsilon decay ",
)

parser.add_argument(
    "--epsilon_decay_coffient",
    type=float,
    default=0.9,
    help="the coffient for decay",
)


def read_yaml_to_dict(yaml_path: str, ):
    with open(yaml_path) as file:
        dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
        return dict_value


def seed_torch(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class DQN(object):
    def __init__(self, args):  # 定义DQN的一系列属性
        self.seed = args.seed
        seed_torch(self.seed)
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.model_path = os.path.join(
            args.result_path,
            "commission_fee_{}".format(args.transcation_cost),
            "seed_{}".format(self.seed),
        )
        # new dataset
        self.max_holding_number = 0.01

        self.train_df = pd.read_feather("data/BTCTUSD/2023/train.feather")
        self.test_df = pd.read_feather("data/BTCTUSD/2023/test.feather")

        self.preserve_bonus = args.preserve_bonus
        self.reward_scale = args.reward_scale
        self.log_path = os.path.join(self.model_path, "log")
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.writer = SummaryWriter(self.log_path)
        self.update_counter = 0
        self.q_value_memorize_freq = args.q_value_memorize_freq
        self.grad_clip = 3
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        data_path = os.path.join(os.getcwd(), "..", "short_term_data")

        self.tech_indicator_list = read_yaml_to_dict(
            os.path.abspath(os.path.join(
                data_path, "tech_indicator_list.yml")))["tech_indicator_list"]
        self.transcation_cost = args.transcation_cost
        self.back_time_length = args.back_time_length
        self.test_ev_instance = Testing_env(
            df=self.test_df,
            tech_indicator_list=self.tech_indicator_list,
            transcation_cost=self.transcation_cost,
            back_time_length=self.back_time_length,
            max_holding_number=self.max_holding_number,
        )
        # here is where we define the difference among different net
        self.n_action = self.test_ev_instance.action_space.n
        self.n_state = self.test_ev_instance.reset()[0].reshape(-1).shape[0]
        self.eval_net, self.target_net = masked_net1(
            self.n_state, self.n_action, 128).to(self.device), masked_net1(
                self.n_state, self.n_action,
                128).to(self.device)  # 利用Net创建两个神经网络: 评估网络和目标网络
        self.hardupdate()
        self.update_times = args.update_times
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(),
                                          lr=args.lr,
                                          weight_decay=1e-2)
        self.loss_func = nn.MSELoss()  # 使用均方损失函数 (loss(xi, yi)=(xi-yi)^2)
        self.target_freq = args.target_freq
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.n_step = args.n_step
        self.eval_update_freq = args.eval_update_freq
        self.buffer_size = args.buffer_size
        self.num_sample = args.num_sample
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.target_freq, gamma=1)
        self.chunk_length = args.chunk_length
        self.epsilon = args.epsilon
        self.ada = args.ada
        self.epsilon_decay = args.epsilon_decay
        self.ada_decay = args.ada_decay
        self.epsilon_decay_coffient = args.epsilon_decay_coffient
        self.ada_decay_coffient = args.ada_decay_coffient

    def update(
        self,
        states: torch.tensor,
        info: dict,
        actions: torch.tensor,
        rewards: torch.tensor,
        next_states: torch.tensor,
        info_: dict,
        dones: torch.tensor,
    ):
        print("updating")
        b = states.shape[0]
        q_eval = self.eval_net(
            states.reshape(b, -1),
            info["previous_action"].long(),
            info["avaliable_action"],
        ).gather(1, actions)
        demonstration = info["q_action"]
        predict_action_distrbution = self.eval_net(
            states.reshape(b, -1),
            info["previous_action"].long(),
            info["avaliable_action"],
        )
        KL_div = F.kl_div(
            (predict_action_distrbution.softmax(dim=-1) + 1e-8).log(),
            demonstration,
            reduction="batchmean",
        )

        q_next = self.target_net(
            next_states.reshape(b, -1),
            info_["previous_action"].long(),
            info_["avaliable_action"],
        ).detach()

        q_target = rewards + torch.max(q_next, 1)[0].view(self.batch_size,
                                                          1) * (1 - dones)
        td_error = self.loss_func(q_eval, q_target)
        loss = td_error + KL_div * self.ada
        # print(td_error)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(),
                                       self.grad_clip)
        self.optimizer.step()
        self.update_counter += 1
        if self.update_counter % self.target_freq == 1:
            self.hardupdate()
            self.scheduler.step()
        return (
            KL_div.cpu(),
            td_error.cpu(),
            torch.mean(q_eval.cpu()),
            torch.mean(q_target.cpu()),
        )

    def hardupdate(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def act(self, state, info):
        x = torch.unsqueeze(torch.FloatTensor(state).reshape(-1),
                            0).to(self.device)
        previous_action = torch.unsqueeze(
            torch.tensor(info["previous_action"]).long().to(self.device),
            0).to(self.device)
        avaliable_action = torch.unsqueeze(
            info["avaliable_action"].to(self.device), 0).to(self.device)

        if np.random.uniform() < self.epsilon:
            actions_value = self.eval_net.forward(x, previous_action,
                                                  avaliable_action)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0]
        else:
            action_choice = []
            for i in range(len(info["avaliable_action"])):
                if info["avaliable_action"][i] == 1:
                    action_choice.append(i)
            action = random.choice(action_choice)
        return action

    def act_test(self, state, info):
        x = torch.unsqueeze(torch.FloatTensor(state).reshape(-1),
                            0).to(self.device)
        previous_action = torch.unsqueeze(
            torch.tensor(info["previous_action"]).long(), 0).to(self.device)
        avaliable_action = torch.unsqueeze(info["avaliable_action"],
                                           0).to(self.device)
        actions_value = self.eval_net.forward(x, previous_action,
                                              avaliable_action)
        action = torch.max(actions_value, 1)[1].data.cpu().numpy()
        action = action[0]
        return action

    def train(self):
        epoch_return_rate_train_list = []
        epoch_final_balance_train_list = []
        epoch_required_money_train_list = []
        epoch_reward_sum_train_list = []

        # epoch_number = int(len(self.train_df) / self.chunk_length)
        epoch_number = 10

        replay_buffer = Multi_step_ReplayBuffer_multi_info(
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            device=self.device,
            seed=self.seed,
            gamma=self.gamma,
            n_step=self.n_step,
        )
        step_counter = 0
        for sample in range(self.num_sample):
            
            self.actor = actor(self.eval_net, self.seed, epsilon=self.epsilon)
            self.start_list = range(0, len(self.train_df), self.chunk_length)
            
            Partial_training_env = partial(
                Training_Env,
                df=self.train_df,
                tech_indicator_list=self.tech_indicator_list,
                transcation_cost=self.transcation_cost,
                back_time_length=self.back_time_length,
                max_holding_number=self.max_holding_number,
                chunck_length=self.chunk_length,
            )
            index_list, id_list, priority_list, optimal_list, final_return_rate_list = collect_multiple_experience(
                self.start_list, self.actor, Partial_training_env)
            self.start_selector = start_selector(self.start_list, priority_list)
            start, index = self.start_selector.sample()
            train_env = Training_Env(
                df=self.train_df,
                tech_indicator_list=self.tech_indicator_list,
                random_start=start,
                chunck_length=self.chunk_length,
                transcation_cost=self.transcation_cost,
                back_time_length=self.back_time_length,
                max_holding_number=self.max_holding_number,
            )
            self.eval_net.to(self.device)
            s, info = train_env.reset()
            episode_reward_sum = 0
            while True:
                a = self.act(s, info)
                s_, r, done, info_ = train_env.step(a)
                if a == info["previous_action"]:
                    r = r + self.preserve_bonus
                replay_buffer.add(s, info, a, r * self.reward_scale, s_, info_,
                                  done)
                episode_reward_sum += r

                s, info = s_, info_
                step_counter += 1
                if step_counter % self.eval_update_freq == 0 and step_counter > (
                        self.batch_size + self.n_step):
                    for i in range(self.update_times):
                        (
                            states,
                            infos,
                            actions,
                            rewards,
                            next_states,
                            next_infos,
                            dones,
                        ) = replay_buffer.sample()
                        KL_div, td_error, q_eval, q_target = self.update(
                            states,
                            infos,
                            actions,
                            rewards,
                            next_states,
                            next_infos,
                            dones,
                        )
                        if self.update_counter % self.q_value_memorize_freq == 1:
                            self.writer.add_scalar(
                                tag="KL",
                                scalar_value=KL_div,
                                global_step=self.update_counter,
                                walltime=None,
                            )
                            self.writer.add_scalar(
                                tag="td_error",
                                scalar_value=td_error,
                                global_step=self.update_counter,
                                walltime=None,
                            )
                            self.writer.add_scalar(
                                tag="q_eval",
                                scalar_value=q_eval,
                                global_step=self.update_counter,
                                walltime=None,
                            )
                            self.writer.add_scalar(
                                tag="q_target",
                                scalar_value=q_target,
                                global_step=self.update_counter,
                                walltime=None,
                            )
                if done:
                    break
            final_balance, required_money = (
                train_env.final_balance,
                train_env.required_money,
            )
            self.writer.add_scalar(
                tag="return_rate_train",
                scalar_value=final_balance / required_money,
                global_step=sample,
                walltime=None,
            )
            self.writer.add_scalar(
                tag="final_balance_train",
                scalar_value=final_balance,
                global_step=sample,
                walltime=None,
            )
            self.writer.add_scalar(
                tag="required_money_train",
                scalar_value=required_money,
                global_step=sample,
                walltime=None,
            )
            self.writer.add_scalar(
                tag="reward_sum_train",
                scalar_value=episode_reward_sum,
                global_step=sample,
                walltime=None,
            )
            epoch_return_rate_train_list.append(final_balance / required_money)
            epoch_final_balance_train_list.append(final_balance)
            epoch_required_money_train_list.append(required_money)
            epoch_reward_sum_train_list.append(episode_reward_sum)

            
            
            
            if len(epoch_return_rate_train_list) == epoch_number:
                epoch_index = int((sample + 1) / epoch_number)
                if self.ada_decay:
                    self.ada = get_ada(
                        self.ada,
                        decay_freq=2,
                        ada_counter=epoch_index,
                        decay_coffient=self.ada_decay_coffient,
                    )
                if self.epsilon_decay:
                    self.epsilon = get_epsilon(
                        self.epsilon,
                        max_epsilon=1,
                        epsilon_counter=epoch_index,
                        decay_freq=2,
                        decay_coffient=self.epsilon_decay_coffient,
                    )
                mean_return_rate_train = np.mean(epoch_return_rate_train_list)

                self.writer.add_scalar(
                    tag="epoch_return_rate_train",
                    scalar_value=mean_return_rate_train,
                    global_step=epoch_index,
                    walltime=None,
                )

                epoch_path = os.path.join(self.model_path,
                                          "epoch_{}".format(epoch_index))
                if not os.path.exists(epoch_path):
                    os.makedirs(epoch_path)
                torch.save(
                    self.eval_net.state_dict(),
                    os.path.join(epoch_path, "trained_model.pkl"),
                )
                self.test(epoch_path)
                epoch_return_rate_train_list = []

    def test(self, epoch_path):
        self.eval_net.load_state_dict(
            torch.load(os.path.join(epoch_path, "trained_model.pkl")))
        s, info = self.test_ev_instance.reset()
        done = False
        action_list = []
        reward_list = []
        while not done:
            a = self.act_test(s, info)
            s_, r, done, info_ = self.test_ev_instance.step(a)
            reward_list.append(r)
            s = s_
            info = info_
            action_list.append(a)
        (
            portfit_magine,
            final_balance,
            required_money,
            commission_fee,
        ) = self.test_ev_instance.get_final_return_rate(slient=True)

        action_list = np.array(action_list)
        reward_list = np.array(reward_list)
        np.save(os.path.join(epoch_path, "action.npy"), action_list)
        np.save(os.path.join(epoch_path, "reward.npy"), reward_list)
        np.save(
            os.path.join(epoch_path, "final_balance.npy"),
            self.test_ev_instance.final_balance,
        )
        np.save(
            os.path.join(epoch_path, "pure_balance.npy"),
            self.test_ev_instance.pured_balance,
        )
        np.save(
            os.path.join(epoch_path, "require_money.npy"),
            self.test_ev_instance.required_money,
        )
        np.save(
            os.path.join(epoch_path, "commission_fee_history.npy"),
            self.test_ev_instance.comission_fee_history,
        )


if __name__ == "__main__":
    args = parser.parse_args()
    agent = DQN(args)
    agent.train()
