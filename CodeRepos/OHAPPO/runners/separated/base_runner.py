    
import time
import os
import numpy as np
from itertools import chain
import torch
from tensorboardX import SummaryWriter
from utils.separated_buffer import SeparatedReplayBuffer
from utils.util import update_linear_schedule

def _t2n(x):
    return x.detach().cpu().numpy()

class Runner(object):
    def __init__(self, config):

        self.all_args = config['all_args']  # 经过两次继承
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']

        # parameters 从all_args来，此过程也可以省略直接使用all_args.
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N
        self.use_single_network = self.all_args.use_single_network
        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_render:
            import imageio
            self.run_dir = config["run_dir"]
            self.gif_dir = str(self.run_dir / 'gifs')
            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

            # 虽然不知道这些路径干什么用
            # run_dir=/home/lyy/HAPPO-HATRPO/scripts/results/StarCraft2/3s5z/happo/mlp/1548/run1
            # log_dir=/home/lyy/HAPPO-HATRPO/scripts/results/StarCraft2/3s5z/happo/mlp/1900/run1/logs
            # save_dir=/home/lyy/HAPPO-HATRPO/scripts/results/StarCraft2/3s5z/happo/mlp/1900/run1/models


        if self.all_args.algorithm_name == "happo":
            from algorithms.happo_trainer import HAPPO as TrainAlgo
            from algorithms.happo_policy import HAPPO_Policy as Policy
        elif self.all_args.algorithm_name == "hatrpo":
            from algorithms.hatrpo_trainer import HATRPO as TrainAlgo
            from algorithms.hatrpo_policy import HATRPO_Policy as Policy
        else:
            raise NotImplementedError

        print("share_observation_space: ", self.envs.share_observation_space)
        print("observation_space: ", self.envs.observation_space)
        print("action_space: ", self.envs.action_space) # 每个agent有14个离散动作

        # share_observation_space: 使用中心化V时采用这个观测空间，全局观测空间？
        # observation_space: [252, [7, 22], [8, 8], [1, 4], [1, 30]] 共八组，8个agents
        # observation_space and action_space come from env, that we have not defined before.


        self.policy = []
        for agent_id in range(self.num_agents):
            # HAPPO self.use_centralized_V=true 采用全局状态空间
            share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]

            # policy network 策略网络实例化
            po = Policy(self.all_args,
                        self.envs.observation_space[agent_id],
                        share_observation_space,
                        self.envs.action_space[agent_id],
                        device = self.device)
            self.policy.append(po)
        # !!!:不知道为何，因为本条if只会执行一次
        # 是否：模型非空，直接接着上次的训练，加快训练速度。
        if self.model_dir is not None:
            self.restore()

        self.trainer = []  # 里面存的是地址
        self.buffer = []
        for agent_id in range(self.num_agents):
            # algorithm  ---tr 是一个实例化,bu同理，其地址存储在trainer中，可通过trainer调用
            # 如调用第一个agent的algorithm， trainer[agent_id]=trainer[0]
            tr = TrainAlgo(self.all_args, self.policy[agent_id], device = self.device)
            # buffer
            share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
            # 每一个agent有一个buffer，同时送入自身observation and share_observation
            # print("==")
            # print(self.envs.share_observation_space)
            # print("=-")
            # print(self.envs.observation_space)
            # input()

            bu = SeparatedReplayBuffer(self.all_args,
                                       self.envs.observation_space[agent_id],
                                       share_observation_space,
                                       self.envs.action_space[agent_id])
            self.buffer.append(bu)
            self.trainer.append(tr)

            
    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError
    
    @torch.no_grad()  # compute 中的数据不会计算梯度
    def compute(self):
        """compute 一次性处理全部step， 而不是某一个step  gae"""
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            # next_value是最后一个step的next_value，而前面所有step的value实际上已经计算过了
            # 只有最后一个step的next_value没有计算过，所以这里单独计算了一次（注意最后一个step的next_obs等也前面已经计算过）
            next_value = self.trainer[agent_id].policy.get_values(self.buffer[agent_id].share_obs[-1], 
                                                                self.buffer[agent_id].rnn_states_critic[-1],
                                                                self.buffer[agent_id].masks[-1])
            next_value = _t2n(next_value)
            # compute_returns: 会计算每一个step的return （用GAE）（step实际上就相当于s_t时刻）
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)

    def train(self):
        train_infos = []
        # random update order

        action_dim=self.buffer[0].actions.shape[-1]  # 1
        # 400x8x1 factor---指的什么
        factor = np.ones((self.episode_length, self.n_rollout_threads, action_dim), dtype=np.float32)

        for agent_id in torch.randperm(self.num_agents):
            self.trainer[agent_id].prep_training()  # 是准备开始训练，不是预训练   --- 转换到训练模式
            # self.factor = factor.copy() 更新separated_buffer里的self.factor 400X8X1
            self.buffer[agent_id].update_factor(factor)
            # available_actions 3200x14  3200=400x8 虽然不太明白， 但暂且可以跳过
            available_actions = None if self.buffer[agent_id].available_actions is None \
                else self.buffer[agent_id].available_actions[:-1].reshape(-1, *self.buffer[agent_id].available_actions.shape[2:])

            # 走else  shape[2:]表示选取第3个及其之后的维度，如401x8x252 , 则选取252
            # *self.buffer[agent_id].active_masks.shape[2:]  *表示解包
            # 如果前面.shape[2:]选取了252 1，那么把这两个数送入函数，当然下面代码只选取到252，所以加不加*都可以
            if self.all_args.algorithm_name == "hatrpo":
                old_actions_logprob, _, _, _, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            else:
                """里面的reshape是把维度 400x8x1 ---> 3200x1
                注意：这是在for agent_id里，400-step 8-并行env 1-agent ，每个agent的obs不一样，所以for agent_id
                因为train, 不需要管并行那个维度， 所有的并行环境都是用来训练的
                而且这里只是传递到evaluate_actions中的数据维度会变为3200x1 ，真正的obs等等不会改变，注意这里只是传递函数参数
                """
                old_actions_logprob, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            # 仍然在 for agent_id 内
            train_info = self.trainer[agent_id].train(self.buffer[agent_id])

            # 此时策略已经发生了更新，计算新的new_actions_logprob 其实是为了更新factor
            """上面的old_actions_logprob和下面的new_actions_logprob是策略更新前后的同一动作的log_prob，注意，策略已经更新"""
            if self.all_args.algorithm_name == "hatrpo":
                new_actions_logprob, _, _, _, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            else:
                new_actions_logprob, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))

            factor = factor*_t2n(torch.exp(new_actions_logprob-old_actions_logprob).reshape(self.episode_length,self.n_rollout_threads,action_dim))
            train_infos.append(train_info)      
            self.buffer[agent_id].after_update()

        return train_infos

    def save(self):
        for agent_id in range(self.num_agents):
            if self.use_single_network:
                policy_model = self.trainer[agent_id].policy.model
                torch.save(policy_model.state_dict(), str(self.save_dir) + "/model_agent" + str(agent_id) + ".pt")
            else:
                policy_actor = self.trainer[agent_id].policy.actor
                torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt")
                policy_critic = self.trainer[agent_id].policy.critic
                torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + str(agent_id) + ".pt")

    def restore(self):
        for agent_id in range(self.num_agents):
            if self.use_single_network:
                policy_model_state_dict = torch.load(str(self.model_dir) + '/model_agent' + str(agent_id) + '.pt')
                self.policy[agent_id].model.load_state_dict(policy_model_state_dict)
            else:
                policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + str(agent_id) + '.pt')
                self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
                policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_agent' + str(agent_id) + '.pt')
                self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)

    def log_train(self, train_infos, total_num_steps): 
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
