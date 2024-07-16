import time
import numpy as np
from functools import reduce
import torch
from runners.separated.base_runner import Runner

def _t2n(x):
    return x.detach().cpu().numpy()

class SMACRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        super(SMACRunner, self).__init__(config)

    def run(self):
        self.warmup() # none

        start = time.time()

        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        # 6250 = 20000000 // 400 // 8
        # episode:常规含义  episode_length: 每一个episode的长度
        # n_rollout_threads： 并行env数量，并行越多，episodes越少，但length不变

        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)
        # [0. 0. 0. 0. 0. 0. 0. 0.]

        for episode in range(episodes):

            if self.use_linear_lr_decay:  # default=False 自适应线性学习率
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                """走完400个episode_length很好走，目的是收集experience
                    for循环后面的compute和train才是重头戏
                    collect中直接调用buffer，而bu在定义的时候已经是实例化（继承自Runner），那时初始化了全部401个step的内容
                    然后根据第1个step的内容，计算第2个step的内容并存储到buffer，第二次进入collect时，第2个step的内容由于在上一个
                    step中已经存储了，所以会直接读取第2个step的内容。
                    """
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)

                # make a 整合，方便放入buffer
                data = obs, share_obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic 
                
                # insert data into buffer
                self.insert(data)

            # compute return 计算每一个step的return （用GAE）（step实际上就相当于s_t时刻） 然后存入buffer[agent_id]
            self.compute()

            train_infos = self.train()
            
            # post process 后加工，总共走的步数
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads           
            # save model ---> 经过 svae_interval 个 episode 或 执行全部episode后 存储
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information 日志打印
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Map {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.map_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "StarCraft2":
                    battles_won = []
                    battles_game = []
                    incre_battles_won = []
                    incre_battles_game = []                    

                    for i, info in enumerate(infos):
                        if 'battles_won' in info[0].keys():
                            battles_won.append(info[0]['battles_won'])
                            incre_battles_won.append(info[0]['battles_won']-last_battles_won[i])
                        if 'battles_game' in info[0].keys():
                            battles_game.append(info[0]['battles_game'])
                            incre_battles_game.append(info[0]['battles_game']-last_battles_game[i])

                    incre_win_rate = np.sum(incre_battles_won)/np.sum(incre_battles_game) if np.sum(incre_battles_game)>0 else 0.0
                    print("incre win rate is {}.".format(incre_win_rate))
                    self.writter.add_scalars("incre_win_rate", {"incre_win_rate": incre_win_rate}, total_num_steps)
                    
                    last_battles_game = battles_game
                    last_battles_won = battles_won
                # modified

                for agent_id in range(self.num_agents):
                    train_infos[agent_id]['dead_ratio'] = 1 - self.buffer[agent_id].active_masks.sum() /(self.num_agents* reduce(lambda x, y: x*y, list(self.buffer[agent_id].active_masks.shape)))
                
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()
        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs
        for agent_id in range(self.num_agents):
            self.buffer[agent_id].share_obs[0] = share_obs[:,agent_id].copy()
            self.buffer[agent_id].obs[0] = obs[:,agent_id].copy()
            self.buffer[agent_id].available_actions[0] = available_actions[:,agent_id].copy()

    @torch.no_grad()
    def collect(self, step):
        value_collector=[]
        action_collector=[]
        action_log_prob_collector=[]
        rnn_state_collector=[]
        rnn_state_critic_collector=[]
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout() # 切换至评估模式
            """ ------输入
            以下代码：以self.buffer[agent_id].share_obs[step]为例:
            首先这一整句是送入.getactions的参数
            buffer[agent_id]存储的是第agent_id个buffer的实例化地址，它的实例化在里面弄好了不用管
            .share_obs[step] ---> 8x299 并行数量8，share_obs纬数299
            .share_obs里面是401个重复的8x299,纬数401x8x299，且是初始化的数据
            其余类似  mask--->401x8x1-->全1   available_actions--->401x8x14
            """
            """------输出
            value, action, action_log_prob---8x1
            这里的8x1是8个并行环境，1个agent。因为现在处于for agent_id 循环中，循环结束理论上才是8x8x1的纬度
            一个agent对应着1个动作，一个价值，一个log_prob，这里的变量都没加s,这个意思。
            """
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                self.buffer[agent_id].obs[step],
                                                self.buffer[agent_id].rnn_states[step],
                                                self.buffer[agent_id].rnn_states_critic[step],
                                                self.buffer[agent_id].masks[step],
                                                self.buffer[agent_id].available_actions[step])
            value_collector.append(_t2n(value))
            action_collector.append(_t2n(action))
            action_log_prob_collector.append(_t2n(action_log_prob))
            rnn_state_collector.append(_t2n(rnn_state))
            rnn_state_critic_collector.append(_t2n(rnn_state_critic))
        # [self.envs, agents, dim]
        values = np.array(value_collector).transpose(1, 0, 2)
        actions = np.array(action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
        rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_state_critic_collector).transpose(1, 0, 2, 3)
        """  上面5行代码的意思，以values为例
        将values转换成8x8x1---> values[0] = 同一环境中8个agent的value输出  np.array形式  values[0]-->8x1
        最开始的value (8x1)，表示的同一agent，8个环境下的value输出
        ？？======= 可也对应一个疑问，同一环境下的不同agent，其share_obs应该是一样的，为什么输出value不一样
        ------->>>>因为有bias的存在，critic设置了bias
        """

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data
        # done是8个并行env，8个agent是否done，当一个并行env下的所有agent全部done，该并行环境才done
        dones_env = np.all(dones, axis=1)
        """rnn_states---> 8x8x1x64 ---> 并行env x num_agent x recurrent_N x hidden_size
        (dones_env == True).sum(): dones_env中有几个True，结果就为几
        ===>>> 它的意思是说，把已经done的env，他所对应的数据全部置0。
        即，清空在已完成的环境中所有agent的RNN状态，不过本来就是全0，因为本人物用不到rnn_states和rnn_states_critic
        """
        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer[0].rnn_states_critic.shape[2:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # active_masks: 8x8x1 先全部赋值为1（相当于默认1），然后根据每个agent的done的情况，把done的agent处赋0
        # dones == True先个别赋值（8个并行env中的8个agent）
        # dones_env == True 某个并行env中agent全部done了，那就把这个并行env对应8个agent全部赋值
        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
        # bad_masks <---info
        bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])
        
        if not self.use_centralized_V:
            share_obs = obs
        for agent_id in range(self.num_agents):
            self.buffer[agent_id].insert(share_obs[:,agent_id], obs[:,agent_id], rnn_states[:,agent_id],
                    rnn_states_critic[:,agent_id],actions[:,agent_id], action_log_probs[:,agent_id],
                    values[:,agent_id], rewards[:,agent_id], masks[:,agent_id], bad_masks[:,agent_id], 
                    active_masks[:,agent_id], available_actions[:,agent_id])

    def log_train(self, train_infos, total_num_steps):
        for agent_id in range(self.num_agents):
            train_infos[agent_id]["average_step_rewards"] = np.mean(self.buffer[agent_id].rewards)
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)
    
    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_battles_won = 0
        eval_episode = 0
        eval_episode_rewards = []
        one_episode_rewards = []
        for eval_i in range(self.n_eval_rollout_threads):
            one_episode_rewards.append([])
            eval_episode_rewards.append([])

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            eval_actions_collector=[]
            eval_rnn_states_collector=[]
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_actions, temp_rnn_state = \
                    self.trainer[agent_id].policy.act(eval_obs[:,agent_id],
                                            eval_rnn_states[:,agent_id],
                                            eval_masks[:,agent_id],
                                            eval_available_actions[:,agent_id],
                                            deterministic=True)
                eval_rnn_states[:,agent_id]=_t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1,0,2)

            
            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
            for eval_i in range(self.n_eval_rollout_threads):
                one_episode_rewards[eval_i].append(eval_rewards[eval_i])

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards[eval_i].append(np.sum(one_episode_rewards[eval_i], axis=0))
                    one_episode_rewards[eval_i] = []
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.concatenate(eval_episode_rewards)
                eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards}                
                self.log_env(eval_env_infos, total_num_steps)
                eval_win_rate = eval_battles_won/eval_episode
                print("eval win rate is {}.".format(eval_win_rate))
                self.writter.add_scalars("eval_win_rate", {"eval_win_rate": eval_win_rate}, total_num_steps)
                break
