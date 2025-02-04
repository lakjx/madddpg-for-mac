import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from envs.macprotocol_mappo import MacProtocolEnv 
import os
import wandb
from collections import deque
# 改进的策略网络（支持RNN）
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, is_discrete=True, use_rnn=False):
        super().__init__()
        self.is_discrete = is_discrete
        self.use_rnn = use_rnn
        
        # 基础特征提取层
        self.base_net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.Tanh()
        )
        
        # RNN层（可选）
        if use_rnn:
            self.rnn = nn.GRU(128, 128, batch_first=True)
        
        # 动作输出层
        if is_discrete:
            self.action_head = nn.Sequential(
                nn.Linear(128, action_dim),
                nn.Softmax(dim=-1)
            )
        else:
            self.action_mean = nn.Linear(128, action_dim)
            self.action_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        seq_len = 1 if x.dim() == 2 else x.size(1)
        
        x = self.base_net(x)
        
        if self.use_rnn:
            x, hidden = self.rnn(x.view(batch_size, seq_len, -1), hidden)
            x = x.contiguous().view(batch_size * seq_len, -1)
        
        if self.is_discrete:
            probs = self.action_head(x)
            return torch.distributions.Categorical(probs), hidden
        else:
            mean = self.action_mean(x)
            logstd = self.action_logstd.expand_as(mean)
            return torch.distributions.Normal(mean, torch.exp(logstd)), hidden

# 改进的价值网络（支持RNN）
class Critic(nn.Module):
    def __init__(self, obs_dim, use_rnn=False):
        super().__init__()
        self.use_rnn = use_rnn
        
        self.base_net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.Tanh()
        )
        
        if use_rnn:
            self.rnn = nn.GRU(128, 128, batch_first=True)
            
        self.value_head = nn.Linear(128, 1)

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        seq_len = 1 if x.dim() == 2 else x.size(1)
        
        x = self.base_net(x)
        
        if self.use_rnn:
            x, hidden = self.rnn(x.view(batch_size, seq_len, -1), hidden)
            x = x.contiguous().view(batch_size * seq_len, -1)
            
        return self.value_head(x).squeeze(-1), hidden
    
# 经验回放数据集
class MARL_Dataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data['obs'])
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

class EnhancedMAPPO:
    def __init__(self,ppo_config, agent_configs, run_name,env):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agents = {}
        self.config = ppo_config
        
        # 初始化每个智能体
        for agent_id, config in agent_configs.items():
            agent = {}
            # 网络初始化
            agent['actor'] = Actor(
                config['obs_dim'], 
                config['action_dim'],
                config['is_discrete'],
                self.config['use_rnn']
            ).to(self.device)
            agent['critic'] = Critic(
                config['obs_dim'],
                self.config['use_rnn']
            ).to(self.device)
            agent['target_critic'] = Critic(
                config['obs_dim'],
                self.config['use_rnn']
            ).to(self.device)
            agent['target_critic'].load_state_dict(agent['critic'].state_dict())

            # 优化器配置
            agent['actor_optim'] = optim.AdamW(
                agent['actor'].parameters(), 
                lr=self.config['lr'],
                eps=1e-5
            )
            agent['critic_optim'] = optim.AdamW(
                agent['critic'].parameters(),
                lr=self.config['lr'],
                eps=1e-5
            )
            
            # 学习率调度器
            if self.config['lr_decay']:
                agent['actor_scheduler'] = optim.lr_scheduler.LambdaLR(
                    agent['actor_optim'],
                    lr_lambda=lambda epoch: 1 - epoch / 1000
                )
                agent['critic_scheduler'] = optim.lr_scheduler.LambdaLR(
                    agent['critic_optim'],
                    lr_lambda=lambda epoch: 1 - epoch / 1000
                )
            
            # 经验缓冲区
            agent['buffer'] = {
                'obs': deque(maxlen=self.config['buffer_size']),
                'actions': deque(maxlen=self.config['buffer_size']),
                'log_probs': deque(maxlen=self.config['buffer_size']),
                'rewards': deque(maxlen=self.config['buffer_size']),
                'next_obs': deque(maxlen=self.config['buffer_size']),
                'dones': deque(maxlen=self.config['buffer_size'])
            }
            self.agents[agent_id] = agent
        
        # 新增Wandb初始化
        wandb_dir = f"logs/wandb/{run_name}"
        os.makedirs(wandb_dir, exist_ok=True)
        wandb.init(project="mappo", 
                 name=run_name,
                 config={
                     "gamma": self.config['gamma'],
                     "gae_lambda": self.config['gae_lambda'],
                     "entropy_coef": self.config['entropy_coef'],
                     "learning_rate": self.config['lr'],
                     "batch_size": self.config['batch_size']
                 },
                 dir=wandb_dir
                 )
        
        # 创建模型保存目录
        self.save_dir = f"logs/checkpoints/{run_name}"
        os.makedirs(self.save_dir, exist_ok=True)

    # 新增模型保存方法
    def save_models(self, episode):
        save_path = f"{self.save_dir}/epoch_{episode}"
        os.makedirs(save_path, exist_ok=True)  # 创建目录
        for agent_id in self.agents:
            agent = self.agents[agent_id]
            torch.save({
                'actor': agent['actor'].state_dict(),
                'critic': agent['critic'].state_dict(),
                'actor_optim': agent['actor_optim'].state_dict(),
                'critic_optim': agent['critic_optim'].state_dict(),
            }, f"{save_path}/{agent_id}.pt")
            
    def load_models(self, loding_path):
        for agent_id in self.agents:
            agent = self.agents[agent_id]
            checkpoint = torch.load(f"{loding_path}/{agent_id}.pt")
            agent['actor'].load_state_dict(checkpoint['actor'])
            agent['critic'].load_state_dict(checkpoint['critic'])
            agent['actor_optim'].load_state_dict(checkpoint['actor_optim'])
            agent['critic_optim'].load_state_dict(checkpoint['critic_optim'])
        
    
    # 新增测试方法
    def test_agents(self, test_env, num_episodes=10, deterministic=True):
        test_rewards = {agent_id: [] for agent_id in self.agents}
        test_env.is_training = False
        goodputs,collision_rates,arriv_rates = [],[],[]
        for _ in range(num_episodes):
            obs = test_env.reset()
            episode_rewards = defaultdict(float)
            hidden_states = {agent_id: None for agent_id in self.agents}
            done = False
            step_count = 0
            
            while not done and step_count < test_env.TTLs:
                actions = {}
                new_hidden = {}
                
                for agent_id, obs in obs.items():
                    agent = self.agents[agent_id]
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        dist, new_h = agent['actor'](obs_tensor, hidden_states[agent_id])
                        
                        if deterministic:
                            if agent['actor'].is_discrete:
                                action = torch.argmax(dist.probs)
                            else:
                                action = dist.mean
                        else:
                            action = dist.sample()
                            
                    actions[agent_id] = action.squeeze(0).cpu().numpy()
                    new_hidden[agent_id] = new_h
                
                next_obs, rewards, dones, _ = test_env.step(actions)
                
                for agent_id, reward in rewards.items():
                    episode_rewards[agent_id] += reward
                
                obs = next_obs
                done = any(dones.values())
                step_count += 1
                hidden_states = new_hidden
            goodputs.append(test_env.get_Goodput())
            collision_rates.append(test_env.get_collision_rate())
            arriv_rates.append(test_env.get_packet_arrival_rate())
            for agent_id, reward in episode_rewards.items():
                test_rewards[agent_id].append(reward)
        
        _, tol_rwd = next(iter(test_rewards.items()))
        return np.mean(tol_rwd),np.mean(goodputs),np.mean(collision_rates),np.mean(arriv_rates)
    
    def act(self, obs_dict, hidden_states=None):
        actions = {}
        new_hidden = {}
        for agent_id, obs in obs_dict.items():
            agent = self.agents[agent_id]
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            # 处理RNN隐藏状态
            h = hidden_states[agent_id] if hidden_states else None
            
            with torch.no_grad():
                dist, new_h = agent['actor'](obs_tensor, h)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
            # 存储数据
            agent['buffer']['obs'].append(obs)
            agent['buffer']['actions'].append(action.squeeze(0).cpu().numpy())
            agent['buffer']['log_probs'].append(log_prob.squeeze(0))
            
            actions[agent_id] = action.squeeze(0).cpu().numpy()
            new_hidden[agent_id] = new_h
            
        return actions, new_hidden

    def store_experience(self, rewards_dict, next_obs_dict, dones_dict):
        for agent_id in self.agents:
            agent = self.agents[agent_id]
            buffer = agent['buffer']
            
            # 存储奖励和终止标志
            buffer['rewards'].append(rewards_dict[agent_id])
            buffer['next_obs'].append(next_obs_dict[agent_id])
            buffer['dones'].append(dones_dict[agent_id])

    # def compute_gae(self, agent):
    #     buffer = agent['buffer']
    #     # 处理跨episode的mask
    #     dones = np.array(buffer['dones'], dtype=bool)
    #     episode_ends = np.where(dones)[0] + 1
    #     # 分episode计算GAE
    #     start_idx = 0
    #     all_advantages = []
    #     all_returns = []
    #     for end_idx in episode_ends:
    #         episode_slice = slice(start_idx, end_idx)
    #         with torch.no_grad():
    #             # 将列表转换为单个 numpy.ndarray，然后再转换为张量
    #             obs = torch.FloatTensor(np.array(list(buffer['obs'])[episode_slice])).to(self.device)
    #             next_obs = torch.FloatTensor(np.array(list(buffer['next_obs'])[episode_slice])).to(self.device)
    #             rewards = torch.FloatTensor(list(buffer['rewards'])[episode_slice]).to(self.device)
    #             dones = torch.FloatTensor(list(buffer['dones'])[episode_slice]).to(self.device)
            
    #             # 计算价值估计
    #             values, _ = agent['critic'](obs)
    #             next_values, _ = agent['critic'](next_obs)
                
    #             # 多步TD目标计算
    #             deltas = rewards + self.config['gamma'] * next_values * (1 - dones) - values
    #             advantages = torch.zeros_like(rewards)
    #             last_advantage = 0
                
    #             # 逆向计算GAE
    #             for t in reversed(range(len(rewards))):
    #                 advantages[t] = deltas[t] + self.config['gamma'] * self.config['gae_lambda'] * (1 - dones[t]) * last_advantage
    #                 last_advantage = advantages[t]
                
    #             # 标准化优势函数
    #             if self.config['normalize_advantage']:
    #                 advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
    #             returns = advantages + values
    #             all_advantages.append(advantages)
    #             all_returns.append(returns)
    #         start_idx = end_idx
    #     advantages = torch.cat(all_advantages)
    #     returns = torch.cat(all_returns)
        
    #     return advantages, returns
    def compute_gae(self, agent):
        buffer = agent['buffer']
        dones = np.array(buffer['dones'], dtype=bool)
        episode_ends = np.where(dones)[0] + 1  # 找到每个episode结束的位置
        
        # 添加buffer结尾作为最后一个分割点
        all_segments = []
        start_idx = 0
        for end_idx in episode_ends:
            all_segments.append((start_idx, end_idx))
            start_idx = end_idx
        
        # 处理最后未完成的episode（如果有剩余数据）
        if start_idx < len(dones):
            all_segments.append((start_idx, len(dones)))
        
        all_advantages = []
        all_returns = []
        
        for (seg_start, seg_end) in all_segments:
            episode_slice = slice(seg_start, seg_end)
            
            with torch.no_grad():
                # 确保数据对齐
                obs = torch.FloatTensor(np.array(list(buffer['obs'])[episode_slice])).to(self.device)
                next_obs = torch.FloatTensor(np.array(list(buffer['next_obs'])[episode_slice])).to(self.device)
                rewards = torch.FloatTensor(list(buffer['rewards'])[episode_slice]).to(self.device)
                dones = torch.FloatTensor(list(buffer['dones'])[episode_slice]).to(self.device)
                # 计算价值估计
                values = agent['critic'](obs)[0]
                next_values = agent['target_critic'](next_obs)[0]
                
                # 多步TD目标计算
                deltas = rewards + self.config['gamma'] * next_values * (1 - dones) - values
                advantages = torch.zeros_like(rewards)
                last_advantage = 0
                
                # 逆向计算GAE
                for t in reversed(range(len(rewards))):
                    advantages[t] = deltas[t] + self.config['gamma'] * self.config['gae_lambda'] * (1 - dones[t]) * last_advantage
                    last_advantage = advantages[t]
                
                # 标准化优势函数（只在当前episode内标准化）
                if self.config['normalize_advantage'] and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                returns = advantages + values
                
                all_advantages.append(advantages)
                all_returns.append(returns)
        
        # 合并所有结果
        final_advantages = torch.cat(all_advantages)
        final_returns = torch.cat(all_returns)
        
        # 验证数据对齐
        assert len(final_advantages) == len(buffer['obs']), \
            f"GAE计算后数据长度不匹配!输入{len(buffer['obs'])}条，输出{len(final_advantages)}条"
        
        return final_advantages, final_returns
    def update_agent(self, agent, advantages, returns):
        dataset = MARL_Dataset({
        'obs': np.array(agent['buffer']['obs']),
        'actions': np.array(agent['buffer']['actions']),
        'log_probs': torch.stack(list(agent['buffer']['log_probs'])).cpu().numpy(),
        'advantages': advantages.cpu().numpy(),
        'returns': returns.cpu().numpy()
        })
        
        loader = DataLoader(dataset, 
                          batch_size=self.config['batch_size'],
                          shuffle=True,
                          pin_memory=True)
        actor_losses,critic_losses,entropys = [],[],[]
        for _ in range(self.config['epochs']):
            for batch in loader:
                obs = torch.FloatTensor(np.array(batch['obs'], dtype=np.float32)).to(self.device)
                actions = torch.FloatTensor(np.array(batch['actions'], dtype=np.float32)).to(self.device)
                old_log_probs = torch.FloatTensor(np.array(batch['log_probs'], dtype=np.float32)).to(self.device)
                advantages = torch.FloatTensor(np.array(batch['advantages'], dtype=np.float32)).to(self.device)
                returns = torch.FloatTensor(np.array(batch['returns'], dtype=np.float32)).to(self.device)
            
                # 计算新策略的概率
                dist, _ = agent['actor'](obs)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                
                # PPO损失计算（包含clip和熵正则）
                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.config['clip_epsilon'], 1 + self.config['clip_epsilon']) * advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.config['entropy_coef'] * entropy
                
                # 价值函数损失（带clip）
                values, _ = agent['critic'](obs)
                values_clipped = values + (values - values.detach()).clamp(-0.5, 0.5)
                critic_loss1 = (returns - values).pow(2)
                critic_loss2 = (returns - values_clipped).pow(2)
                critic_loss = 0.5 * torch.max(critic_loss1, critic_loss2).mean()
                
                # 梯度裁剪
                agent['actor_optim'].zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(agent['actor'].parameters(), self.config['max_grad_norm'])
                agent['actor_optim'].step()
                
                agent['critic_optim'].zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(agent['critic'].parameters(), self.config['max_grad_norm'])
                agent['critic_optim'].step()
                
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropys.append(entropy.item())
                
        # 更新学习率
        if self.config['lr_decay']:
            agent['actor_scheduler'].step()
            agent['critic_scheduler'].step()
        
        return np.mean(actor_losses), np.mean(critic_losses) ,np.mean(entropys)
    
    def update(self, episode):   
        agent_actor_losses,agent_critic_losses,agent_rewards = {},{},[]
        for agent_id in self.agents:
            agent = self.agents[agent_id]
            advantages, returns = self.compute_gae(agent)
            a_loss, c_loss, entr =self.update_agent(agent, advantages, returns)
            agent_actor_losses[agent_id] = a_loss
            agent_critic_losses[agent_id] = c_loss
            agent_rewards.append(sum(agent['buffer']['rewards']) / len(agent['buffer']['rewards']))  # 计算平均奖励

        # # 训练指标记录
        wandb.log({
                "train_reward": np.mean(agent_rewards),
                **{f"{agent_id}_actor_loss": loss for agent_id, loss in agent_actor_losses.items()},
                **{f"{agent_id}_critic_loss": loss for agent_id, loss in agent_critic_losses.items()}
                },step=episode)
        # 定期更新目标网络
        if episode % self.config['target_update_interval'] == 0:
            self.update_target_networks()
        # 清空buffer
        self._clear_buffers()

    def collect_episode_data(self):
        # 持续与环境交互直到buffer填满
        obs = self.env.reset()
        hidden_states = {agent_id: None for agent_id in self.agents}
        
        while not self._buffers_ready():
            actions, hidden_states = self.act(obs, hidden_states)
            next_obs, rewards, dones, _ = self.env.step(actions)
            self.store_experience(rewards, next_obs, dones)
            obs = next_obs
            
            if any(dones.values()):  # 处理环境提前终止
                obs = self.env.reset()
                hidden_states = {agent_id: None for agent_id in self.agents}
    def update_target_networks(self):
        for agent in self.agents.values():
            agent['target_critic'].load_state_dict(agent['critic'].state_dict())

    def _buffers_ready(self):
        # 检查所有agent的buffer是否达到最小训练要求
        return all(len(agent['buffer']['obs']) >= self.config['min_buffer_size'] 
                 for agent in self.agents.values())

    def _clear_buffers(self):
        for agent in self.agents.values():
            agent['buffer']['obs'].clear()
            agent['buffer']['actions'].clear()
            agent['buffer']['log_probs'].clear()
            agent['buffer']['rewards'].clear()
            agent['buffer']['next_obs'].clear()
            agent['buffer']['dones'].clear()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rho', type=int, default=3)
    parser.add_argument('--recent_k', type=int, default=2)
    parser.add_argument('--UE_num', type=int, default=3)
    parser.add_argument('--UE_txbuff_len', type=int, default=20)
    parser.add_argument('--p_SDU_arrival', type=float, default=0.48)
    parser.add_argument('--tbl_error_rate', type=float, default=1e-3)
    parser.add_argument('--TTLs', type=int, default=24)
    parser.add_argument('--UCM', type=int, default=None)
    parser.add_argument('--DCM', type=int, default=None)
    parser.add_argument('--need_comm', type=bool, default=True)
    parser.add_argument('--exp_name', type=str, default='UE3_batch512')
    
    env_args = parser.parse_args()
    # 初始化配置
    agent_configs = {
        'agent_BS': {
            'obs_dim': (1+2*env_args.UE_num)*(env_args.recent_k+1),
            'action_dim': 3**env_args.UE_num,
            'is_discrete': True
            }
        }
    for id in range(env_args.UE_num):
        ue_id = 'agent_UE_'+str(id)
        agent_configs[ue_id] = {
            'obs_dim': 4*(env_args.recent_k+1),
            'action_dim': 6,
            'is_discrete': True
        }
    ppo_cfig = {
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.1,
            'entropy_coef': 0.01,
            'value_coef': 0.5,
            'max_grad_norm': 0.5,
            'epochs': 4,
            'batch_size': 512,
            'lr': 3e-5,
            'lr_decay': True,
            'use_rnn': False,
            'normalize_advantage': True,
            'target_update_interval': 5,
            'buffer_size': 10000,    # 每个agent的buffer容量（时间步数）
            'min_buffer_size': 2048
        }
    # 初始化训练和测试环境
    train_env = MacProtocolEnv(env_args)
    test_env = MacProtocolEnv(env_args)
    
    # 带实验命名的初始化
    mappo = EnhancedMAPPO(ppo_cfig, agent_configs, run_name=env_args.exp_name, env=train_env)
    
    # 训练参数
    total_epochs = 20000  
    test_interval = 5  
    save_interval = 80
    
    step_count = 0
    while step_count < total_epochs:
        # 收集数据直到buffer填满
        mappo.collect_episode_data()
        
        # 更新步数计数
        step_count += 1
        
        # 执行训练
        mappo.update(step_count)
        
        # 定期测试和保存
        if step_count % test_interval == 0:
            test_rwd,goodput,colli_rate,arriv_rate = mappo.test_agents(test_env)
            wandb.log({
                        "test_reward": test_rwd,
                        "goodput": goodput,
                        "collision_rate": colli_rate,
                        "arrival_rate": arriv_rate
                       },step=step_count)
            
        if step_count % save_interval == 0:
            mappo.save_models(step_count)
    
    wandb.finish()