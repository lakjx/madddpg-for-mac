import argparse
from html import parser
import os
import itertools
import copy
import numpy as np
import torch
import gym
from gym import spaces

# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# sys.path.append(parent_dir)
class DotDic(dict):
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

	def __deepcopy__(self, memo=None):
		return DotDic(copy.deepcopy(dict(self), memo=memo))
class MultiDiscrete(gym.Space):
    """
    - The multi-discrete action space consists of a series of discrete action spaces with different parameters
    - It can be adapted to both a Discrete action space or a continuous (Box) action space
    - It is useful to represent game controllers or keyboards where each key can be represented as a discrete action space
    - It is parametrized by passing an array of arrays containing [min, max] for each discrete action space
       where the discrete action space can take any integers from `min` to `max` (both inclusive)
    Note: A value of 0 always need to represent the NOOP action.
    e.g. Nintendo Game Controller
    - Can be conceptualized as 3 discrete action spaces:
        1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
        2) Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
        3) Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
    - Can be initialized as
        MultiDiscrete([ [0,4], [0,1], [0,1] ])
    """

    def __init__(self, array_of_param_array):
        self.low = np.array([x[0] for x in array_of_param_array])
        self.high = np.array([x[1] for x in array_of_param_array])
        self.num_discrete_space = self.low.shape[0]

    def sample(self):
        """ Returns a array with one sample from each discrete action space """
        # For each row: round(random .* (max - min) + min, 0)
        #random_array = prng.np_random.rand(self.num_discrete_space)
        random_array = np.random.rand(self.num_discrete_space)

        return [int(x) for x in np.floor(np.multiply((self.high - self.low + 1.), random_array) + self.low)]

    def contains(self, x):
        return len(x) == self.num_discrete_space and (np.array(x) >= self.low).all() and (np.array(x) <= self.high).all()

    @property
    def shape(self):
        return self.num_discrete_space

    def __repr__(self):
        return "MultiDiscrete" + str(self.num_discrete_space)

    def __eq__(self, other):
        return np.array_equal(self.low, other.low) and np.array_equal(self.high, other.high)

class MacProtocolEnv():
    def __init__(self, args, discrete_action=True):
        self.args = args
        self.is_training = True
        self.rho = self.args.rho
        self.UE_num = self.args.UE_num
        self.p_SDU_arrival = self.args.p_SDU_arrival
        self.tbl_error_rate = self.args.tbl_error_rate
        self.TTLs = self.args.TTLs  # Max. duration of episode
        self.recent_k = self.args.recent_k
        self.collision_count = 0
        self.gen_data_count = 0
        self.UE_act_space = DotDic({
            'Do Nothing': 0,
            'Transmit': 1,
            'Delete': 2
        })
        # UE_obs \in [0,|B|]
        self.UE_obs_space = spaces.Discrete(self.args.UE_txbuff_len + 1)
        # BS_obs \in [0,|U|+1]
        self.BS_obs_space = spaces.Discrete(self.UE_num + 2)

        self.BS_msg_space = DotDic({
            'Null': 0,
            'SG': 1,
            'ACK': 2
        })
        self.BS_msg_total_space = list(itertools.product(range(len(self.BS_msg_space)), repeat=self.UE_num))
        self.UE_msg_space = DotDic({
            'Null': 0,
            'SR': 1
        })

        self.agents = ['UE_' + str(i) for i in range(self.UE_num)] + ['BS']
        self.num_agents = len(self.agents)
        self.discrete_action_space = discrete_action
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0
        for agent in self.agents:
            total_action_space = []
            #physical action
            u_action_space = spaces.Discrete(len(self.UE_act_space))
            if agent != 'BS':
                total_action_space.append(u_action_space)
            # elif agent == 'BS' and self.args.need_comm == False:
            #     total_action_space.append(spaces.Discrete(2))
            #communication action
            if self.args.need_comm:
                if agent != 'BS':
                    c_action_space = spaces.Discrete(len(self.UE_msg_space))
                    total_action_space.append(c_action_space)
                else:
                    c_action_space = spaces.Discrete(len(self.BS_msg_space)**self.UE_num)
                    total_action_space.append(c_action_space)
            
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    action_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    raise NotImplementedError
                self.action_space.append(action_space)
            elif len(total_action_space) == 1:
                self.action_space.append(total_action_space[0])
            else:
                self.action_space.append([])
            # observation space
            if agent != 'BS':
                obs_dim = 4*(self.recent_k+1) if self.args.need_comm else 2*(self.recent_k+1)
                self.observation_space.append(spaces.Discrete(obs_dim))
            else:
                obs_dim = self.recent_k+1 + self.UE_num*2*(self.recent_k+1) if self.args.need_comm else self.recent_k+1
                self.observation_space.append(spaces.Discrete(obs_dim))
            share_obs_dim += obs_dim
        self.share_observation_space = [spaces.Discrete(share_obs_dim)] * self.num_agents

        self.reset()

    def reset(self):
        self.step_count = 0
        self.collision_count = 0
        self.gen_data_count = 0

        self.UEs = [UE(i,self.args) for i in range(self.UE_num)]

        # # self.UE_SDU_Generate()
        self.UE_obs = np.zeros((self.UE_num,), dtype=np.int32)
        self.UE_actions = np.zeros((self.UE_num,), dtype=np.int32)
        self.BS_obs = np.zeros((1,), dtype=np.int32)
        self.BS_msg = np.zeros((self.UE_num,), dtype=np.int32)
        self.UE_msg = np.zeros((self.UE_num,), dtype=np.int32)

        self.trajact_UE_obs = [copy.deepcopy(self.UE_obs) for _ in range(self.recent_k + 1)]
        self.trajact_UE_actions = [copy.deepcopy(self.UE_actions) for _ in range(self.recent_k + 1)]
        self.trajact_BS_obs = [copy.deepcopy(self.BS_obs) for _ in range(self.recent_k + 1)]
        self.trajact_BS_msg = [copy.deepcopy(self.BS_msg) for _ in range(self.recent_k + 1)]
        self.trajact_UE_msg = [copy.deepcopy(self.UE_msg) for _ in range(self.recent_k + 1)]


        
        self.sdus_received = []
        self.data_channel = []
        self.rewards = 0
        self.done = False
        # record observations for each agent
        obs_n = []
        for agent in self.agents:
            if agent == 'BS':
                obs_n.append(self.get_bs_internal_stat())
            else:
                tar_ue_idx = int(agent.split('_')[1])
                obs_n.append(self.get_ue_internal_stat(tar_ue_idx))
        return obs_n

    def step(self, action_n,UCM=None,DCM=None):
        UE_actions = [act[0] for (i, act) in enumerate(action_n) if self.agents[i] != 'BS']
        UCM = [act[1] for (i, act) in enumerate(action_n) if self.agents[i] != 'BS'] if self.args.need_comm else None
        DCM = self.BS_msg_total_space[action_n[-1][0]] if self.args.need_comm else None
        # UE_actions = action_n
        #测试状态下，打印每个UE的buffer状态
        if not self.is_training:
            for UE in self.UEs:
                print(UE.name,UE.buff)         

        if isinstance(UE_actions, list):
            UE_actions = np.array(UE_actions)
        elif isinstance(UE_actions, torch.Tensor):
            UE_actions = UE_actions.cpu().numpy()
        
        #随机生成UE的SDU
        new_data_list = self.UE_SDU_Generate()
        print('new_data_list:',new_data_list) if not self.is_training else None

        self.UE_actions = UE_actions
        self.UE_Signaling_policy(np.array(UCM)) if UCM is not None else self.UE_Signaling_policy()
        self.BS_Signaling_policy(np.array(DCM)) if DCM is not None else self.BS_Signaling_policy()
        error_del = False
        self.data_channel = []
        for UE in self.UEs:
            if len(UE.buff) > 0:
                if UE_actions[UE.name_id] == self.UE_act_space.Transmit and UE.buff[0] != new_data_list[UE.name_id]:
                    data = UE.transmit_SDU()
                    self.data_channel.append(data)
                
                elif UE_actions[UE.name_id] == self.UE_act_space.Delete and UE.buff[0] != new_data_list[UE.name_id]:
                    del_data = UE.delete_SDU()
                    if del_data not in self.sdus_received:
                        error_del = True
            else:
                pass
        self.check_channel(error_del)                 
    
        self.trajact_UE_obs.append(copy.deepcopy(np.array([UE.get_obs() for UE in self.UEs])))
        self.trajact_UE_actions.append(copy.deepcopy(self.UE_actions))
        self.trajact_BS_obs.append(copy.deepcopy(self.BS_obs) if isinstance(self.BS_obs, np.ndarray) else np.array([self.BS_obs]))
        self.trajact_BS_msg.append(copy.deepcopy(self.BS_msg))
        self.trajact_UE_msg.append(copy.deepcopy(self.UE_msg))
        if len(self.trajact_UE_obs) > self.recent_k+1:
            self.trajact_UE_obs.pop(0)
            self.trajact_UE_actions.pop(0)
            self.trajact_BS_obs.pop(0)
            self.trajact_BS_msg.pop(0)
            self.trajact_UE_msg.pop(0)

        self.done = self.step_count >= self.TTLs
        self.step_count += 1
        if not self.is_training:
            print('step:',self.step_count,'UE_act:',UE_actions,'datachannel:',self.data_channel,'rewards:',self.rewards)
            print('BS_recieved:',self.sdus_received)
        
        obs_n, reward_n, done_n, info_n = [], [], [], []
        for agent in self.agents:
            if agent == 'BS':
                obs_n.append(self.get_bs_internal_stat())
            else:
                tar_ue_idx = int(agent.split('_')[1])
                obs_n.append(self.get_ue_internal_stat(tar_ue_idx))
            reward_n.append(self.get_rwd())
            done_n.append(self.done)
            info_n.append({})

        return obs_n, reward_n, done_n, info_n

            
    def get_ue_internal_stat(self,tar_ue_idx):
        # x =(o,a,n,m) o: UE_obs, a: UE_actions, n: UE_msg, m: BS_msg
        # o = np.transpose([UE.get_obs() for UE in self.UEs])
        # o = UE.get_obs()
        # a = self.UE_actions[UE.name_id]
        # n = self.UE_msg[UE.name_id]
        # m = self.BS_msg[UE.name_id]
        # return np.array([o,a,n,m])
        if len(self.trajact_UE_obs) < self.recent_k + 1:
            # 填充缺失的轨迹数据，使用最近的观测值
            gap = self.recent_k + 1 - len(self.trajact_UE_obs)
            o = [self.trajact_UE_obs[0][tar_ue_idx]] * gap + [self.trajact_UE_obs[i][tar_ue_idx] for i in range(len(self.trajact_UE_obs))]
            a = [self.trajact_UE_actions[0][tar_ue_idx]] * gap + [self.trajact_UE_actions[i][tar_ue_idx] for i in range(len(self.trajact_UE_actions))]
            n = [self.trajact_UE_msg[0][tar_ue_idx]] * gap + [self.trajact_UE_msg[i][tar_ue_idx] for i in range(len(self.trajact_UE_msg))]
            m = [self.trajact_BS_msg[0][tar_ue_idx]] * gap + [self.trajact_BS_msg[i][tar_ue_idx] for i in range(len(self.trajact_BS_msg))]
        else:
            o = [self.trajact_UE_obs[i][tar_ue_idx] for i in range(self.recent_k + 1)]
            a = [self.trajact_UE_actions[i][tar_ue_idx] for i in range(self.recent_k + 1)]
            n = [self.trajact_UE_msg[i][tar_ue_idx] for i in range(self.recent_k + 1)]
            m = [self.trajact_BS_msg[i][tar_ue_idx] for i in range(self.recent_k + 1)]

        if self.args.need_comm:
            return np.concatenate((o, a, n, m), axis=0).flatten()
        else:
            return np.concatenate((o, a), axis=0).flatten()
            
            

    def get_bs_internal_stat(self):
        #x=(o_b,n_all,m_all)
        #检查BS_obs的数据类型
        assert self.BS_obs_space.contains(self.BS_obs[0] if isinstance(self.BS_obs,np.ndarray) else self.BS_obs)
        
        if len(self.trajact_BS_obs) < self.recent_k + 1:
            # 填充缺失的轨迹数据，使用最近的观测值
            gap = self.recent_k + 1 - len(self.trajact_BS_obs)
            self.trajact_BS_obs = [self.trajact_BS_obs[0]] * gap + self.trajact_BS_obs
            self.trajact_BS_msg = [self.trajact_BS_msg[0]] * gap + self.trajact_BS_msg
            self.trajact_UE_msg = [self.trajact_UE_msg[0]] * gap + self.trajact_UE_msg

        if self.args.need_comm:
            return np.concatenate((self.trajact_BS_obs, self.trajact_UE_msg, self.trajact_BS_msg), axis=1).flatten()
        else:
            return np.array(self.trajact_BS_obs).flatten()
    
    def get_rwd(self):
        return self.rewards
    
    def check_channel(self,error_del):
        # rho = 3
        #查看data通道上是否有冲突  e.g. data_channel = ['UE0_1', 'UE1_0', 'UE2_1']
        if len(self.data_channel) == 1: # 正常数传
            data = self.data_channel[0]
            self.BS_obs = int(data.split('_')[0][2:])+1
            if np.random.rand() > self.tbl_error_rate: #正确接收
                if data not in self.sdus_received:
                    self.sdus_received.append(data)
                    self.rewards = 2*self.rho
                else:
                    self.rewards = -1
        elif self.data_channel == []: # 空闲
            self.BS_obs = 0
            self.rewards = -1
        else:
            self.collision_count += 1
            self.BS_obs = self.UE_num + 1
            self.rewards = -1
        if error_del:
            self.rewards = self.rewards-2*self.rho  
        #判断BS_obs是否合法
        assert self.BS_obs_space.contains(self.BS_obs)
                                   
    def UE_SDU_Generate(self):
        gen_data_list = []
        for UE in self.UEs:
            if np.random.rand() < self.p_SDU_arrival:
                cur_gen_data = UE.generate_SDU()
                self.gen_data_count += 1
            else:
                cur_gen_data = None
            gen_data_list.append(cur_gen_data)
        return gen_data_list

    def BS_Signaling_policy(self,DCM=None):
        # BS can send one control message to each UE
        if DCM is None:
            DCM = np.random.randint(0, len(self.BS_msg_space), self.UE_num)
            self.BS_msg = DCM
        else:
            self.BS_msg = DCM
    def UE_Signaling_policy(self,UCM=None):
        # each UE can send one control message to BS
        if UCM is None:
            UCM = np.random.randint(0, len(self.UE_msg_space), self.UE_num)
            self.UE_msg = UCM
        else:
            self.UE_msg = UCM
    
    def get_Goodput(self):
        if self.step_count == 0:
            raise ValueError('step_count is 0!')
        return len(self.sdus_received)/self.step_count
    def get_collision_rate(self):
        return self.collision_count / self.step_count
    def get_buffer_occupancy(self):
        return [UE.get_obs()/UE.buff_size for UE in self.UEs]
    def get_packet_arrival_rate(self):
        return len(self.sdus_received)/self.gen_data_count
    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

class UE():
    def __init__(self, name_id , args):
        self.name_id = name_id
        self.name = 'UE' + str(name_id)
        self.buff_size = args.UE_txbuff_len
        self.buff = []
        self.datacount = 0
        self.SG = False
        self.ACK = False

    def generate_SDU(self):
        gen_data = None
        if len(self.buff) < self.buff_size:
            gen_data = self.name + '_' + str(self.datacount)
            self.buff.append(gen_data)
            self.datacount += 1
        return gen_data
    
    def delete_SDU(self):
        if len(self.buff) > 0:
            del_data = self.buff.pop(0)
            return del_data
        else:
            print('Delete_SDU error!'+ self.name + ' buffer is empty!')
            return None      
    
    def transmit_SDU(self):
        if len(self.buff) > 0:
            return self.buff[0]
        else:
            print('Transmit_SDU error!'+ self.name + ' buffer is empty!')
            return None
    
    def get_obs(self):
        return len(self.buff)
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rho', type=int, default=3)
    parser.add_argument('--recent_k', type=int, default=0)
    parser.add_argument('--UE_num', type=int, default=2)
    parser.add_argument('--UE_txbuff_len', type=int, default=20)
    # parser.add_argument('--UE_max_generate_SDUs', type=int, default=2)
    parser.add_argument('--p_SDU_arrival', type=float, default=0.48)
    parser.add_argument('--tbl_error_rate', type=float, default=1e-3)
    parser.add_argument('--TTLs', type=int, default=24)
    parser.add_argument('--UCM', type=int, default=None)
    parser.add_argument('--DCM', type=int, default=None)
    parser.add_argument('--need_comm', type=bool, default=False)
    args = parser.parse_args()
    env = MacProtocolEnv(args)
    env.is_training = False
    print("init obs:",env.reset())
    t=0
    while t<args.TTLs:
        UE_actions = np.random.randint(0, 3, env.UE_num)
        o,r,_,_ =env.step(UE_actions)
        t = t+1
        print("observation:{}".format(o))
        print("reward:{}".format(r))
    print('Goodput:',env.get_Goodput())
    print('collision rate:',env.get_collision_rate())
    print('packet arrival rate:',env.get_packet_arrival_rate())
        