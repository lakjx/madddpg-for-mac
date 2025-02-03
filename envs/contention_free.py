import numpy as np
import copy
from collections import deque
import json
import random
class UE:
    def __init__(self, buffer_size, id):
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.packets_dropped = 0
        self.generated_packets = 0
        self.packets_transmitted = 0
        self.waiting_for_ack = False
        self.has_sg = False
        
        self.id = id
        # 时序相关变量
        self.sg_processing_delay = 1  # SG处理延迟
        self.sg_receive_time = None   # SG接收时间
        self.transmission_delay = 1    # 数据传输延迟
        self.transmission_start_time = None  # 传输开始时间

        # 动作相关变量
        self.phy_act = None
        self.dcm_act = None
        self.ucm_act = None
        self.suceess = False
        
    def add_sdu(self):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(1)
            return True
        else:
            self.packets_dropped += 1
            return False
    
    def can_send_sr(self):
        return len(self.buffer) > 0 and not self.has_sg
    
    def receive_sg(self, current_time):
        self.has_sg = True
        self.sg_receive_time = current_time
    
    def can_transmit(self, current_time):
        if self.has_sg and self.sg_receive_time is not None:
            # 检查是否已经过了SG处理延迟
            return current_time >= self.sg_receive_time + self.sg_processing_delay
        return False
    
    def transmit_sdu(self, current_time):
        if len(self.buffer) > 0 and self.can_transmit(current_time):
            self.waiting_for_ack = True
            self.has_sg = False
            self.transmission_start_time = current_time
            return True
        return False
    
    def can_receive_ack(self, current_time):
        if self.waiting_for_ack and self.transmission_start_time is not None:
            # 检查是否已经过了传输延迟
            return current_time >= self.transmission_start_time + self.transmission_delay
        return False
    
    def receive_ack(self, current_time):
        if self.can_receive_ack(current_time):
            self.buffer.popleft()
            self.packets_transmitted += 1
            self.waiting_for_ack = False
            self.transmission_start_time = None
            return True
        return False

class BaseStation:
    def __init__(self, num_ues, buffer_size, arrival_prob):
        self.ues = [UE(buffer_size,ii) for ii in range(num_ues)]
        self.arrival_prob = arrival_prob
        self.sr_queue = set()
        self.current_transmitting = None
        self.sg_delay = 1  # SG发送延迟
        self.sg_send_times = {}  # 记录每个UE的SG发送时间

        self.num_UEs = num_ues
        self.recent_k = 2
        self.reset()

    def reset(self):
        self.UE_obs = np.zeros((self.num_UEs,), dtype=np.int32)
        self.BS_obs = np.zeros((1,), dtype=np.int32)
        self.UE_actions = np.array([None for _ in range(self.num_UEs)])
        self.BS_msg = np.array([None for _ in range(self.num_UEs)])
        self.UE_msg = np.array([None for _ in range(self.num_UEs)])

        self.trajact_UE_obs = [copy.deepcopy(self.UE_obs) for _ in range(self.recent_k + 1)]
        self.trajact_UE_actions = [copy.deepcopy(self.UE_actions) for _ in range(self.recent_k + 1)]
        self.trajact_BS_obs = [copy.deepcopy(self.BS_obs) for _ in range(self.recent_k + 1)]
        self.trajact_BS_msg = [copy.deepcopy(self.BS_msg) for _ in range(self.recent_k + 1)]
        self.trajact_UE_msg = [copy.deepcopy(self.UE_msg) for _ in range(self.recent_k + 1)] 


    def step(self, current_time):
        # 1. SDU到达过程
        # for ue in self.ues:
        #     ue.ucm_act = None
        #     ue.dcm_act = None
        #     ue.phy_act = None
        #     if np.random.random() < self.arrival_prob:
        #         ue.add_sdu()
        # self.UE_obs = np.array([len(ue.buffer) for ue in self.ues])
        print(f"\nStep {current_time}:")
        print(f"Buffer lengths: {self.UE_obs}",end=' ')
        # 2. 处理当前传输
        if self.current_transmitting is not None:
            # ue = self.ues[self.current_transmitting]
            ue = next(u for u in self.ues if u.id == self.current_transmitting)
            if ue.can_receive_ack(current_time):
                # if np.random.random() < 0.99:  # 传输成功率0.9
                if ue.suceess:
                    ue.receive_ack(current_time)
                    ue.phy_act = 'Delete'
                    ue.suceess = False
                    # ue.dcm_act = 'ACK'
                else:
                    ue.waiting_for_ack = False
                    ue.transmission_start_time = None
                self.current_transmitting = None
        # 3. 收集SR
        new_sr_queue = set()
        for i, ue in enumerate(self.ues):
            if ue.can_send_sr():
                new_sr_queue.add(i)
                ue.ucm_act = 'SR'
        self.sr_queue = new_sr_queue
        
        # 4. 发送调度许可（SG）
        if self.sr_queue:
            # 随机选择一个UE发送SG
            selected_ue = np.random.choice(list(self.sr_queue))
            # self.ues[selected_ue].dcm_act = 'SG' if self.ues[selected_ue].dcm_act == None else self.ues[selected_ue].dcm_act
            # 检查是否满足SG发送延迟要求
            if current_time >= self.sg_send_times.get(selected_ue, 0) + self.sg_delay and self.ues[selected_ue].dcm_act != 'ACK':
                self.ues[selected_ue].dcm_act = 'SG' if self.ues[selected_ue].dcm_act == None else self.ues[selected_ue].dcm_act
                self.ues[selected_ue].receive_sg(current_time)
                self.sg_send_times[selected_ue] = current_time
                self.sr_queue.remove(selected_ue)
        
        # 5. 数据传输
        if self.current_transmitting is None:  # 只有当前没有传输时才开始新的传输
            random.shuffle(self.ues)
            for ue in self.ues:
                if ue.transmit_sdu(current_time):
                    self.current_transmitting = ue.id
                    ue.phy_act = 'Transmit'
                    if np.random.random() < 0.99:  # 传输成功率0.9
                        ue.dcm_act = 'ACK'
                        ue.suceess = True
                    break
        print(f'UE pysical actions: {[ue.phy_act for ue in self.ues]}',end=' ')
        print(f'UCM actions: {[ue.ucm_act for ue in self.ues]}',end=' ')
        print(f'DCM actions: {[ue.dcm_act for ue in self.ues]}',end=' ')
        # 6. 记录轨迹
        #当前哪个UE在传输
        self.BS_obs = np.array([self.current_transmitting]) if self.current_transmitting is not None else np.array([-1])
        self.BS_msg = np.array([ue.dcm_act for ue in self.ues])
        self.UE_msg = np.array([ue.ucm_act for ue in self.ues])
        self.UE_actions = np.array([ue.phy_act for ue in self.ues])


    def get_statistics(self):
        total_transmitted = sum(ue.packets_transmitted for ue in self.ues)
        total_dropped = sum(ue.packets_dropped for ue in self.ues)
        buffer_occupancy = [len(ue.buffer) for ue in self.ues]
        sr_queue_size = len(self.sr_queue)
        return total_transmitted, total_dropped, buffer_occupancy, sr_queue_size
    def init_buffer(self,init_data_num = 0):
        for ue in self.ues:
            for _ in range(init_data_num):
                ue.add_sdu()
        self.UE_obs = np.array([len(ue.buffer) for ue in self.ues])
def run_simulation(num_ues, buffer_size, arrival_prob, num_steps):
    repititions = 1
    avg_goodput = 0

    for _ in range(repititions):
        bs = BaseStation(num_ues, buffer_size, arrival_prob)
        bs.init_buffer(init_data_num=5)
        for step in range(num_steps):
            # 1. SDU到达过程
            for ue in bs.ues:
                ue.ucm_act = None
                ue.dcm_act = None
                ue.phy_act = None
                if np.random.random() < 0.2:
                    ue.generated_packets += 1
                    ue.add_sdu()
            bs.UE_obs = np.array([len(ue.buffer) for ue in bs.ues])
            bs.step(step)

        transmitted, dropped, occupancy, sr_queue_size = bs.get_statistics()
        avg_goodput += transmitted / (num_steps+1)
    gen_data_num = [bs.ues[i].generated_packets for i in range(num_ues)]
    print(f"Generated data number: {gen_data_num}")
    print(f"Goodputs: {avg_goodput/repititions:.3f} packets/step")
    print(f"Buffer occupancy: {np.mean(occupancy):.3f}")
    # print(f"Packet arrival probability: {arrival_prob}")

def collect_datasets(ue_nums=[2,5,8],pa=[0.5,0.41,0.33,0.25,0.16,0.083],buffer_sizes=20,steps=24,episode=300):
    dataset = []
    for num_ues in ue_nums:
        for p in pa:
            for _ in range(episode):
                bs = BaseStation(num_ues, buffer_sizes, p)
                for step in range(steps):
                    for ue in bs.ues:
                        ue.ucm_act = None
                        ue.dcm_act = None
                        ue.phy_act = None
                        if np.random.random() < bs.arrival_prob:
                            ue.add_sdu()
                    bs.UE_obs = np.array([len(ue.buffer) for ue in bs.ues])
                    bs.trajact_UE_obs = bs.trajact_UE_obs[1:] + [copy.deepcopy(bs.UE_obs)]
                    #记录当前状态
                    current_state = {
                        "num_UEs": num_ues,
                        "time_step": step,
                        "ue_obs_history": [arr.tolist() for arr in bs.trajact_UE_obs[-bs.recent_k:]],
                        "ue_actions_history": [arr.tolist() for arr in bs.trajact_UE_actions[-bs.recent_k:]],
                        "bs_obs_history": [arr.tolist() for arr in bs.trajact_BS_obs[-bs.recent_k:]],
                        "bs_msg_history": [arr.tolist() for arr in bs.trajact_BS_msg[-bs.recent_k:]],
                        "ue_msg_history": [arr.tolist() for arr in bs.trajact_UE_msg[-bs.recent_k:]]
                    }
                    # 执行步骤
                    bs.step(step)
                    # 记录结果动作
                    result = {
                        'ue_actions': [ue.phy_act if ue.phy_act is not None else None for ue in bs.ues],
                        'ucm': [ue.ucm_act if ue.ucm_act is not None else None for ue in bs.ues],
                        'dcm': [ue.dcm_act if ue.dcm_act is not None else None for ue in bs.ues]
                    }
                    
                    # 将状态-动作对添加到数据集
                    dataset.append({
                        'state': current_state,
                        'action': result
                    })
                    
                    #更新轨迹信息
                    # bs.trajact_UE_obs = bs.trajact_UE_obs[1:] + [copy.deepcopy(bs.UE_obs)]
                    bs.trajact_UE_actions = bs.trajact_UE_actions[1:] + [copy.deepcopy(bs.UE_actions)]
                    bs.trajact_BS_obs = bs.trajact_BS_obs[1:] + [copy.deepcopy(bs.BS_obs)]
                    bs.trajact_BS_msg = bs.trajact_BS_msg[1:] + [copy.deepcopy(bs.BS_msg)]
                    bs.trajact_UE_msg = bs.trajact_UE_msg[1:] + [copy.deepcopy(bs.UE_msg)]
    return dataset

# 运行模拟
if __name__ == "__main__":
    # 设置参数
    L = 2       # UE数量
    B = 20      # 缓冲区大小
    T = 24     # 模拟步数
    pa = 0.48    # SDU到达概率

    
    run_simulation(L, B, pa, T)
    # dataset = collect_datasets()
    # with open('raw_datasets4.json', 'w') as f:
    #     json.dump(dataset, f, indent=4)

    #raw_dataset2 是UE=2,3,4,5,6
    #raw_dataset3 是UE=2,5
    #raw_dataset4 是UE=2,5,8 & p=0.5,0.41,0.33,0.25,0.16,0.083