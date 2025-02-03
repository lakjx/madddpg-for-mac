import numpy as np
from collections import deque

class UE:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.packets_dropped = 0
        self.packets_transmitted = 0
        self.waiting_for_ack = False
        
        # 时序相关变量
        self.transmission_delay = 2    # 数据传输延迟
        self.transmission_start_time = None  # 传输开始时间
        
    def add_sdu(self):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(1)
            return True
        else:
            self.packets_dropped += 1
            return False
    
    def can_transmit(self):
        return len(self.buffer) > 0 and not self.waiting_for_ack
    
    def transmit_sdu(self, current_time):
        if self.can_transmit():
            self.waiting_for_ack = True
            self.transmission_start_time = current_time
            return True
        return False
    
    def can_receive_ack(self, current_time):
        if self.waiting_for_ack and self.transmission_start_time is not None:
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
    
    def transmission_failed(self):
        self.waiting_for_ack = False
        self.transmission_start_time = None

class BaseStation:
    def __init__(self, num_ues, buffer_size, arrival_prob, transmission_prob):
        self.ues = [UE(buffer_size) for _ in range(num_ues)]
        self.arrival_prob = arrival_prob
        self.transmission_prob = transmission_prob
        self.current_transmissions = {}  # 使用字典记录传输时间
        self.collision_count = 0  # 碰撞次数
    def step(self, current_time):
        # 1. SDU到达过程
        for ue in self.ues:
            if np.random.random() < self.arrival_prob:
                ue.add_sdu()
        
        # 2. 处理当前传输
        completed_transmissions = set()
        current_time_transmissions = set()  # 记录当前时间步的传输
        
        # 检查哪些传输完成
        for ue_id, start_time in self.current_transmissions.items():
            ue = self.ues[ue_id]
            if ue.can_receive_ack(current_time):
                if len(self.current_transmissions) == 1:  # 只有一个传输才可能成功
                    if np.random.random() < 0.9:  # 传输成功率0.9
                        ue.receive_ack(current_time)
                    else:
                        ue.transmission_failed()
                else:  # 发生碰撞
                    ue.transmission_failed()
                    self.collision_count += 1  # 记录碰撞
                completed_transmissions.add(ue_id)
        
        # 移除完成的传输
        for ue_id in completed_transmissions:
            del self.current_transmissions[ue_id]
        
        # 3. 新的传输尝试
        for i, ue in enumerate(self.ues):
            if ue.can_transmit() and np.random.random() < self.transmission_prob:
                if ue.transmit_sdu(current_time):
                    self.current_transmissions[i] = current_time
    
    def get_statistics(self):
        total_transmitted = sum(ue.packets_transmitted for ue in self.ues)
        total_dropped = sum(ue.packets_dropped for ue in self.ues)
        buffer_occupancy = [len(ue.buffer) for ue in self.ues]
        return total_transmitted, total_dropped, buffer_occupancy

def run_simulation(num_ues, buffer_size, arrival_prob, transmission_prob, num_steps):
    bs = BaseStation(num_ues, buffer_size, arrival_prob, transmission_prob)
    
    repititions = 100
    avg_goodput = 0
    for _ in range(repititions):
        for step in range(num_steps):
            bs.step(step)  # 传入当前时间步
        transmitted, dropped, occupancy = bs.get_statistics()
        avg_goodput += transmitted / (num_steps+1)

    print(f"Total packets transmitted: {avg_goodput/repititions:.3f} packets/step")
    # print(f"Total packets dropped: {dropped}")
    # print(f"Buffer occupancy: {occupancy}")
    # print(f"Average throughput: {transmitted/(step+1):.3f} packets/step")

# 运行模拟
if __name__ == "__main__":
    # 设置参数
    L = 2       # UE数量
    B = 20      # 缓冲区大小
    T = 24      # 模拟步数
    pa = 2/T   # SDU到达概率
    pt = 0.5    # 传输概率
    
    run_simulation(L, B, pa, pt, T)