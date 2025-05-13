import environment as env
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import namedtuple, deque

# Hyperparameters
NUM_DEVICES = 3  # Số thiết bị (K=3, scenario 1)
NUM_SUBCHANNELS = 4  # Số subchannel Sub-6GHz (N)
NUM_BEAMS = 4  # Số beam mmWave (M)
MAX_PACKETS = 6  # Số gói tin tối đa mỗi frame (L_k(t))
PLR_MAX = 0.1  # Giới hạn PLR tối đa
GAMMA = 0.9  # Discount factor
EPS_START = 0.5  # Khởi đầu epsilon
EPS_END = 0.05  # Kết thúc epsilon
EPS_DECAY = 0.995  # Decay factor
BETA = -0.5
EPSILON = 0.5
NUM_OF_FRAME = 10000
T = 1e-3
D = 40000
I = 2  # Số lượng mạng Q-Network
LAMBDA_P = 0.5
LAMBDA = 0.995
X0 = 1

# Thông số cho DQN
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE = 10  # Số frame để cập nhật target network
LEARNING_RATE = 0.001
STATE_SIZE = NUM_DEVICES * 4  # Mỗi thiết bị có 4 trạng thái
ACTION_SIZE = 3 ** NUM_DEVICES  # 3 hành động cho mỗi thiết bị

# Define PyTorch device
device = torch.device("mps" if torch.cuda.is_available() else "cpu")
# Replay Memory để lưu trữ các chuyển trạng thái (transitions)
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# DQN Network
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # Input size: NUM_DEVICES * 4 (mỗi thiết bị có 4 giá trị trạng thái)
        self.fc1 = nn.Linear(STATE_SIZE, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, ACTION_SIZE)  # Output: 3^NUM_DEVICES (tất cả tổ hợp hành động)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def initialize_state():
    state = np.zeros(shape=(NUM_DEVICES, 4))
    return state

def update_state(state, plr, feedback): #update trạng thái so với PLR_MAX = 0.1
    next_state = np.zeros(shape=(NUM_DEVICES, 4))
    for k in range(NUM_DEVICES):
        for i in range(2):
            if(plr[k, i] <= PLR_MAX):
                next_state[k, i] = 1
            elif(plr[k, i] > PLR_MAX):
                next_state[k, i] = 0
            next_state[k, i+2] = feedback[k, i]
    return next_state

def state_to_tensor(state):
    # Chuyển đổi state từ np.array (NUM_DEVICES, 4) sang tensor 1D với shape (STATE_SIZE,)
    return torch.FloatTensor(state.flatten())

def action_to_index(action):
    # Chuyển đổi action tuple thành một index duy nhất
    index = 0
    for i, a in enumerate(action):
        index += int(a) * (3 ** i)
    return index

def index_to_action(index):
    # Chuyển đổi index thành action tuple
    action = np.zeros(NUM_DEVICES, dtype=int)
    temp = index
    for i in range(NUM_DEVICES):
        action[i] = temp % 3
        temp //= 3
    return action

def initialize_action():
    action = np.random.randint(0, 3, NUM_DEVICES)
    return action

def choose_action(state, policy_net, epsilon):
    # Epsilon-Greedy
    if random.random() < epsilon:
        return initialize_action()
    else:
        with torch.no_grad():
            state_tensor = state_to_tensor(state).unsqueeze(0)
            action_values = policy_net(state_tensor)
            action_index = action_values.max(1)[1].item()
            return index_to_action(action_index)

# Tạo h cho mỗi frame
def create_h_base(num_of_frame, mean=0, sigma=1):
    h_base = []
    h_base_sub = env.gennerate_h_base(mean, sigma, num_of_frame*NUM_DEVICES*NUM_SUBCHANNELS)
    h_base_mW = env.gennerate_h_base(mean, sigma, num_of_frame*NUM_DEVICES*NUM_BEAMS)

    for frame in range(num_of_frame):
        h_base_sub_t = np.empty(shape=(NUM_DEVICES, NUM_SUBCHANNELS), dtype=complex)
        for k in range(NUM_DEVICES):
            for n in range(NUM_SUBCHANNELS):
                h_base_sub_t[k, n] = h_base_sub[frame*NUM_DEVICES*NUM_SUBCHANNELS + k*NUM_SUBCHANNELS + n]

        h_base_mW_t = np.empty(shape=(NUM_DEVICES, NUM_BEAMS), dtype=complex)
        for k in range(NUM_DEVICES):
            for n in range(NUM_BEAMS):
                h_base_mW_t[k, n] = h_base_mW[frame*NUM_DEVICES*NUM_BEAMS + k*NUM_BEAMS + n]
        
        h_base_t = [h_base_sub_t, h_base_mW_t]
        h_base.append(h_base_t)
    return h_base

def compute_r(device_positions, h_base, allocation, frame):
    r = []
    r_sub = np.zeros(NUM_DEVICES)
    r_mW = np.zeros(NUM_DEVICES)
    h_base_sub = h_base[0]

    for k in range(NUM_DEVICES):
        sub_channel_index = allocation[0][k]
        mW_beam_index = allocation[1][k]

        if(sub_channel_index != -1):
            h_sub_k = env.h_sub(device_positions, k, h_base_sub[k, sub_channel_index])
            r_sub[k] = env.r_sub(h_sub_k, device_index=k)
        if(mW_beam_index != -1):
            h_mW_k = env.h_mW(device_positions, k, frame)
            r_mW[k] = env.r_mW(h_mW_k, device_index=k)

        r.append(r_sub)
        r.append(r_mW)
    return r

def l_kv_success(r):
    l_kv_success = np.floor(np.multiply(r, T/D))
    return l_kv_success

def compute_average_rate(average_r, last_r, frame_num):
    avg_r = average_r.copy()
    for k in range(NUM_DEVICES):
        avg_r[0][k] = (last_r[0][k] + avg_r[0][k]*(frame_num - 1))/frame_num
        avg_r[1][k] = (last_r[1][k] + avg_r[1][k]*(frame_num - 1))/frame_num
    return avg_r

def allocate(action):
    sub = [-1] * NUM_DEVICES
    mW = [-1] * NUM_DEVICES

    rand_sub = list(range(NUM_SUBCHANNELS))
    rand_mW = list(range(NUM_BEAMS))
    
    for k in range(NUM_DEVICES):
        if(action[k] == 0):
            if rand_sub:  # Check if list is not empty
                rand_index = np.random.randint(len(rand_sub))
                sub[k] = rand_sub[rand_index]
                rand_sub.pop(rand_index)
        if(action[k] == 1):
            if rand_mW:  # Check if list is not empty
                rand_index = np.random.randint(len(rand_mW))
                mW[k] = rand_mW[rand_index]
                rand_mW.pop(rand_index)
        if(action[k] == 2):
            if rand_sub and rand_mW:  # Check if both lists are not empty
                rand_sub_index = np.random.randint(len(rand_sub))
                rand_mW_index = np.random.randint(len(rand_mW))
                
                sub[k] = rand_sub[rand_sub_index]
                mW[k] = rand_mW[rand_mW_index]

                rand_sub.pop(rand_sub_index)
                rand_mW.pop(rand_mW_index)
    
    return [sub, mW]

def perform_action(action, l_sub_max, l_mW_max):
    number_of_packet = np.zeros(shape=(NUM_DEVICES, 2))
    for k in range(NUM_DEVICES):
        l_sub_max_k = l_sub_max[k]
        l_mW_max_k = l_mW_max[k]
        if(action[k] == 0):
            number_of_packet[k, 0] = min(l_sub_max_k, MAX_PACKETS)
            number_of_packet[k, 1] = 0
        if(action[k] == 1):
            number_of_packet[k, 0] = 0
            number_of_packet[k, 1] = min(l_mW_max_k, MAX_PACKETS)
        if(action[k] == 2):
            if(l_mW_max_k < MAX_PACKETS):
                number_of_packet[k, 1] = min(l_mW_max_k, MAX_PACKETS)
                number_of_packet[k, 0] = min(l_sub_max_k, MAX_PACKETS - number_of_packet[k, 1])
            else:
                number_of_packet[k, 1] = MAX_PACKETS - 1
                number_of_packet[k, 0] = 1
    return number_of_packet

def receive_feedback(packet_send, l_sub_max, l_mW_max):
    feedback = np.zeros(shape=(NUM_DEVICES, 2))

    for k in range(NUM_DEVICES):
        l_sub_k = packet_send[k, 0]
        l_mW_k = packet_send[k, 1]

        feedback[k, 0] = min(l_sub_k, l_sub_max[k])
        feedback[k, 1] = min(l_mW_k, l_mW_max[k])

    return feedback

def compute_packet_loss_rate(frame_num, old_packet_loss_rate, received_paket_num, sent_packet_num):
    plr = np.zeros(shape=(NUM_DEVICES, 2))
    for k in range(NUM_DEVICES):
        plr[k, 0] = env.packet_loss_rate(frame_num, old_packet_loss_rate[k, 0], received_paket_num[k, 0], sent_packet_num[k, 0])
        plr[k, 1] = env.packet_loss_rate(frame_num, old_packet_loss_rate[k, 1], received_paket_num[k, 1], sent_packet_num[k, 1])
        
    return plr

def compute_reward(state, num_of_send_packet, num_of_received_packet, old_reward_value, frame_num):
    sum_reward = 0
    for k in range(NUM_DEVICES):
        state_k = state[k]
        numerator = num_of_received_packet[k, 0] + num_of_received_packet[k, 1]
        denominator = num_of_send_packet[k, 0] + num_of_send_packet[k, 1]

        if denominator == 0:
            success_rate_k = 0.0
        else:
            success_rate_k = numerator / denominator

        plr_penalty_sub = (1 - state_k[0])
        plr_penalty_mW = (1 - state_k[1])

        sum_reward = sum_reward + success_rate_k - plr_penalty_sub - plr_penalty_mW

    reward = ((frame_num - 1) * old_reward_value + sum_reward) / frame_num
    return reward

def u(x):
    return -np.exp(BETA*x)

def optimize_model(policy_net, target_net, optimizer, memory):
    if len(memory) < BATCH_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    
    # Tính toán mặt nạ cho các trạng thái non-final
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    # Tính Q(s_t, a) - DQN output cho các hành động đã chọn
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    # Tính V(s_{t+1}) cho tất cả trạng thái tiếp theo
    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    
    # Tính expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    # Tính Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def compute_risk_averse_Q(Q_networks, random_Q_index):
    # Tạo danh sách để lưu outputs của các mạng trên một batch state
    state_samples = np.random.rand(100, STATE_SIZE)  # Lấy mẫu random 100 state
    state_tensor = torch.FloatTensor(state_samples)
    
    # Tính Q trung bình
    Q_avg = torch.zeros(100, ACTION_SIZE)
    for net in Q_networks:
        with torch.no_grad():
            Q_avg += net(state_tensor) / len(Q_networks)
    
    # Tính phương sai
    variance = torch.zeros(100, ACTION_SIZE)
    for net in Q_networks:
        with torch.no_grad():
            diff = net(state_tensor) - Q_avg
            variance += diff * diff / (len(Q_networks) - 1)
    
    # Tính giá trị né tránh rủi ro
    risk_term = -LAMBDA_P * variance
    
    # Kết hợp giá trị Q của mạng ngẫu nhiên với thành phần né tránh rủi ro
    with torch.no_grad():
        Q_random = Q_networks[random_Q_index](state_tensor)
        risk_averse_values = Q_random + risk_term
    
    return risk_averse_values

def init_V_tables():
    V_tables = []
    for i in range(I):
        V = {}
        for a in range(ACTION_SIZE):
            V[a] = 0  # Khởi tạo tất cả các giá trị V(s,a) = 0
        V_tables.append(V)
    return V_tables

def update_V(V, action_index):
    if action_index in V:
        V[action_index] += 1
    else:
        V[action_index] = 1
    return V

def init_alpha_tables():
    alpha_tables = []
    for i in range(I):
        alpha = {}
        for a in range(ACTION_SIZE):
            alpha[a] = 1.0  # Khởi tạo alpha = 1.0
        alpha_tables.append(alpha)
    return alpha_tables

def update_alpha(alpha, V, action_index):
    if action_index in V and V[action_index] > 0:
        alpha[action_index] = 1.0 / V[action_index]
    else:
        alpha[action_index] = 1.0
    return alpha

# Main Training Loop
def train():
    # Khởi tạo mạng DQN
    policy_nets = [DQN() for _ in range(I)]
    target_nets = [DQN() for _ in range(I)]
    
    # Khởi tạo target networks với trọng số giống policy networks
    for i in range(I):
        target_nets[i].load_state_dict(policy_nets[i].state_dict())
        target_nets[i].eval()  # Đặt target network vào chế độ đánh giá
    
    # Khởi tạo optimizers
    optimizers = [optim.Adam(net.parameters(), lr=LEARNING_RATE) for net in policy_nets]
    
    # Khởi tạo replay memory
    memories = [ReplayMemory(MEMORY_SIZE) for _ in range(I)]
    
    # Khởi tạo V tables và alpha tables
    V_tables = init_V_tables()
    alpha_tables = init_alpha_tables()
    
    # Khởi tạo môi trường
    device_positions = env.initialize_pos_of_devices()
    state = initialize_state()
    action = initialize_action()
    reward_value = 0.0
    allocation = allocate(action)
    packet_loss_rate = np.zeros(shape=(NUM_DEVICES, 2))
    
    # Generate h_base for each frame
    h_base = create_h_base(NUM_OF_FRAME + 1)
    h_base_t = h_base[0]
    average_r = compute_r(device_positions, h_base_t, allocation, frame=1)
    
    reward_plot = []
    packet_loss_rate_plot = []
    rate_plot = []
    epsilon = EPSILON
    
    for frame in range(1, NUM_OF_FRAME + 1):
        # Random Q-network
        H = np.random.randint(0, I)
        # risk_adverse_Q = compute_risk_averse_Q(Q_networks, H)
        
        # Decay epsilon for epsilon-greedy policy
        epsilon = max(EPS_END, epsilon * LAMBDA)
        # epsilon = EPSILON * LAMBDA
        
        # Set up environment
        h_base_t = h_base[frame]
        
        # Select action using the current policy network
        action = choose_action(state, policy_nets[H], epsilon)
        action_index = action_to_index(action)
        
        allocation = allocate(action)
        
        # Perform action
        l_max_estimate = l_kv_success(average_r)
        l_sub_max_estimate = l_max_estimate[0]
        l_mW_max_estimate = l_max_estimate[1]
        number_of_send_packet = perform_action(action, l_sub_max_estimate, l_mW_max_estimate)
        
        # Get feedback
        r = compute_r(device_positions, h_base_t, allocation, frame)
        rate_plot.append(r)
        
        l_max = l_kv_success(r)
        l_sub_max = l_max[0]
        l_mW_max = l_max[1]
        
        number_of_received_packet = receive_feedback(number_of_send_packet, l_sub_max, l_mW_max)
        
        packet_loss_rate = compute_packet_loss_rate(frame, packet_loss_rate, number_of_received_packet, number_of_send_packet)
        packet_loss_rate_plot.append(packet_loss_rate)
        
        average_r = compute_average_rate(average_r, r, frame)
        
        # Compute reward
        reward_value = compute_reward(state, number_of_send_packet, number_of_received_packet, reward_value, frame)
        next_state = update_state(state, packet_loss_rate, number_of_received_packet)
        reward_plot.append(reward_value)
        
        # Convert state, action, reward to tensors for PyTorch
        state_tensor = state_to_tensor(state).unsqueeze(0)
        action_tensor = torch.tensor([[action_index]], dtype=torch.long)
        next_state_tensor = state_to_tensor(next_state).unsqueeze(0)
        reward_tensor = torch.tensor([reward_value], dtype=torch.float)
        
        # Generate mask J (Poisson distribution with mean 1)
        J = np.random.poisson(1, I)
        
        # Cập nhật từng mạng Q và V tables
        for i in range(I):
            # Lưu transition vào replay memory
            memories[i].push(state_tensor, action_tensor, next_state_tensor, reward_tensor)
            
            # Chỉ cập nhật mạng được chọn theo mask J
            if J[i] == 1:
                # Cập nhật V và alpha
                V_tables[i] = update_V(V_tables[i], action_index)
                alpha_tables[i] = update_alpha(alpha_tables[i], V_tables[i], action_index)
                
                # Optimize model
                optimize_model(policy_nets[i], target_nets[i], optimizers[i], memories[i])
            
            # Cập nhật target network
            if frame % TARGET_UPDATE == 0:
                target_nets[i].load_state_dict(policy_nets[i].state_dict())
        
        # Cập nhật state
        state = next_state
        
        if frame % 100 == 0:
            print(f'Frame: {frame}, Reward: {reward_value:.4f}, Epsilon: {epsilon:.4f}')
    
    # Vẽ đồ thị
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, NUM_OF_FRAME + 1), reward_plot, label='Reward theo frame', color='green')
    plt.title('Biểu đồ Reward theo từng Frame')
    plt.xlabel('Frame')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Vẽ biểu đồ PLR
    packet_loss_rate_plot = np.array(packet_loss_rate_plot)
    frames = np.arange(1, packet_loss_rate_plot.shape[0] + 1)
    
    plt.figure(figsize=(12, 6))
    for device_idx in range(NUM_DEVICES):
        plt.plot(frames, packet_loss_rate_plot[:, device_idx, 0], label=f'Device {device_idx+1} - sub-6GHz')
        plt.plot(frames, packet_loss_rate_plot[:, device_idx, 1], label=f'Device {device_idx+1} - mmWave')
    
    plt.title('Tỉ lệ mất gói tin (PLR) theo từng Frame')
    plt.xlabel('Frame')
    plt.ylabel('PLR')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
#     # Lưu kết quả
#     import os
#     from datetime import datetime
    
#     save_dir = "results"
#     os.makedirs(save_dir, exist_ok=True)
    
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     file_path = os.path.join(save_dir, f"DQN_Training_results_{timestamp}.npz")
    
#     np.savez(file_path,
#              reward_plot=reward_plot,
#              packet_loss_rate_plot=packet_loss_rate_plot,
#              rate_plot=rate_plot)
    
#     print(f"✅ Đã lưu kết quả vào file: {file_path}")
    
#     # Lưu các model DQN
#     for i in range(I):
#         torch.save(policy_nets[i].state_dict(), os.path.join(save_dir, f"dqn_model_{i}_{timestamp}.pth"))
    
#     print(f"✅ Đã lưu các model DQN")
    
#     return policy_nets, reward_plot, packet_loss_rate_plot, rate_plot

# if __name__ == "__main__":
#     policy_nets, reward_plot, packet_loss_rate_plot, rate_plot = train()