import environment as env
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque

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
D = 8000
I = 2  # Số lượng Q-network
LAMBDA_P = 0.5
LAMBDA = 0.995
X0 = 1

# Định nghĩa kiến trúc mạng Q-network
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        # Tính số đặc trưng đầu vào: NUM_DEVICES * 4 (mỗi thiết bị có 4 đặc trưng)
        self.state_size = NUM_DEVICES * 4
        # Số hành động khả thi: 3^NUM_DEVICES (mỗi thiết bị có 3 lựa chọn hành động)
        self.action_size = 3**NUM_DEVICES
        
        # Xây dựng mạng neural network
        self.fc1 = nn.Linear(self.state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self.action_size)
        
    def forward(self, state):
        """
        Truyền đầu vào qua mạng để tính Q-values
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Chuyển đổi từ state (numpy array) sang tensor
def state_to_tensor(state):
    """Chuyển state dạng numpy array sang tensor để đưa vào neural network"""
    state_flat = state.flatten()
    return torch.FloatTensor(state_flat).unsqueeze(0)  # Thêm batch dimension

# Chuyển đổi action từ dạng tuple/array sang số nguyên
def action_to_index(action):
    """Chuyển đổi action từ mảng/tuple thành chỉ số để truy cập Q-value"""
    index = 0
    for i, a in enumerate(action):
        index += int(a) * (3 ** i)
    return index

# Chuyển đổi index thành action
def index_to_action(index):
    """Chuyển đổi chỉ số thành action dạng numpy array"""
    action = np.zeros(NUM_DEVICES, dtype=int)
    for i in range(NUM_DEVICES):
        action[i] = index % 3
        index = index // 3
    return action

# Hàm utility
def u(x):
    return -np.exp(BETA * x)

# Các hàm khởi tạo và cập nhật state vẫn giữ nguyên
def initialize_state():
    state = np.zeros(shape=(NUM_DEVICES, 4))
    return state

def update_state(state, plr, feedback):
    next_state = np.zeros(shape=(NUM_DEVICES, 4))
    for k in range(NUM_DEVICES):
        for i in range(2):
            if(plr[k, i] <= PLR_MAX):
                next_state[k, i] = 1
            elif(plr[k, i] > PLR_MAX):
                next_state[k, i] = 0
            next_state[k, i+2] = feedback[k, i]
    return next_state

# Khởi tạo action
def initialize_action():
    action = np.random.randint(0, 3, NUM_DEVICES)
    return action

# Tính toán Q trung bình rủi ro
def compute_risk_averse_Q(Q_networks, random_Q_index, state_tensor):
    """
    Tính Q trung bình rủi ro từ các Q-networks
    """
    # Lấy Q-values từ mạng được chọn ngẫu nhiên
    Q_random = Q_networks[random_Q_index](state_tensor)
    
    # Tính Q-values trung bình
    Q_average = torch.zeros_like(Q_random)
    for network in Q_networks:
        with torch.no_grad():
            Q_average += network(state_tensor)
    Q_average /= len(Q_networks)
    
    # Tính phương sai của Q-values
    variance_sum = torch.zeros_like(Q_random)
    for network in Q_networks:
        with torch.no_grad():
            diff = network(state_tensor) - Q_average
            variance_sum += diff * diff
    
    # Tính Q rủi ro
    risk_term = -LAMBDA_P * variance_sum / (len(Q_networks) - 1)
    risk_averse_q = Q_random + risk_term
    
    return risk_averse_q

# Chọn hành động dựa trên Q-network
def choose_action(state, Q_networks, epsilon):
    """
    Chọn hành động theo chiến lược epsilon-greedy từ Q-network
    """
    # Chuyển state thành tensor
    state_tensor = state_to_tensor(state)
    
    # Chọn ngẫu nhiên một trong các Q-network
    random_Q_index = random.randint(0, len(Q_networks) - 1)
    
    # Chọn action ngẫu nhiên với xác suất epsilon
    if random.random() < epsilon:
        return np.random.randint(0, 3, NUM_DEVICES)
    else:
        # Tính Q rủi ro và chọn action tốt nhất
        risk_averse_q = compute_risk_averse_Q(Q_networks, random_Q_index, state_tensor)
        best_action_index = torch.argmax(risk_averse_q).item()
        return index_to_action(best_action_index)

# Giữ nguyên các hàm khác
def create_h_base(num_of_frame, mean = 0, sigma = 1):
    h_base = []
    h_base_sub = env.generate_h_base(mean, sigma, num_of_frame*NUM_DEVICES*NUM_SUBCHANNELS)
    h_base_mW = env.generate_h_base(mean, sigma, num_of_frame*NUM_DEVICES*NUM_BEAMS)

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
    sub =[]
    mW = []
    for i in range(NUM_DEVICES):
        sub.append(-1)
        mW.append(-1)

    rand_sub = []
    rand_mW = []
    for i in range(NUM_SUBCHANNELS):
        rand_sub.append(i)
    for i in range(NUM_BEAMS):
        rand_mW.append(i)
        
    for k in range(NUM_DEVICES):
        if(action[k] == 0):
            rand_index = np.random.randint(len(rand_sub))
            sub[k] = rand_sub[rand_index]
            rand_sub.pop(rand_index)
        if(action[k] == 1):
            rand_index = np.random.randint(len(rand_mW))
            mW[k] = rand_mW[rand_index]
            rand_mW.pop(rand_index)
        if(action[k] == 2):
            rand_sub_index = np.random.randint(len(rand_sub))
            rand_mW_index = np.random.randint(len(rand_mW))
            sub[k] = rand_sub[rand_sub_index]
            mW[k] = rand_mW[rand_mW_index]

            rand_sub.pop(rand_sub_index)
            rand_mW.pop(rand_mW_index)
    allocate = [sub, mW]
    return allocate

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
    sum = 0
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

        sum = sum + success_rate_k - plr_penalty_sub - plr_penalty_mW
    reward = ((frame_num - 1) * old_reward_value + sum) / frame_num
    return reward

# Hàm cập nhật Q-network (thay thế cho update_Q_table)
def update_Q_network(Q_network, optimizer, state, action, reward, next_state, all_networks, alpha=0.01):
    """
    Cập nhật trọng số của Q-network dựa trên TD error và hàm utility
    """
    # Chuyển state thành tensor
    state_tensor = state_to_tensor(state)
    next_state_tensor = state_to_tensor(next_state)
    
    # Chuyển action thành index
    action_index = action_to_index(action)
    
    # Tính current Q value
    q_values = Q_network(state_tensor)
    current_q = q_values[0, action_index]
    
    # Tính next max Q value (từ mạng hiện tại)
    with torch.no_grad():
        # Chọn ngẫu nhiên một Q-network để tính next Q-value
        random_Q_index = random.randint(0, len(all_networks) - 1)
        risk_averse_q_next = compute_risk_averse_Q(all_networks, random_Q_index, next_state_tensor)
        max_next_q = torch.max(risk_averse_q_next).item()
    
    # Tính TD error
    target_q = reward + GAMMA * max_next_q
    td_error = target_q - current_q.item()
    
    # Áp dụng hàm utility
    utility_value = u(td_error) - X0
    
    # Tính loss và cập nhật mạng
    # Ở đây ta sẽ sử dụng utility_value như một scalar để scale loss
    loss = -utility_value * current_q  # Nhân utility_value với current_q
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Hàm update_V và update_alpha không cần thiết khi sử dụng neural network
# vì learning rate được quản lý bởi optimizer

# Chương trình chính
if __name__ == "__main__":
    # Khởi tạo các Q-networks
    Q_networks = []
    optimizers = []
    
    for i in range(I):
        network = QNetwork()
        optimizer = optim.Adam(network.parameters(), lr=0.001)
        Q_networks.append(network)
        optimizers.append(optimizer)
    
    # Khởi tạo môi trường
    device_positions = env.initialize_pos_of_devices()
    state = initialize_state()
    action = initialize_action()
    reward_value = 0.0
    allocation = allocate(action)
    packet_loss_rate = np.zeros(shape=(NUM_DEVICES, 2))
    
    # Tạo h_base cho mỗi frame
    h_base = create_h_base(NUM_OF_FRAME + 1)
    h_base_t = h_base[0]
    average_r = compute_r(device_positions, h_base_t, allocation=allocate(action), frame=1)
    
    # Các biến lưu kết quả
    reward_plot = []
    packet_loss_rate_plot = []
    rate_plot = []
    
    # Vòng lặp chính
    for frame in range(1, NUM_OF_FRAME + 1):
        # Cập nhật epsilon
        EPSILON = EPSILON * LAMBDA
        
        # Thiết lập môi trường
        h_base_t = h_base[frame]
        
        # Chọn action
        action = choose_action(state, Q_networks, EPSILON)
        allocation = allocate(action)
        
        # Thực hiện action
        l_max_estimate = l_kv_success(average_r)
        l_sub_max_estimate = l_max_estimate[0]
        l_mW_max_estimate = l_max_estimate[1]
        number_of_send_packet = perform_action(action, l_sub_max_estimate, l_mW_max_estimate)
        
        # Nhận feedback
        r = compute_r(device_positions, h_base_t, allocation, frame)
        rate_plot.append(r)
        
        l_max = l_kv_success(r)
        l_sub_max = l_max[0]
        l_mW_max = l_max[1]
        
        number_of_received_packet = receive_feedback(number_of_send_packet, l_sub_max, l_mW_max)
        
        packet_loss_rate = compute_packet_loss_rate(frame, packet_loss_rate, number_of_received_packet, number_of_send_packet)
        packet_loss_rate_plot.append(packet_loss_rate)
        
        average_r = compute_average_rate(average_r, r, frame)
        
        # Tính reward
        reward_value = compute_reward(state, number_of_send_packet, number_of_received_packet, reward_value, frame)
        reward_plot.append(reward_value)
        
        next_state = update_state(state, packet_loss_rate, number_of_received_packet)
        
        # Tạo mask J (Poisson)
        J = np.random.poisson(1, I)
        
        # Cập nhật Q-networks
        for i in range(I):
            if J[i] == 1:
                update_Q_network(Q_networks[i], optimizers[i], state, action, reward_value, next_state, Q_networks)
        
        # Chuyển sang trạng thái mới
        state = next_state
        
        # In thông tin
        print(f"Frame {frame}: Reward = {reward_value}")
        print(f"number_of_received_packet {number_of_received_packet} tai frame {frame}")
        print(f"packet_loss_rate {packet_loss_rate} tai frame {frame}")
        print('frame: ', frame)
    
    # Vẽ đồ thị reward
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, NUM_OF_FRAME + 1), reward_plot, label='Reward theo frame', color='green')
    plt.title('Biểu đồ Reward theo từng Frame')
    plt.xlabel('Frame')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Vẽ đồ thị PLR
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