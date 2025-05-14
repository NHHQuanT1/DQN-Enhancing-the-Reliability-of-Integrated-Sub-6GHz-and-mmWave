import environment as env
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import defaultdict

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

# Chuyển state, action thành key
def state_action_to_key(state, action):
    """Chuyển state và action thành key để lưu trong dictionary"""
    state_key = tuple(map(tuple, state))
    action_key = tuple(action) if isinstance(action, np.ndarray) else tuple(action)
    return (state_key, action_key)

# Hàm utility
def u(x):
    """Hàm utility theo công thức trong paper"""
    return -np.exp(BETA * x)

# Mã hóa action thành index và ngược lại
def action_to_index(action):
    """Chuyển action từ array sang index"""
    index = 0
    for i, a in enumerate(action):
        index += int(a) * (3 ** i)
    return index

def index_to_action(index):
    """Chuyển index thành action array"""
    action = np.zeros(NUM_DEVICES, dtype=int)
    for i in range(NUM_DEVICES):
        action[i] = index % 3
        index = index // 3
    return action

# ===== Định nghĩa kiến trúc mạng neural network mới =====
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.state_dim = NUM_DEVICES * 4  # Mỗi thiết bị có 4 đặc trưng
        self.action_size = 3**NUM_DEVICES  # Tổng số action (3 actions cho mỗi thiết bị)
        
        # Xây dựng mạng neural
        self.fc1 = nn.Linear(self.state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        # Đầu ra có kích thước bằng số lượng action có thể
        self.fc3 = nn.Linear(64, self.action_size)
        
    def forward(self, state):
        """
        Nhận đầu vào là state, trả về Q-values cho tất cả action khả thi
        """
        # Chuyển state thành tensor và làm phẳng
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state.flatten()).unsqueeze(0)
        
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def get_q_value(self, state, action):
        """Lấy Q-value cho một action cụ thể"""
        with torch.no_grad():
            q_values = self(state)
            action_idx = action_to_index(action)
            return q_values[0, action_idx].item()

# ===== Hàm xử lý bảng V và alpha =====
def initialize_V():
    """Khởi tạo I bảng V cho các cặp (state, action)"""
    V_tables = [{} for _ in range(I)]
    return V_tables

def update_V(V, state, action):
    """Cập nhật bảng V - đếm số lần truy cập (state, action)"""
    key = state_action_to_key(state, action)
    if key not in V:
        V[key] = 0
    V[key] += 1
    return V

def initialize_alpha():
    """Khởi tạo I bảng alpha"""
    return [{} for _ in range(I)]

def update_alpha(alpha, V, state, action):
    """Cập nhật bảng alpha - learning rate giảm theo số lần truy cập"""
    key = state_action_to_key(state, action)
    alpha[key] = 1.0 / V[key] if key in V and V[key] > 0 else 1.0
    return alpha

# ===== Lớp quản lý Q Networks với bảng V và alpha =====
class QNetworkManager:
    def __init__(self):
        # Khởi tạo I mạng neural network
        self.q_networks = []
        self.optimizers = []
        
        for _ in range(I):
            network = QNetwork()
            optimizer = optim.Adam(network.parameters(), lr=0.001)
            self.q_networks.append(network)
            self.optimizers.append(optimizer)
        
        # Khởi tạo bảng V và alpha
        self.V = initialize_V()
        self.alpha = initialize_alpha()
    
    def update_v_alpha(self, network_idx, state, action):
        """Cập nhật bảng V và alpha cho cặp (state, action)"""
        self.V[network_idx] = update_V(self.V[network_idx], state, action)
        self.alpha[network_idx] = update_alpha(
            self.alpha[network_idx], 
            self.V[network_idx], 
            state, 
            action
        )
    
    def compute_risk_averse_Q(self, random_idx, state):
        """
        Tính Q risk-averse cho tất cả action theo công thức 22
        Q̂(s,a) = Q_H(s,a) - λ_p * sqrt(Var[Q(s,a)])
        """
        # Chuyển state thành tensor
        state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0)
        
        # Lấy Q-values từ mạng được chọn ngẫu nhiên H
        with torch.no_grad():
            q_random = self.q_networks[random_idx](state_tensor)
        
        # Tính Q-values trung bình từ tất cả mạng
        q_avg = torch.zeros_like(q_random)
        for i in range(I):
            with torch.no_grad():
                q_avg += self.q_networks[i](state_tensor)
        q_avg /= I
        
        # Tính tổng bình phương độ lệch
        sum_squared = torch.zeros_like(q_random)
        for i in range(I):
            with torch.no_grad():
                diff = self.q_networks[i](state_tensor) - q_avg
                sum_squared += diff * diff
        
        # Tính Q risk-averse
        risk_term = -LAMBDA_P * sum_squared / (I - 1) if I > 1 else 0
        risk_averse_q = q_random + risk_term
        
        return risk_averse_q
    
    def choose_action(self, state, epsilon):
        """Chọn action theo epsilon-greedy với Q risk-averse"""
        if random.random() < epsilon:
            return np.random.randint(0, 3, NUM_DEVICES)
        
        # Chọn ngẫu nhiên một Q-network
        random_idx = random.randint(0, I - 1)
        
        # Tính Q risk-averse cho tất cả action
        risk_averse_q = self.compute_risk_averse_Q(random_idx, state)
        
        # Chọn action với Q risk-averse cao nhất
        best_action_idx = torch.argmax(risk_averse_q).item()
        return index_to_action(best_action_idx)
    
    def update_q_network(self, network_idx, state, action, reward, next_state):
        """
        Cập nhật Q-network theo công thức 23:
        Q(s,a) = Q(s,a) + α(s,a) * [u(r + γ*max_a'Q(s',a') - Q(s,a)) - x_0]
        """
        # 1. Cập nhật V và alpha
        self.update_v_alpha(network_idx, state, action)
        
        # 2. Lấy network và optimizer
        network = self.q_networks[network_idx]
        optimizer = self.optimizers[network_idx]
        
        # 3. Chuyển state thành tensor
        state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state.flatten()).unsqueeze(0)
        
        # 4. Tính Q-values hiện tại
        q_values = network(state_tensor)
        action_idx = action_to_index(action)
        current_q = q_values[0, action_idx]
        
        # 5. Tính max Q-value cho next state
        with torch.no_grad():
            next_q_values = network(next_state_tensor)
            max_next_q = torch.max(next_q_values).item()
        
        # 6. Tính TD error: r + γ*max_a'Q(s',a') - Q(s,a)
        td_error = reward + GAMMA * max_next_q - current_q.item()
        
        # 7. Tính utility: u(TD error) - x_0
        utility_value = u(td_error) - X0
        
        # 8. Lấy alpha từ bảng
        key = state_action_to_key(state, action)
        alpha_value = self.alpha[network_idx].get(key, 1.0)
        
        # 9. Tạo target Q-values (chỉ cập nhật cho action đã chọn)
        target_q_values = q_values.clone().detach()
        target_q_values[0, action_idx] = current_q + alpha_value * utility_value
        
        # 10. Tính loss và cập nhật network
        # Chỉ tính loss cho action được chọn
        loss = F.mse_loss(q_values[0, action_idx:action_idx+1], target_q_values[0, action_idx:action_idx+1])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Các hàm khởi tạo và cập nhật state
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

# Các hàm khác từ mã gốc giữ nguyên
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
            if len(rand_sub) > 0:
                rand_index = np.random.randint(len(rand_sub))
                sub[k] = rand_sub[rand_index]
                rand_sub.pop(rand_index)
        if(action[k] == 1):
            if len(rand_mW) > 0:
                rand_index = np.random.randint(len(rand_mW))
                mW[k] = rand_mW[rand_index]
                rand_mW.pop(rand_index)
        if(action[k] == 2):
            if len(rand_sub) > 0 and len(rand_mW) > 0:
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

# Chương trình chính
if __name__ == "__main__":
    # Khởi tạo manager cho các Q-Networks
    q_manager = QNetworkManager()
    
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
        
        # Chọn ngẫu nhiên một Q-network (tương ứng với H trong mã gốc)
        H = np.random.randint(0, I)
        
        # Chọn action sử dụng Q risk-averse
        action = q_manager.choose_action(state, EPSILON)
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
        
        # Cập nhật các Q-networks
        for i in range(I):
            if J[i] == 1:
                q_manager.update_q_network(i, state, action, reward_value, next_state)
        
        # Chuyển sang trạng thái mới
        state = next_state
        
        # In thông tin
        if frame % 100 == 0:
            print(f"Frame {frame}: Reward = {reward_value}")
            print(f"PLR = {packet_loss_rate}")
    
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