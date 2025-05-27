import environment as env
import save_result as save
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import defaultdict, deque

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
EPSILON = 1
NUM_OF_FRAME = 10000
T = 1e-3
D = 8000
I = 2  # Số lượng Q-network
LAMBDA_P = 0.5
LAMBDA = 0.995
X0 = 1

# Tham số mới cho Replay Buffer
REPLAY_BUFFER_SIZE = 10000     # Kích thước buffer
BATCH_SIZE = 32               # Kích thước batch để học
MIN_REPLAY_SIZE = 1000        # Kích thước tối thiểu để bắt đầu học
# Tham số cho Target Network
TARGET_UPDATE_FREQ = 10  # Cập nhật target network mỗi 100 frames

# Tham số mới cho việc chọn action từ replay buffer
ACTION_SELECT_BATCH_SIZE = 16  # Kích thước batch để chọn action
MIN_ACTION_SELECT_SIZE = 500   # Kích thước tối thiểu để bắt đầu chọn action từ buffer

# ===== Định nghĩa lớp ReplayBuffer =====
class ReplayBuffer:
    def __init__(self, buffer_size):
        """Khởi tạo buffer với kích thước cho trước"""
        self.buffer = deque(maxlen=buffer_size)
    
    def add(self, state, action, reward, next_state):
        """
        Thêm một mẫu trải nghiệm vào buffer
        state: trạng thái hiện tại
        action: hành động được thực hiện
        reward: phần thưởng nhận được
        next_state: trạng thái tiếp theo
        """
        experience = (state, action, reward, next_state)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """
        Lấy mẫu ngẫu nhiên một batch từ buffer
        Trả về tuple (states, actions, rewards, next_states)
        """
        # Đảm bảo batch_size không lớn hơn kích thước hiện tại của buffer
        batch_size = min(batch_size, len(self.buffer))
        
        # Lấy mẫu ngẫu nhiên các chỉ số
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        # Lấy dữ liệu từ các chỉ số đã chọn
        states = []
        actions = []
        rewards = []
        next_states = []
        
        for idx in indices:
            state, action, reward, next_state = self.buffer[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
        
        # Chuyển sang numpy array để dễ xử lý
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states)
        )
    
    def sample_states_only(self, batch_size):
        """
        Lấy mẫu ngẫu nhiên chỉ các states từ buffer (để chọn action)
        """
        batch_size = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states = []
        for idx in indices:
            state, _, _, _ = self.buffer[idx]
            states.append(state)
        
        return np.array(states)
    
    def __len__(self):
        """Trả về kích thước hiện tại của buffer"""
        return len(self.buffer)

# Chuyển state, action thành key
def state_action_to_key(state, action):
    """Chuyển state và action thành key để lưu trong dictionary"""
    state_key = tuple(map(tuple, state))
    action_key = tuple(action) if isinstance(action, np.ndarray) else tuple(action)
    return (state_key, action_key)

# Hàm utility
def u(x):
    """Hàm utility"""
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
        self.fc3 = nn.Linear(64, self.action_size)
        
    def forward(self, state):
        """
        Nhận đầu vào là state, trả về Q-values cho tất cả action khả thi
        """
        # Chuyển state thành tensor và làm phẳng
        if not isinstance(state, torch.Tensor):
            if len(state.shape) == 2:  # Single state
                state = torch.FloatTensor(state.flatten()).unsqueeze(0)
            else:  # Batch of states
                state = torch.FloatTensor(state.reshape(state.shape[0], -1))
        
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def get_q_value(self, state, action):
        """Lấy Q-value cho một action cụ thể sử dụng cho đánh giá, kiểm tra"""
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

# ===== Lớp quản lý Q Networks với Target Network =====
class QNetworkManager:
    def __init__(self):
        # Khởi tạo I mạng neural network
        self.q_networks = []
        self.optimizers = []
        self.target_networks = []
        
        for _ in range(I):
            network = QNetwork()
            target_network = QNetwork()
            target_network.load_state_dict(network.state_dict())
            optimizer = optim.Adam(network.parameters(), lr=0.01)
            
            self.q_networks.append(network)
            self.target_networks.append(target_network)
            self.optimizers.append(optimizer)
        
        # Khởi tạo bảng V và alpha
        self.V = initialize_V()
        self.alpha = initialize_alpha()
        
        # Khởi tạo Replay Buffer cho từng mạng
        self.replay_buffers = [ReplayBuffer(REPLAY_BUFFER_SIZE) for _ in range(I)]
        
        # Biến đếm để cập nhật target networks
        self.update_counter = 0
    
    def update_target_networks(self):
        """Cập nhật target networks từ main networks"""
        for i in range(I):
            self.target_networks[i].load_state_dict(
                self.q_networks[i].state_dict()
            )
    
    def update_v_alpha(self, network_idx, state, action):
        """Cập nhật bảng V và alpha cho cặp (state, action)"""
        self.V[network_idx] = update_V(self.V[network_idx], state, action) 
        self.alpha[network_idx] = update_alpha(
            self.alpha[network_idx], 
            self.V[network_idx], 
            state, 
            action
        )
    
    def compute_risk_averse_Q_batch(self, random_idx, states):
        """
        Tính Q risk-averse cho batch states theo công thức 22
        Q̂(s,a) = Q_H(s,a) - λ_p * sqrt(Var[Q(s,a)])
        """
        # Chuyển states thành tensor
        if len(states.shape) == 2:  # Single state
            states_tensor = torch.FloatTensor(states.flatten()).unsqueeze(0)
        else:  # Batch of states
            states_tensor = torch.FloatTensor(states.reshape(states.shape[0], -1))
        
        # Lấy Q-values từ mạng được chọn ngẫu nhiên H
        with torch.no_grad():
            q_random = self.target_networks[random_idx](states_tensor)
        
        # Tính Q-values trung bình từ tất cả mạng
        q_avg = torch.zeros_like(q_random)
        for i in range(I):
            with torch.no_grad():
                q_avg += self.target_networks[i](states_tensor)
        q_avg /= I
        
        # Tính tổng bình phương độ lệch
        sum_squared = torch.zeros_like(q_random)
        for i in range(I):
            with torch.no_grad():
                diff = self.q_networks[i](states_tensor) - q_avg
                sum_squared += diff * diff
        
        # Tính Q risk-averse
        risk_term = -LAMBDA_P * sum_squared / (I - 1) if I > 1 else 0
        risk_averse_q = q_random + risk_term
        
        return risk_averse_q
    
    def compute_risk_averse_Q(self, random_idx, state):
        """
        Tính Q risk-averse cho single state (giữ lại để tương thích)
        """
        return self.compute_risk_averse_Q_batch(random_idx, state)
    
    def choose_action_from_replay_buffer(self, current_state, epsilon, H):
        """
        Chọn action dựa trên mini-batch từ replay buffer và state hiện tại
        """
        if random.random() < epsilon:
            return np.random.randint(0, 3, NUM_DEVICES)
        
        # Kiểm tra xem có đủ dữ liệu trong buffer không
        total_buffer_size = sum(len(buf) for buf in self.replay_buffers)
        
        if total_buffer_size < MIN_ACTION_SELECT_SIZE:
            # Nếu chưa có đủ dữ liệu, sử dụng phương pháp cũ
            risk_averse_q = self.compute_risk_averse_Q(H, current_state)
            best_action_idx = torch.argmax(risk_averse_q).item()
            return index_to_action(best_action_idx)
        
        # Lấy mini-batch states từ các replay buffers
        all_states = [current_state]  # Bắt đầu với state hiện tại
        
        # Lấy states từ các buffer có dữ liệu
        for buf in self.replay_buffers:
            if len(buf) > 0:
                # Lấy một số states từ buffer này
                sample_size = min(ACTION_SELECT_BATCH_SIZE // I, len(buf))
                if sample_size > 0:
                    sampled_states = buf.sample_states_only(sample_size)
                    all_states.extend(sampled_states)
        
        # Chuyển thành numpy array
        batch_states = np.array(all_states)
        
        # Tính Q risk-averse cho toàn bộ batch
        risk_averse_q_batch = self.compute_risk_averse_Q_batch(H, batch_states)
        
        # Tìm action tốt nhất từ toàn bộ batch
        # Lấy max Q-value cho mỗi state, sau đó chọn state có max Q-value cao nhất
        max_q_per_state = torch.max(risk_averse_q_batch, dim=1)[0]  # Max Q-value cho mỗi state
        max_q_indices = torch.max(risk_averse_q_batch, dim=1)[1]    # Index của action tốt nhất cho mỗi state
        
        # Tìm state có max Q-value cao nhất
        best_state_idx = torch.argmax(max_q_per_state).item()
        best_action_idx = max_q_indices[best_state_idx].item()
        
        return index_to_action(best_action_idx)
    
    def choose_action(self, state, epsilon, H):
        """Wrapper function để chọn action - sử dụng replay buffer nếu có đủ dữ liệu"""
        return self.choose_action_from_replay_buffer(state, epsilon, H)
    
    def update_q_network(self, network_idx, state, action, reward, next_state):
        """
        Cập nhật Q-network theo công thức 23:
        Q(s,a) = Q(s,a) + α(s,a) * [u(r + γ*max_a'Q(s',a') - Q(s,a)) - x_0]
        """
        # 1. Cập nhật V và alpha
        self.update_v_alpha(network_idx, state, action)
        
        # 2. Thêm trải nghiệm vào buffer
        self.replay_buffers[network_idx].add(state, action, reward, next_state)
        
        # 3. Nếu buffer đủ lớn, tiến hành học từ mini-batch
        if len(self.replay_buffers[network_idx]) >= MIN_REPLAY_SIZE:
            self.learn_from_replay_buffer(network_idx)
        
        # 4. Cập nhật target networks định kỳ
        self.update_counter += 1
        if self.update_counter % TARGET_UPDATE_FREQ == 0:
            self.update_target_networks()
    
    def learn_from_replay_buffer(self, network_idx):
        """Học từ replay buffer bằng cách lấy mẫu mini-batch ngẫu nhiên"""
        # 1. Lấy mạng và optimizer tương ứng
        network = self.q_networks[network_idx]
        target_network = self.target_networks[network_idx]
        optimizer = self.optimizers[network_idx]
        
        # 2. Lấy mẫu ngẫu nhiên từ replay buffer
        states, actions, rewards, next_states = self.replay_buffers[network_idx].sample(BATCH_SIZE)
        
        # 3. Chuyển dữ liệu sang tensor
        states_tensor = torch.FloatTensor(states.reshape(BATCH_SIZE, -1))
        next_states_tensor = torch.FloatTensor(next_states.reshape(BATCH_SIZE, -1))
        
        # 4. Tính Q-values hiện tại cho trạng thái
        q_values = network(states_tensor)
        
        # 5. Tính max Q-values cho trạng thái tiếp theo từ TARGET NETWORK
        with torch.no_grad():
            next_q_values = target_network(next_states_tensor)
            max_next_q = torch.max(next_q_values, dim=1)[0]
        
        # 6. Khởi tạo tensor cho target Q-values
        target_q_values = q_values.clone().detach()
        
        # 7. Tính toán và cập nhật cho từng mẫu trong batch
        for idx in range(BATCH_SIZE):
            # Chuyển action sang index
            action_idx = action_to_index(actions[idx])
            
            # Lấy alpha từ bảng
            key = state_action_to_key(states[idx], actions[idx])
            alpha_value = self.alpha[network_idx].get(key, 1.0)
            
            # Tính TD error
            td_error = rewards[idx] + GAMMA * max_next_q[idx].item() - q_values[idx, action_idx].item()
            
            # Tính utility
            utility_value = u(td_error) - X0
            
            # Cập nhật target Q-value
            target_q_values[idx, action_idx] = q_values[idx, action_idx] + alpha_value * utility_value
        
        # 8. Tính loss và cập nhật mạng
        loss = F.mse_loss(q_values, target_q_values)
        
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
    state_plot = []
    action_plot = []
    reward_plot = []
    packet_loss_rate_plot = []
    rate_plot = []
    number_of_received_packet_plot = []
    number_of_send_packet_plot = []
    
    # Vòng lặp chính
    for frame in range(1, NUM_OF_FRAME + 1):
        # Cập nhật epsilon
        # EPSILON = EPSILON * LAMBDA
        EPSILON = max(EPS_END, EPSILON * LAMBDA)
        # if frame > 1000:
        #     EPSILON = max(EPS_END, EPSILON * LAMBDA)
        # else: 
        #     EPSILON = 1
        # Thiết lập môi trường
        h_base_t = h_base[frame]
        
        # Chọn ngẫu nhiên một Q-network (tương ứng với H trong mã gốc)
        H = np.random.randint(0, I)
        
        # Chọn action sử dụng Q risk-averse
        action = q_manager.choose_action(state, EPSILON, H)
        action_plot.append(action)

        allocation = allocate(action)
        
        # Thực hiện action
        l_max_estimate = l_kv_success(average_r)
        l_sub_max_estimate = l_max_estimate[0]
        l_mW_max_estimate = l_max_estimate[1]

        number_of_send_packet = perform_action(action, l_sub_max_estimate, l_mW_max_estimate)
        number_of_send_packet_plot.append(number_of_send_packet)

        # Nhận feedback
        r = compute_r(device_positions, h_base_t, allocation, frame)
        rate_plot.append(r)
        
        l_max = l_kv_success(r)
        l_sub_max = l_max[0]
        l_mW_max = l_max[1]
        
        number_of_received_packet = receive_feedback(number_of_send_packet, l_sub_max, l_mW_max)
        number_of_received_packet_plot.append(number_of_received_packet)

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
            print(f"Replay buffer size: {[len(buf) for buf in q_manager.replay_buffers]}")
    


    total_reward = np.sum(reward_plot)
    print("Avg reward:", total_reward/10000)
    total_received = sum(np.sum(arr) for arr in number_of_received_packet_plot)
    total_send = sum(np.sum(arr) for arr in number_of_send_packet_plot)
    print("Avg success:", total_received/total_send)
    # ==== PLR per device =====
    packet_loss_rate_plot = np.array(packet_loss_rate_plot)
    plr_sum_per_device = np.sum(packet_loss_rate_plot, axis=2) #plr của từng thiết bị qua từng frame 
    total_plr_per_device = np.sum(plr_sum_per_device, axis=0) #tổng của plr của tất cả frames cho từng thiết bị (10000 frames)

    #tính trung bình plr của từng thiết bị qua tất cả frames
    for i, total in enumerate(total_plr_per_device):
        avg_plr_of_devices = 0.0
        print(f"Thiết bị {i + 1}: Avg packet loss rate = {total/NUM_OF_FRAME}")
        avg_plr_of_devices += PLR_MAX*NUM_OF_FRAME - total
    avg_plr_total_of_device = avg_plr_of_devices / (NUM_DEVICES*NUM_OF_FRAME) #giá trị trung bình lỗi trên tất cả các thiết bị

    print("Avg plr_total_of_device:", avg_plr_total_of_device)
    # tunable_parameters = {
    # 'h_base_sub6': h_base_t,
    # 'state': state_plot,
    # 'action': action_plot,
    # 'reward': reward_plot,
    # 'packet_loss_rate': packet_loss_rate_plot,
    # 'rate_plot': rate_plot,
    # 'number_of_received_packet': number_of_received_packet_plot,
    # 'number_of_send_packet': number_of_send_packet_plot,
    # 'Avg reward': total_reward/10000,
    # 'Avg success': total_received/total_send,
    # }

    # save.save_tunable_parameters_txt(I, NUM_DEVICES, tunable_parameters, save_dir='tunable_para_test_04')

    # Vẽ đồ thị reward
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, NUM_OF_FRAME + 1), reward_plot, label='Reward theo frame', color='green')
    plt.title('Biểu đồ Reward theo từng Frame (với Replay Buffer)')
    plt.xlabel('Frame')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Vẽ đồ thị PLR
    packet_loss_rate_plot = np.array(packet_loss_rate_plot)
    frames = np.arange(1, packet_loss_rate_plot.shape[0] + 1)
    plr_sum_per_device = np.sum(packet_loss_rate_plot, axis=2)

    
    plt.figure(figsize=(12, 6))
    for device_idx in range(NUM_DEVICES):
        # plt.plot(frames, packet_loss_rate_plot[:, device_idx, 0], label=f'Device {device_idx+1} - sub-6GHz')
        # plt.plot(frames, packet_loss_rate_plot[:, device_idx, 1], label=f'Device {device_idx+1} - mmWave')
        plt.plot(frames, plr_sum_per_device[:, device_idx], label=f'Device {device_idx+1}')
    
    # Thêm đường chuẩn y = 0.1
    plt.axhline(y=0.1, color='black', linestyle='--', linewidth=1.5, label='plr_max')

    plt.title('Tỉ lệ mất gói tin (PLR) theo từng Frame (với Replay Buffer)')
    plt.xlabel('Frame')
    plt.ylabel('PLR')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ===== Biểu đồ tỉ lệ sử dụng action cho từng thiết bị =====
    # Giả sử action_plot là một list chứa các array shape (NUM_DEVICES,)
    action_array = np.array(action_plot)  # shape: (num_frames, num_devices)
    num_frames, num_devices = action_array.shape
    # Chuẩn bị mảng lưu phần trăm
    # shape: (3 hành động, num_devices)
    percentages = np.zeros((3, num_devices))  # 3 dòng: 0, 1, 2

    # Tính phần trăm cho từng hành động theo từng thiết bị
    for action in [0, 1, 2]:
        # Đếm số lần action xuất hiện ở từng cột (thiết bị)
        counts = np.sum(action_array == action, axis=0)
        percentages[action] = counts / num_frames * 100  # Chuyển sang phần trăm

    labels = [f'Device {i+1}' for i in range(num_devices)]
    x = np.arange(num_devices)  # Vị trí cột
    width = 0.25  # Độ rộng của mỗi nhóm cột

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, percentages[0], width, label='sub-6GHz (0)', color='skyblue')
    plt.bar(x,         percentages[1], width, label='mmWave (1)', color='orange')
    plt.bar(x + width, percentages[2], width, label='Cả hai (2)', color='green')

    plt.ylabel('Ratio (%)')
    plt.title('Interface usage distribution per device, scenario 1 (với Replay Buffer)')
    plt.xticks(x, labels)
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()