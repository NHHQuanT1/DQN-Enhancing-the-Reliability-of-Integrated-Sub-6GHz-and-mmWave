import environment as env
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
GAMMA = 0.99  # Discount factor tăng lên để quan tâm nhiều hơn đến phần thưởng tương lai
EPS_START = 0.5  # Khởi đầu epsilon
EPS_END = 0.05  # Kết thúc epsilon
EPS_DECAY = 0.998  # Decay factor tăng lên để khai thác lâu hơn
BETA = -0.5
EPSILON = 0.5
NUM_OF_FRAME = 10000
T = 1e-3
D = 8000
I = 4  # Số lượng Q-network
LAMBDA_P = 0.7  # Tăng tham số điều chỉnh rủi ro
LAMBDA = 0.998  # Điều chỉnh để epsilon giảm chậm hơn
X0 = 1
BATCH_SIZE = 64  # Kích thước batch cho Experience Replay
BUFFER_SIZE = 10000  # Kích thước của Replay Buffer
TAU = 0.001  # Soft update param cho target network
UPDATE_EVERY = 4  # Cập nhật target network mỗi 4 bước
REWARD_SCALING = 1.5  # Hệ số tỷ lệ phần thưởng

# Class Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))
        
    def sample(self, batch_size):
        if len(self) < batch_size:
            batch_size = len(self)
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# Chuyển state, action thành key
def state_action_to_key(state, action):
    """Chuyển state và action thành key để lưu trong dictionary"""
    state_key = tuple(map(tuple, state)) # chuyển từng hàng thành 1 tuple, sau đó chuyển thành tuple lồng tuple
    action_key = tuple(action) if isinstance(action, np.ndarray) else tuple(action) # kiểm tra xem có phải array hay danh sách (python) để chuyển về tuple cả
    return (state_key, action_key)

# Hàm utility
def u(x):
    """Hàm utility"""
    return -np.exp(BETA * x)

# Mã hóa action thành index và ngược lại
def action_to_index(action):
    """Chuyển action từ array sang index"""
    index = 0
    for i, a in enumerate(action): # Duyệt qua phần tử a của action ứng với chỉ số i, tức giá trị a[i] trong action truyền vào
        index += int(a) * (3 ** i) # index = a + 3^i
    return index

def index_to_action(index):
    """Chuyển index thành action array"""
    action = np.zeros(NUM_DEVICES, dtype=int)
    for i in range(NUM_DEVICES):
        action[i] = index % 3
        index = index // 3
    return action

# ===== Định nghĩa kiến trúc mạng neural network cải tiến =====
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.state_dim = NUM_DEVICES * 4  # Mỗi thiết bị có 4 đặc trưng
        self.action_size = 3**NUM_DEVICES  # Tổng số action (3 actions cho mỗi thiết bị)
        
        # Xây dựng mạng neural với kiến trúc lớn hơn
        self.fc1 = nn.Linear(self.state_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)  # Thêm batch normalization
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, self.action_size)
        
        # Khởi tạo trọng số
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, state):
        """
        Nhận đầu vào là state, trả về Q-values cho tất cả action khả thi
        """
        # Chuyển state thành tensor và làm phẳng
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state.flatten()).unsqueeze(0) #thêm một chiều 
        
        # Forward pass với batch normalization và LeakyReLU
        x = F.leaky_relu(self.bn1(self.fc1(state)) if state.size(0) > 1 else self.fc1(state))
        x = F.leaky_relu(self.bn2(self.fc2(x)) if state.size(0) > 1 else self.fc2(x))
        x = F.leaky_relu(self.bn3(self.fc3(x)) if state.size(0) > 1 else self.fc3(x))
        return self.fc4(x)
    
    def get_q_value(self, state, action):
        """Lấy Q-value cho một action cụ thể sử dụng cho đánh giá, kiểm tra"""
        with torch.no_grad(): #tắt tính toán gradient cho dự đoán
            q_values = self(state) #bản chất  self.__call__(input)
            action_idx = action_to_index(action)
            return q_values[0, action_idx].item() 

# ===== Hàm xử lý bảng V và alpha =====
def initialize_V():
    """Khởi tạo I bảng V cho các cặp (state, action)"""
    V_tables = [{} for _ in range(I)] #tạo các từ điển rỗng
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
    # Cập nhật alpha với decay chậm hơn
    alpha[key] = 1.0 / (V[key]**0.5) if key in V and V[key] > 0 else 1.0 # anpha = 1 nếu chưa truy cập
    return alpha

# ===== Lớp quản lý Q Networks với bảng V và alpha, target network và experience replay =====
class QNetworkManager:
    def __init__(self):
        # Khởi tạo I mạng neural network và target networks
        self.q_networks = []
        self.target_networks = []
        self.optimizers = []
        self.replay_buffers = []
        self.t_steps = 0
        
        for _ in range(I):
            # Khởi tạo mạng chính
            network = QNetwork()
            optimizer = optim.Adam(network.parameters(), lr=0.001) #tạo bộ tối ưu hoá cho mạng
            
            # Khởi tạo target network
            target_network = QNetwork()
            target_network.load_state_dict(network.state_dict())
            target_network.eval()  # Đặt ở chế độ evaluation
            
            # Khởi tạo replay buffer
            replay_buffer = ReplayBuffer()
            
            self.q_networks.append(network)
            self.target_networks.append(target_network)
            self.optimizers.append(optimizer)
            self.replay_buffers.append(replay_buffer)
        
        # Khởi tạo bảng V và alpha
        self.V = initialize_V()
        self.alpha = initialize_alpha()
    
    def update_v_alpha(self, network_idx, state, action):
        """Cập nhật bảng V và alpha cho cặp (state, action)"""
        self.V[network_idx] = update_V(self.V[network_idx], state, action) #cập nhật bảng V ứng với network tương ứng với i sinh ra ở pp Poisson
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
        state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0) #làm phẳng trạng thái và thêm 1 chiều vào
        
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
        
        # Tính Q risk-averse với lambda_p tăng lên
        var_term = sum_squared / (I - 1) if I > 1 else torch.zeros_like(q_random)
        risk_term = -LAMBDA_P * torch.sqrt(var_term + 1e-6)  # Thêm epsilon nhỏ để tránh sqrt(0)
        risk_averse_q = q_random + risk_term
        
        return risk_averse_q
    
    def choose_action(self, state, epsilon, H):
        """Chọn action theo epsilon-greedy với Q risk-averse"""
        if random.random() < epsilon:
            # Thêm khả năng khám phá thông minh hơn: 80% ngẫu nhiên hoàn toàn, 20% chọn hành động tốt hơn
            if random.random() < 0.8:
                return np.random.randint(0, 3, NUM_DEVICES)
            else:
                # Chọn ngẫu nhiên một trong các action có Q-value cao
                state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0)
                with torch.no_grad():
                    q_values = self.q_networks[H](state_tensor)[0].numpy()
                
                # Lấy top 5 action có Q-value cao nhất
                top_indices = np.argsort(q_values)[-5:]
                selected_idx = random.choice(top_indices)
                return index_to_action(selected_idx)
        
        # Chọn ngẫu nhiên một Q-network
        random_idx = H
        # Tính Q risk-averse cho tất cả action
        risk_averse_q = self.compute_risk_averse_Q(random_idx, state)
        
        # Chọn action với Q risk-averse cao nhất
        best_action_idx = torch.argmax(risk_averse_q).item()
        return index_to_action(best_action_idx)
    
    def update_q_network(self, network_idx, state, action, reward, next_state):
        """
        Cập nhật Q-network sử dụng Experience Replay và Target Network
        """
        # 1. Lưu trải nghiệm vào replay buffer
        self.replay_buffers[network_idx].push(state, action, reward, next_state)
        
        # 2. Cập nhật V và alpha
        self.update_v_alpha(network_idx, state, action)
        
        # 3. Cập nhật mạng theo chu kỳ
        self.t_steps += 1
        if self.t_steps % UPDATE_EVERY != 0 or len(self.replay_buffers[network_idx]) < BATCH_SIZE:
            return
            
        # 4. Lấy batch từ replay buffer
        batch = self.replay_buffers[network_idx].sample(BATCH_SIZE)
        states, actions, rewards, next_states = zip(*batch)
        
        # 5. Chuyển đổi thành numpy arrays
        states_array = np.vstack([s.flatten() for s in states])
        next_states_array = np.vstack([ns.flatten() for ns in next_states])
        
        # 6. Chuyển đổi thành tensors
        states_tensor = torch.FloatTensor(states_array)
        actions_tensor = torch.LongTensor([action_to_index(a) for a in actions])
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(next_states_array)
        
        # 7. Tính current Q values
        q_network = self.q_networks[network_idx]
        current_q_values = q_network(states_tensor).gather(1, actions_tensor.unsqueeze(1))
        
        # 8. Tính next Q values sử dụng target network (Double DQN)
        with torch.no_grad():
            # Lấy actions với Q-values lớn nhất từ mạng chính
            best_actions = self.q_networks[network_idx](next_states_tensor).argmax(dim=1, keepdim=True)
            # Lấy giá trị Q cho actions đó từ target network
            next_q_values = self.target_networks[network_idx](next_states_tensor).gather(1, best_actions)
            target_q_values = rewards_tensor.unsqueeze(1) * REWARD_SCALING + GAMMA * next_q_values
        
        # 9. Tính TD errors
        td_errors = target_q_values - current_q_values
        
        # 10. Tính utility từ TD errors
        utilities = torch.tensor([[u(error.item()) - X0] for error in td_errors.squeeze(1)])
        
        # 11. Lấy alpha cho từng (state, action)
        alphas = torch.ones_like(td_errors) * 0.01  # Default alpha
        for i, (s, a) in enumerate(zip(states, actions)):
            key = state_action_to_key(s, a)
            if key in self.alpha[network_idx]:
                alphas[i] = self.alpha[network_idx][key]
        
        # 12. Cập nhật Q-values
        loss = F.mse_loss(current_q_values, current_q_values + alphas * utilities)
        
        # 13. Cập nhật mạng
        optimizer = self.optimizers[network_idx]
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(q_network.parameters(), 1.0)
        optimizer.step()
        
        # 14. Soft update target network
        for target_param, local_param in zip(self.target_networks[network_idx].parameters(), 
                                          q_network.parameters()):
            target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)

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

# Cải tiến thuật toán phân bổ kênh
def allocate(action, device_positions=None, h_base=None, frame=None):
    """
    Phân bổ kênh thông minh hơn dựa trên action và vị trí thiết bị
    """
    sub = [-1] * NUM_DEVICES
    mW = [-1] * NUM_DEVICES
    
    # Tạo danh sách các kênh khả dụng
    available_sub = list(range(NUM_SUBCHANNELS))
    available_mW = list(range(NUM_BEAMS))
    
    # Ưu tiên các thiết bị có action=2 (cần cả 2 loại kênh)
    device_order = []
    for k in range(NUM_DEVICES):
        if action[k] == 2:  # Thiết bị cần cả 2 loại kênh
            device_order.append((k, 2))
    
    # Thêm các thiết bị khác vào cuối danh sách
    for k in range(NUM_DEVICES):
        if action[k] == 0:  # Thiết bị chỉ cần Sub-6GHz
            device_order.append((k, 0))
        elif action[k] == 1:  # Thiết bị chỉ cần mmWave
            device_order.append((k, 1))
    
    # Phân bổ kênh theo thứ tự ưu tiên
    for k, act_type in device_order:
        if act_type == 0 or act_type == 2:  # Cần Sub-6GHz
            if available_sub:
                # Chọn kênh sub tốt nhất nếu có thông tin vị trí và h_base
                if device_positions is not None and h_base is not None and frame is not None:
                    best_sub = available_sub[0]
                    max_rate = -1
                    for s in available_sub:
                        # Giả lập tính toán rate để chọn kênh tốt nhất
                        rate = np.random.random()  # Thay bằng tính toán thực tế nếu có
                        if rate > max_rate:
                            max_rate = rate
                            best_sub = s
                    sub[k] = best_sub
                    available_sub.remove(best_sub)
                else:
                    # Nếu không có thông tin thì chọn ngẫu nhiên
                    rand_index = np.random.randint(len(available_sub))
                    sub[k] = available_sub[rand_index]
                    available_sub.pop(rand_index)
                    
        if act_type == 1 or act_type == 2:  # Cần mmWave
            if available_mW:
                # Tương tự, chọn beam mmWave tốt nhất nếu có thông tin
                if device_positions is not None and h_base is not None and frame is not None:
                    best_mW = available_mW[0]
                    max_rate = -1
                    for m in available_mW:
                        # Giả lập tính toán rate để chọn beam tốt nhất
                        rate = np.random.random()  # Thay bằng tính toán thực tế nếu có
                        if rate > max_rate:
                            max_rate = rate
                            best_mW = m
                    mW[k] = best_mW
                    available_mW.remove(best_mW)
                else:
                    # Nếu không có thông tin thì chọn ngẫu nhiên
                    rand_index = np.random.randint(len(available_mW))
                    mW[k] = available_mW[rand_index]
                    available_mW.pop(rand_index)
    
    return [sub, mW]

# Cải tiến hàm perform_action
def perform_action(action, l_sub_max, l_mW_max):
    number_of_packet = np.zeros(shape=(NUM_DEVICES, 2))
    
    # Tối ưu hóa phân bổ gói tin dựa trên hành động và tốc độ tối đa
    for k in range(NUM_DEVICES):
        l_sub_max_k = l_sub_max[k]
        l_mW_max_k = l_mW_max[k]
        
        if action[k] == 0:  # Chỉ dùng Sub-6GHz
            number_of_packet[k, 0] = min(l_sub_max_k, MAX_PACKETS)
            number_of_packet[k, 1] = 0
        
        elif action[k] == 1:  # Chỉ dùng mmWave
            number_of_packet[k, 0] = 0
            number_of_packet[k, 1] = min(l_mW_max_k, MAX_PACKETS)
        
        elif action[k] == 2:  # Dùng cả hai
            # Nếu tổng số gói tin có thể gửi < MAX_PACKETS, gửi hết có thể
            if l_sub_max_k + l_mW_max_k <= MAX_PACKETS:
                number_of_packet[k, 0] = l_sub_max_k
                number_of_packet[k, 1] = l_mW_max_k
            else:
                # Nếu mmWave có tốc độ tốt hơn, ưu tiên mmWave
                if l_mW_max_k > l_sub_max_k:
                    number_of_packet[k, 1] = min(l_mW_max_k, MAX_PACKETS - 1)
                    number_of_packet[k, 0] = min(l_sub_max_k, MAX_PACKETS - number_of_packet[k, 1])
                else:
                    # Ngược lại ưu tiên Sub-6GHz
                    number_of_packet[k, 0] = min(l_sub_max_k, MAX_PACKETS - 1)
                    number_of_packet[k, 1] = min(l_mW_max_k, MAX_PACKETS - number_of_packet[k, 0])
    
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

# Cải tiến hàm compute_reward
def compute_reward(state, num_of_send_packet, num_of_received_packet, old_reward_value, frame_num):
    sum_reward = 0
    
    for k in range(NUM_DEVICES):
        state_k = state[k]
        # Tính thành công và thất bại
        numerator = num_of_received_packet[k, 0] + num_of_received_packet[k, 1]  # tổng số gói tin nhận được ở UE
        denominator = num_of_send_packet[k, 0] + num_of_send_packet[k, 1]  # tổng số gói tin gửi đi từ AP
        
        if denominator == 0:
            success_rate_k = 0.0
        else:
            # Tăng phần thưởng khi tỉ lệ thành công cao
            success_rate_k = (numerator / denominator) ** 1.5  # Ưu tiên hơn cho tỉ lệ thành công cao
        
        # Điều chỉnh hình phạt PLR
        plr_penalty_sub = 0.8 * (1 - state_k[0]) if state_k[0] == 0 else 0  # Giảm hình phạt
        plr_penalty_mW = 0.8 * (1 - state_k[1]) if state_k[1] == 0 else 0
        
        # Thêm phần thưởng cho việc gửi nhiều gói tin thành công
        packet_bonus = 0.1 * min(1.0, numerator / MAX_PACKETS)
        
        # Tổng hợp phần thưởng cho thiết bị k
        device_reward = success_rate_k + packet_bonus - plr_penalty_sub - plr_penalty_mW
        sum_reward += device_reward
    
    # Áp dụng trung bình di động (exponential moving average) cho phần thưởng
    alpha = 0.1  # Tham số cho trung bình di động
    reward = (1 - alpha) * old_reward_value + alpha * sum_reward
    
    return reward

# Các hàm còn lại giữ nguyên như bản gốc
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

def compute_l_max(r, frame_num):
    """Tính số gói tin tối đa có thể gửi trên mỗi kênh"""
    l_max = []
    l_sub_max = np.zeros(NUM_DEVICES)
    l_mW_max = np.zeros(NUM_DEVICES)
    r_sub = r[0]
    r_mW = r[1]
    
    for k in range(NUM_DEVICES):
        # Cải tiến công thức tính l_max với xét đến điều kiện thực tế hơn
        # Công thức: l_max = floor(r * T / D), trong đó:
        # - r là tốc độ dữ liệu (bit/s)
        # - T là độ dài khung thời gian (giây)
        # - D là kích thước gói tin (bit)
        
        # Thêm nhiễu ngẫu nhiên nhỏ để mô phỏng thực tế
        noise_factor_sub = np.random.uniform(0.95, 1.05)
        noise_factor_mW = np.random.uniform(0.92, 1.08)  # mmWave biến động nhiều hơn
        
        # Tính l_max với ràng buộc tối đa là MAX_PACKETS
        l_sub_max[k] = min(MAX_PACKETS, int(np.floor(r_sub[k] * T * noise_factor_sub / D)))
        l_mW_max[k] = min(MAX_PACKETS, int(np.floor(r_mW[k] * T * noise_factor_mW / D)))
    
    l_max = [l_sub_max, l_mW_max]
    return l_max

# Cải tiến chức năng huấn luyện
def train(num_of_frame=NUM_OF_FRAME):
    """Huấn luyện DQN với các cải tiến trong thời gian num_of_frame"""
    
    # Khởi tạo vị trí thiết bị - ngẫu nhiên trong vùng bán kính 50m
    device_positions = np.zeros((NUM_DEVICES, 2))
    for k in range(NUM_DEVICES):
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(5, 50)  # Khoảng cách từ 5m đến 50m
        device_positions[k, 0] = distance * np.cos(angle)
        device_positions[k, 1] = distance * np.sin(angle)
    
    # Khởi tạo kenh vô tuyến ngẫu nhiên cho toàn bộ quá trình huấn luyện
    h_base = create_h_base(num_of_frame)
    
    # Khởi tạo các biến training
    q_network_manager = QNetworkManager()
    state = initialize_state()
    epsilon = EPS_START
    old_plr = np.zeros(shape=(NUM_DEVICES, 2))
    old_reward_value = 0
    
    # Khởi tạo các biến theo dõi quá trình học
    reward_history = []
    epsilon_history = []
    plr_history = []
    packet_sent_history = []
    packet_received_history = []
    action_counts = np.zeros((NUM_DEVICES, 3))  # Đếm số lượng hành động mỗi loại cho từng thiết bị
    
    print("Bắt đầu huấn luyện DQN với các cải tiến...")
    
    # Vòng lặp chính của quá trình huấn luyện
    for frame in range(num_of_frame):
        # Chọn Q-network ngẫu nhiên theo phân phối Poisson
        H = np.random.randint(0, I)
        
        # Chọn action theo epsilon-greedy
        action = q_network_manager.choose_action(state, epsilon, H)
        
        # Cập nhật bộ đếm hành động
        for k in range(NUM_DEVICES):
            action_counts[k, action[k]] += 1
        
        # Phân bổ kênh
        allocation = allocate(action, device_positions, h_base, frame)
        
        # Tính rate và l_max
        r = compute_r(device_positions, h_base, allocation, frame)
        l_max = compute_l_max(r, frame)
        l_sub_max = l_max[0]
        l_mW_max = l_max[1]
        
        # Thực hiện action và nhận phản hồi
        packet_send = perform_action(action, l_sub_max, l_mW_max)
        feedback = receive_feedback(packet_send, l_sub_max, l_mW_max)
        
        # Tính PLR
        plr = compute_packet_loss_rate(frame, old_plr, feedback, packet_send)
        old_plr = plr
        
        # Cập nhật state
        next_state = update_state(state, plr, feedback)
        
        # Tính reward
        reward = compute_reward(next_state, packet_send, feedback, old_reward_value, frame)
        old_reward_value = reward
        
        # Cập nhật Q-network
        q_network_manager.update_q_network(H, state, action, reward, next_state)
        
        # Cập nhật state
        state = next_state
        
        # Giảm epsilon theo công thức mới
        epsilon = max(EPS_END, epsilon * LAMBDA)
        
        # Lưu trữ dữ liệu cho phân tích
        reward_history.append(reward)
        epsilon_history.append(epsilon)
        
        # Tính PLR trung bình
        avg_plr = np.mean(plr)
        plr_history.append(avg_plr)
        
        # Thống kê gói tin gửi/nhận
        total_sent = np.sum(packet_send)
        total_received = np.sum(feedback)
        packet_sent_history.append(total_sent)
        packet_received_history.append(total_received)
        
        # In thông tin huấn luyện sau mỗi 100 frame
        if (frame + 1) % 100 == 0:
            print(f"Frame: {frame+1}/{num_of_frame}, "
                  f"Reward: {reward:.4f}, "
                  f"Epsilon: {epsilon:.4f}, "
                  f"PLR trung bình: {avg_plr:.4f}, "
                  f"Gói tin gửi/nhận: {total_sent}/{total_received}")
    
    # Trả về kết quả huấn luyện
    training_result = {
        'reward_history': reward_history,
        'epsilon_history': epsilon_history,
        'plr_history': plr_history,
        'packet_sent_history': packet_sent_history,
        'packet_received_history': packet_received_history,
        'action_counts': action_counts,
        'q_network_manager': q_network_manager
    }
    
    return training_result

# Vẽ đồ thị kết quả huấn luyện
def plot_training_results(training_result):
    """Vẽ biểu đồ kết quả huấn luyện"""
    reward_history = training_result['reward_history']
    epsilon_history = training_result['epsilon_history']
    plr_history = training_result['plr_history']
    packet_sent_history = training_result['packet_sent_history']
    packet_received_history = training_result['packet_received_history']
    action_counts = training_result['action_counts']
    
    # Tạo cửa sổ biểu đồ
    plt.figure(figsize=(20, 12))
    
    # Biểu đồ phần thưởng
    plt.subplot(2, 3, 1)
    plt.plot(reward_history)
    plt.title('Phần thưởng theo thời gian')
    plt.xlabel('Frame')
    plt.ylabel('Phần thưởng')
    plt.grid(True)
    
    # Biểu đồ epsilon
    plt.subplot(2, 3, 2)
    plt.plot(epsilon_history)
    plt.title('Epsilon theo thời gian')
    plt.xlabel('Frame')
    plt.ylabel('Epsilon')
    plt.grid(True)
    
    # Biểu đồ PLR
    plt.subplot(2, 3, 3)
    plt.plot(plr_history)
    plt.axhline(y=PLR_MAX, color='red', linestyle='--', label=f'Ngưỡng PLR ({PLR_MAX})')
    plt.title('PLR trung bình theo thời gian')
    plt.xlabel('Frame')
    plt.ylabel('PLR')
    plt.legend()
    plt.grid(True)
    
    # Biểu đồ gói tin
    plt.subplot(2, 3, 4)
    plt.plot(packet_sent_history, label='Gửi')
    plt.plot(packet_received_history, label='Nhận')
    plt.title('Số gói tin gửi/nhận theo thời gian')
    plt.xlabel('Frame')
    plt.ylabel('Số gói tin')
    plt.legend()
    plt.grid(True)
    
    # Biểu đồ hiệu suất
    plt.subplot(2, 3, 5)
    efficiency = np.array(packet_received_history) / np.array(packet_sent_history)
    plt.plot(efficiency)
    plt.title('Hiệu suất gửi/nhận theo thời gian')
    plt.xlabel('Frame')
    plt.ylabel('Hiệu suất')
    plt.grid(True)
    
    # Biểu đồ phân phối hành động
    plt.subplot(2, 3, 6)
    bar_width = 0.25
    x = np.arange(NUM_DEVICES)
    plt.bar(x - bar_width, action_counts[:, 0] / NUM_OF_FRAME, bar_width, label='Sub-6GHz')
    plt.bar(x, action_counts[:, 1] / NUM_OF_FRAME, bar_width, label='mmWave')
    plt.bar(x + bar_width, action_counts[:, 2] / NUM_OF_FRAME, bar_width, label='Cả hai')
    plt.title('Phân phối hành động theo thiết bị')
    plt.xlabel('Thiết bị')
    plt.ylabel('Tỷ lệ lựa chọn')
    plt.xticks(x, [f'UE{i+1}' for i in range(NUM_DEVICES)])
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

# Hàm đánh giá (evaluate) mô hình sau khi huấn luyện
def evaluate(q_network_manager, num_episodes=100):
    """Đánh giá hiệu suất của mô hình đã huấn luyện"""
    print("Bắt đầu đánh giá mô hình...")
    
    # Khởi tạo vị trí thiết bị mới cho đánh giá
    device_positions = np.zeros((NUM_DEVICES, 2))
    for k in range(NUM_DEVICES):
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(5, 50)
        device_positions[k, 0] = distance * np.cos(angle)
        device_positions[k, 1] = distance * np.sin(angle)
    
    # Khởi tạo kenh vô tuyến ngẫu nhiên cho đánh giá
    h_base = create_h_base(num_episodes)
    
    # Khởi tạo các biến đánh giá
    state = initialize_state()
    old_plr = np.zeros(shape=(NUM_DEVICES, 2))
    total_reward = 0
    
    # Thống kê đánh giá
    reward_history = []
    plr_history = []
    packet_sent_total = 0
    packet_received_total = 0
    action_counts = np.zeros((NUM_DEVICES, 3))
    
    # Vòng lặp đánh giá
    for episode in range(num_episodes):
        # Chọn Q-network tốt nhất (ở đây ta chọn Q-network đầu tiên)
        H = 0
        
        # Chọn action tốt nhất (không sử dụng epsilon-greedy)
        action = q_network_manager.choose_action(state, 0.0, H)
        
        # Cập nhật bộ đếm hành động
        for k in range(NUM_DEVICES):
            action_counts[k, action[k]] += 1
        
        # Phân bổ kênh
        allocation = allocate(action, device_positions, h_base, episode)
        
        # Tính rate và l_max
        r = compute_r(device_positions, h_base, allocation, episode)
        l_max = compute_l_max(r, episode)
        l_sub_max = l_max[0]
        l_mW_max = l_max[1]
        
        # Thực hiện action và nhận phản hồi
        packet_send = perform_action(action, l_sub_max, l_mW_max)
        feedback = receive_feedback(packet_send, l_sub_max, l_mW_max)
        
        # Tính PLR
        plr = compute_packet_loss_rate(episode, old_plr, feedback, packet_send)
        old_plr = plr
        
        # Cập nhật state
        next_state = update_state(state, plr, feedback)
        
        # Tính reward
        reward = compute_reward(next_state, packet_send, feedback, 0, episode)
        total_reward += reward
        
        # Cập nhật state
        state = next_state
        
        # Lưu trữ dữ liệu cho phân tích
        reward_history.append(reward)
        
        # Tính PLR trung bình
        avg_plr = np.mean(plr)
        plr_history.append(avg_plr)
        
        # Thống kê gói tin gửi/nhận
        episode_sent = np.sum(packet_send)
        episode_received = np.sum(feedback)
        packet_sent_total += episode_sent
        packet_received_total += episode_received
    
    # Tính các chỉ số đánh giá
    avg_reward = total_reward / num_episodes
    avg_plr = np.mean(plr_history)
    transmission_efficiency = packet_received_total / packet_sent_total if packet_sent_total > 0 else 0
    
    # Phân phối hành động
    action_distribution = action_counts / num_episodes
    
    # In kết quả đánh giá
    print(f"\nKết quả đánh giá sau {num_episodes} episodes:")
    print(f"Phần thưởng trung bình: {avg_reward:.4f}")
    print(f"PLR trung bình: {avg_plr:.4f}")
    print(f"Hiệu suất truyền: {transmission_efficiency:.4f}")
    print(f"Tổng số gói tin gửi: {packet_sent_total}")
    print(f"Tổng số gói tin nhận: {packet_received_total}")
    print("\nPhân phối hành động theo thiết bị:")
    for k in range(NUM_DEVICES):
        print(f"UE{k+1}: Sub-6GHz={action_distribution[k,0]:.2f}, "
              f"mmWave={action_distribution[k,1]:.2f}, "
              f"Cả hai={action_distribution[k,2]:.2f}")
    
    # Trả về kết quả đánh giá
    evaluation_result = {
        'avg_reward': avg_reward,
        'avg_plr': avg_plr,
        'transmission_efficiency': transmission_efficiency,
        'packet_sent_total': packet_sent_total,
        'packet_received_total': packet_received_total,
        'action_distribution': action_distribution,
        'reward_history': reward_history,
        'plr_history': plr_history
    }
    
    return evaluation_result

# Hàm lưu và tải mô hình
def save_model(q_network_manager, filename='dqn_model.pth'):
    """Lưu mô hình DQN đã huấn luyện"""
    model_state = {
        'q_networks': [net.state_dict() for net in q_network_manager.q_networks],
        'target_networks': [net.state_dict() for net in q_network_manager.target_networks],
        'V': q_network_manager.V,
        'alpha': q_network_manager.alpha
    }
    torch.save(model_state, filename)
    print(f"Mô hình đã được lưu vào {filename}")

def load_model(filename='dqn_model.pth'):
    """Tải mô hình DQN đã lưu"""
    if not os.path.exists(filename):
        print(f"Không tìm thấy file {filename}")
        return None
    
    model_state = torch.load(filename)
    q_network_manager = QNetworkManager()
    
    # Nạp trọng số vào các mạng
    for i in range(I):
        q_network_manager.q_networks[i].load_state_dict(model_state['q_networks'][i])
        q_network_manager.target_networks[i].load_state_dict(model_state['target_networks'][i])
    
    # Nạp các bảng V và alpha
    q_network_manager.V = model_state['V']
    q_network_manager.alpha = model_state['alpha']
    
    print(f"Đã tải mô hình từ {filename}")
    return q_network_manager

# Hàm main
def main():
    """Hàm chính thực thi quá trình huấn luyện và đánh giá"""
    print("Bắt đầu chương trình...")
    
    # Huấn luyện mô hình
    training_result = train(NUM_OF_FRAME)
    
    # Vẽ biểu đồ kết quả huấn luyện
    plot_training_results(training_result)
    
    # Lưu mô hình
    save_model(training_result['q_network_manager'])
    
    # Đánh giá mô hình
    evaluation_result = evaluate(training_result['q_network_manager'])
    
    # So sánh với thuật toán cơ bản (nếu cần)
    print("\nChương trình kết thúc.")

# Khởi chạy chương trình khi thực thi trực tiếp
if __name__ == "__main__":
    main()