import environment as env
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
from datetime import datetime

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
D = 20000
I = 2  # Number of Q-networks
LAMBDA_P = 0.5
LAMBDA = 0.995
X0 = 1

# Memory replay buffer size
BUFFER_SIZE = 10000
BATCH_SIZE = 32
TARGET_UPDATE_FREQ = 100  # Update target network every 100 frames

# Define PyTorch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define DQN model class
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Main network for training
        self.model = QNetwork(state_size, action_size).to(device)
        
        # Target network for stable Q-value estimates
        self.target_model = QNetwork(state_size, action_size).to(device)
        self.update_target_network()
        
        # Memory replay buffer
        self.memory = deque(maxlen=BUFFER_SIZE)
        
        # Learning rate and optimizer
        self.learning_rate = 0.001
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
    
    def update_target_network(self):
        """Update target network weights with main network weights"""
        self.target_model.load_state_dict(self.model.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        """Add experience to memory"""
        self.memory.append((state.flatten(), action, reward, next_state.flatten(), done))
        
    def act(self, state, epsilon):
        """Choose an action using epsilon-greedy policy"""
        if np.random.rand() <= epsilon:
            return np.random.randint(0, self.action_size)
        
        state_tensor = torch.FloatTensor(state.flatten()).to(device)
        state_tensor = state_tensor.unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state_tensor)
        self.model.train()
        
        return torch.argmax(q_values).item()
    
    def replay(self, batch_size):
        """Train the network with experiences from memory"""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.zeros((batch_size, self.state_size))
        next_states = np.zeros((batch_size, self.state_size))
        actions = np.zeros(batch_size, dtype=np.int64)
        rewards = np.zeros(batch_size)
        dones = np.zeros(batch_size, dtype=np.bool_)
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            next_states[i] = next_state
            actions[i] = action
            rewards[i] = reward
            dones[i] = done
        
        # Convert to PyTorch tensors
        states_tensor = torch.FloatTensor(states).to(device)
        next_states_tensor = torch.FloatTensor(next_states).to(device)
        actions_tensor = torch.LongTensor(actions).to(device)
        rewards_tensor = torch.FloatTensor(rewards).to(device)
        dones_tensor = torch.BoolTensor(dones).to(device)
        
        # Predict Q-values for current states and next states
        self.model.train()
        current_q_values = self.model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.target_model(next_states_tensor).max(1)[0]
            target_q_values = rewards_tensor + GAMMA * next_q_values * (~dones_tensor)
        
        # Calculate loss and update model
        loss = self.loss_fn(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def predict(self, state):
        """Predict Q-values for a given state"""
        state_tensor = torch.FloatTensor(state.flatten()).to(device)
        state_tensor = state_tensor.unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state_tensor)
        self.model.train()
        
        return q_values.squeeze(0).cpu().numpy()

# Convert actions from (device x action_per_device) format to single action index
def action_to_index(action):
    """Convert multi-device action to a single index"""
    index = 0
    for i, a in enumerate(action):
        index += a * (3 ** i)  # Base-3 encoding since each device has 3 possible actions
    return index

def index_to_action(index, num_devices=NUM_DEVICES):
    """Convert single index back to multi-device action"""
    action = np.zeros(num_devices, dtype=int)
    for i in range(num_devices - 1, -1, -1):
        action[i] = index % 3
        index //= 3
    return action

def initialize_state():
    state = np.zeros(shape=(NUM_DEVICES, 4))
    return state

def update_state(state, plr, feedback):
    """Update state based on PLR and feedback"""
    next_state = np.zeros(shape=(NUM_DEVICES, 4))
    for k in range(NUM_DEVICES):
        for i in range(2):
            if plr[k, i] <= PLR_MAX:
                next_state[k, i] = 1
            elif plr[k, i] > PLR_MAX:
                next_state[k, i] = 0
            next_state[k, i+2] = feedback[k, i]
    return next_state

def initialize_action():
    action = np.random.randint(0, 3, NUM_DEVICES)
    return action

def choose_action(state, q_networks, epsilon):
    """Choose action using epsilon-greedy with Q-networks"""
    if np.random.rand() < epsilon:
        return initialize_action()
    
    # Randomly select one of the Q-networks
    H = np.random.randint(0, I)
    q_network = q_networks[H]
    
    # Get Q-values for all possible actions
    q_values = []
    for i in range(3**NUM_DEVICES):
        action = index_to_action(i)
        q_values.append(q_network.predict(state)[i])
    
    action_index = np.argmax(q_values)
    return index_to_action(action_index)

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
    """Calculate rate for each interface at given frame"""
    r = []
    r_sub = np.zeros(NUM_DEVICES)
    r_mW = np.zeros(NUM_DEVICES)
    h_base_sub = h_base[0]
    for k in range(NUM_DEVICES):
        sub_channel_index = allocation[0][k]
        mW_beam_index = allocation[1][k]

        if sub_channel_index != -1:
            h_sub_k = env.h_sub(device_positions, k, h_base_sub[k, sub_channel_index])
            r_sub[k] = env.r_sub(h_sub_k, device_index=k)
        if mW_beam_index != -1:
            h_mW_k = env.h_mW(device_positions, k, frame)
            r_mW[k] = env.r_mW(h_mW_k, device_index=k)

    r.append(r_sub)
    r.append(r_mW)
    return r

def l_kv_success(r):
    """Compute number of successful packets from AP each frame"""
    l_kv_success = np.floor(np.multiply(r, T/D))
    return l_kv_success

def compute_average_rate(average_r, last_r, frame_num):
    """Compute average rate"""
    avg_r = average_r.copy()
    for k in range(NUM_DEVICES):
        avg_r[0][k] = (last_r[0][k] + avg_r[0][k]*(frame_num - 1))/frame_num
        avg_r[1][k] = (last_r[1][k] + avg_r[1][k]*(frame_num - 1))/frame_num
    return avg_r

def allocate(action):
    """Allocate resources based on action"""
    sub = []
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
        if action[k] == 0:
            if len(rand_sub) > 0:
                rand_index = np.random.randint(len(rand_sub))
                sub[k] = rand_sub[rand_index]
                rand_sub.pop(rand_index)
        if action[k] == 1:
            if len(rand_mW) > 0:
                rand_index = np.random.randint(len(rand_mW))
                mW[k] = rand_mW[rand_index]
                rand_mW.pop(rand_index)
        if action[k] == 2:
            if len(rand_sub) > 0 and len(rand_mW) > 0:
                rand_sub_index = np.random.randint(len(rand_sub))
                rand_mW_index = np.random.randint(len(rand_mW))
                
                sub[k] = rand_sub[rand_sub_index]
                mW[k] = rand_mW[rand_mW_index]

                rand_sub.pop(rand_sub_index)
                rand_mW.pop(rand_mW_index)
    return [sub, mW]

def perform_action(action, l_sub_max, l_mW_max):
    """Decide number of packets to transmit to each device"""
    number_of_packet = np.zeros(shape=(NUM_DEVICES, 2))
    for k in range(NUM_DEVICES):
        l_sub_max_k = l_sub_max[k]
        l_mW_max_k = l_mW_max[k]
        if action[k] == 0:
            number_of_packet[k, 0] = min(l_sub_max_k, MAX_PACKETS)
            number_of_packet[k, 1] = 0
        if action[k] == 1:
            number_of_packet[k, 0] = 0
            number_of_packet[k, 1] = min(l_mW_max_k, MAX_PACKETS)
        if action[k] == 2:
            if l_mW_max_k < MAX_PACKETS:
                number_of_packet[k, 1] = min(l_mW_max_k, MAX_PACKETS)
                number_of_packet[k, 0] = min(l_sub_max_k, MAX_PACKETS - number_of_packet[k, 1])
            else:
                number_of_packet[k, 1] = MAX_PACKETS
                number_of_packet[k, 0] = 0
    return number_of_packet

def receive_feedback(packet_send, l_sub_max, l_mW_max):
    """Receive ACK/NACK feedback"""
    feedback = np.zeros(shape=(NUM_DEVICES, 2))
    
    for k in range(NUM_DEVICES):
        l_sub_k = packet_send[k, 0]
        l_mW_k = packet_send[k, 1]

        feedback[k, 0] = min(l_sub_k, l_sub_max[k])
        feedback[k, 1] = min(l_mW_k, l_mW_max[k])
    
    return feedback

def compute_packet_loss_rate(frame_num, old_packet_loss_rate, received_packet_num, sent_packet_num):
    """Calculate packet loss rate"""
    plr = np.zeros(shape=(NUM_DEVICES, 2))
    for k in range(NUM_DEVICES):
        plr[k, 0] = env.packet_loss_rate(frame_num, old_packet_loss_rate[k, 0], 
                                        received_packet_num[k, 0], sent_packet_num[k, 0])
        plr[k, 1] = env.packet_loss_rate(frame_num, old_packet_loss_rate[k, 1], 
                                        received_packet_num[k, 1], sent_packet_num[k, 1])
    return plr

def compute_reward(state, num_of_send_packet, num_of_received_packet, old_reward_value, frame_num):
    """Compute reward"""
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
    """Utility function for risk-averse approach"""
    return -np.exp(BETA*x)

def compute_risk_averse_q_values(q_networks, state):
    """Compute risk-averse Q-values from multiple Q-networks"""
    # Get all actions' Q-values from all networks
    all_q_values = np.zeros((I, 3**NUM_DEVICES))
    
    for i in range(I):
        all_q_values[i] = q_networks[i].predict(state)
    
    # Calculate average Q-values
    avg_q_values = np.mean(all_q_values, axis=0)
    
    # Calculate variance term
    var_term = np.zeros(3**NUM_DEVICES)
    for i in range(I):
        var_term += (all_q_values[i] - avg_q_values)**2
    
    var_term = -(var_term * LAMBDA_P) / (I-1)
    
    # Random Q-network selection
    H = np.random.randint(0, I)
    
    # Risk-averse Q-values
    risk_q_values = all_q_values[H] + var_term
    
    return risk_q_values

def main():
    print(f"Using device: {device}")
    
    # Setup environment
    device_positions = env.initialize_pos_of_devices()
    
    # Initialize state and action
    state = initialize_state()
    action = initialize_action()
    reward_value = 0.0
    allocation = allocate(action)
    packet_loss_rate = np.zeros(shape=(NUM_DEVICES, 2))
    
    # Generate h_base for each frame
    h_base = create_h_base(NUM_OF_FRAME + 1)
    h_base_t = h_base[0]
    average_r = compute_r(device_positions, h_base_t, allocation=allocate(action), frame=1)
    
    # Initialize Q-networks
    state_size = NUM_DEVICES * 4  # Each device has 4 state dimensions
    action_size = 3**NUM_DEVICES  # 3 possible actions per device
    
    q_networks = []
    for _ in range(I):
        q_networks.append(DQN(state_size, action_size))
    
    # Track metrics
    reward_plot = []
    packet_loss_rate_plot = []
    rate_plot = []
    
    epsilon = EPSILON  # Starting epsilon for exploration
    
    for frame in range(1, NUM_OF_FRAME + 1):
        # Decay epsilon
        epsilon = max(EPS_END, epsilon * LAMBDA)
        
        # Set up environment for current frame
        h_base_t = h_base[frame]
        
        # Select action using risk-averse approach
        action = choose_action(state, q_networks, epsilon)
        allocation = allocate(action)
        
        # Perform action
        l_max_estimate = l_kv_success(average_r)
        l_sub_max_estimate = l_max_estimate[0]
        l_mW_max_estimate = l_max_estimate[1]
        number_of_send_packet = perform_action(action, l_sub_max_estimate, l_mW_max_estimate)
        
        # Get feedback
        r = compute_r(device_positions, h_base_t, allocation, frame)
        rate_plot.append(r)
        
        print(f"Frame {frame}: Calculated Rate r at device = {r}")
        
        l_max = l_kv_success(r)
        print(f"Tong so goi tin nhan duoc thanh cong {l_max} tai frame {frame}")
        l_sub_max = l_max[0]
        print(f"So goi tin l_sub_max nhan duoc thanh cong {l_sub_max} tai frame {frame}")
        l_mW_max = l_max[1]
        print(f"So goi tin l_mW_max nhan duoc thanh cong {l_mW_max} tai frame {frame}")
        
        number_of_received_packet = receive_feedback(number_of_send_packet, l_sub_max, l_mW_max)
        print(f"number_of_received_packet {number_of_received_packet} tai frame {frame}")
        
        packet_loss_rate = compute_packet_loss_rate(frame, packet_loss_rate, number_of_received_packet, number_of_send_packet)
        print(f"packet_loss_rate {packet_loss_rate} tai frame {frame}")
        packet_loss_rate_plot.append(packet_loss_rate)
        
        average_r = compute_average_rate(average_r, r, frame)
        
        # Compute reward
        reward_value = compute_reward(state, number_of_send_packet, number_of_received_packet, reward_value, frame)
        next_state = update_state(state, packet_loss_rate, number_of_received_packet)
        reward_plot.append(reward_value)
        
        print(f"Frame {frame}: Reward = {reward_value}")
        
        # Store experience in memory
        action_index = action_to_index(action)
        for i in range(I):
            # Generate Poisson mask for each network (J)
            if np.random.poisson(1) == 1:
                # Add to memory replay
                q_networks[i].remember(state, action_index, reward_value, next_state, False)
                
                # Train on mini-batch
                loss = q_networks[i].replay(min(BATCH_SIZE, len(q_networks[i].memory)))
                if loss is not None and frame % 100 == 0:
                    print(f"Frame {frame}, Network {i}, Loss: {loss:.4f}")
                
                # Update target network periodically
                if frame % TARGET_UPDATE_FREQ == 0:
                    q_networks[i].update_target_network()
        
        state = next_state
        print(f'frame: {frame}')
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, NUM_OF_FRAME + 1), reward_plot, label='Reward theo frame', color='green')
    plt.title('Biểu đồ Reward theo từng Frame')
    plt.xlabel('Frame')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # # Save model weights
    # save_dir = "models"
    # os.makedirs(save_dir, exist_ok=True)
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # for i, q_network in enumerate(q_networks):
    #     model_path = os.path.join(save_dir, f"q_network_{i}_{timestamp}.pth")
    #     torch.save(q_network.model.state_dict(), model_path)
    #     print(f"Model {i} saved to {model_path}")
    
    # # Save results
    # save_dir = "results"
    # os.makedirs(save_dir, exist_ok=True)
    # file_path = os.path.join(save_dir, f"Training_results_{timestamp}.npz")
    
    # np.savez(file_path,
    #          reward_plot=reward_plot,
    #          packet_loss_rate_plot=packet_loss_rate_plot,
    #          rate_plot=rate_plot)
    
    print(f"✅ Đã lưu kết quả vào file: {file_path}")

if __name__ == "__main__":
    main()