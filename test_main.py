import test_env as env
import IO
import numpy as np
import matplotlib.pyplot as plt
import Plot

# Number of APs
NUM_OF_AP = 1
# Number of Devices
NUM_OF_DEVICE = 3
# Number of Sub-6GHz channels, Number of MmWave beams
N = M = 4
# Maximum Packet Loss Rate
RHO_MAX = 0.1
# L_k
L_k = 6
# Risk control
LAMBDA_P = 0.5
# Ultility function paremeter
BETA = -0.5
# Learning parameters
GAMMA = 0.9
EPSILON = 0.5
# Decay factor
LAMBDA = 0.995
# p_max
P_MAX = 0.1
# Number of Q-tables
I = 2
X0 = -1

# PLOT DATA POINTS

# CREAT STATE
# State is a NUM_OF_DEVICE*4 matrix
# where state[k]=[QoS_satisfaction_sub, QoS_satisfaction_mW, ACK feedback at t-1 on sub, ACK feedback at t-1 on mW]
def initialize_state():
    state = np.zeros(shape=(NUM_OF_DEVICE, 4))
    return state

def update_state(state, packet_loss_rate, feedback):
    next_state = np.zeros(shape=(NUM_OF_DEVICE, 4))
    for k in range(NUM_OF_DEVICE):
        for i in range(2):
            # QoS satisfaction
            if (packet_loss_rate[k, i] <= RHO_MAX):
                next_state[k, i] = 1
            elif (packet_loss_rate[k, i] > RHO_MAX):
                next_state[k, i] = 0
            # Number of successfully delivered packet on each interface
            next_state[k, i+2] = feedback[k, i]
    return next_state

# CREATE ACTION
# Action is an array where action[k] is the action of device k
def initialize_action():
    # Initialize random action at the beginning
    action = np.random.randint(0, 3, NUM_OF_DEVICE)
    return action

def choose_action(state, Q_table):
    # Epsilon-Greedy
    p = np.random.rand()
    action = initialize_action()
    if (p < EPSILON):
        return action
    else:
        max_Q = -np.Infinity
        state = tuple([tuple(row) for row in state])
        action = tuple(action)
        random_action = []  # Containts action with Q_value = 0
        for a in Q_table[state]:
            if(Q_table[state][a]>=max_Q):
                max_Q = Q_table[state][a]
                action = a
                if(max_Q==0):
                    random_action.append(action)
        if(max_Q==0):
            action = random_action[np.random.randint(0,len(random_action))]

        return action

# Chose subchannel/beam
# allocate[0] = [index of subchannel device 0 allocate, subchannel device 1 allocate, subchannel device 2 allocate]
# allocate[1] = [index of beam device 0 allocate, beam device 1 allocate, beam device 2 allocate]

#Phân chia các action
def allocate(action):
    sub = []  # Stores index of subchannel device will allocate
    mW = []  # Stores index of beam device will allocate
    for i in range(env.NUM_OF_DEVICE):
        sub.append(-1)
        mW.append(-1)

    rand_sub = [] 
    rand_mW = []
    for i in range(env.NUM_OF_SUB_CHANNEL):
        rand_sub.append(i)
    for i in range(env.NUM_OF_BEAM):
        rand_mW.append(i)

    for k in range(NUM_OF_DEVICE):
        if (action[k] == 0):
            rand_index = np.random.randint(len(rand_sub))
            sub[k] = rand_sub[rand_index]
            rand_sub.pop(rand_index)
        if (action[k] == 1):
            rand_index = np.random.randint(len(rand_mW))
            mW[k] = rand_mW[rand_index]
            rand_mW.pop(rand_index)
        if (action[k] == 2):
            rand_sub_index = np.random.randint(len(rand_sub))
            rand_mW_index = np.random.randint(len(rand_mW))

            sub[k] = rand_sub[rand_sub_index]
            mW[k] = rand_mW[rand_mW_index]

            rand_sub.pop(rand_sub_index)
            rand_mW.pop(rand_mW_index)

    allocate = [sub, mW]
    return allocate

# Return an matrix where number_of_packet[k] = [number of transmit packets on sub, number of transmit packets on mW]
# Thực hiện các actions 
def perform_action(action, l_sub_max, l_mW_max):
    number_of_packet = np.zeros(shape=(NUM_OF_DEVICE, 2))
    for k in range(NUM_OF_DEVICE):
        l_sub_max_k = l_sub_max[k]
        l_mW_max_k = l_mW_max[k]
        if (action[k] == 0):
            # If l_sub_max too small, sent 1 packet and get bad reward later
            number_of_packet[k, 0] = max(1, min(l_sub_max_k, L_k))
            number_of_packet[k, 1] = 0

        if (action[k] == 1):
            number_of_packet[k, 0] = 0
            number_of_packet[k, 1] = max(1, min(l_mW_max_k, L_k))

        if (action[k] == 2):
            if(l_mW_max_k<L_k):
                number_of_packet[k, 1] = max(1, min(l_mW_max_k, L_k))
                number_of_packet[k, 0] = max(1, min(l_sub_max_k, L_k-number_of_packet[k,1]))
            else:
                number_of_packet[k, 1] = L_k - 1
                number_of_packet[k, 0] = 1
    return number_of_packet

# Return an matrix where feedback[k] = [number of received packets on sub, number of received packets on mW]
# Hàm tính số ACK/NACK (maybe  có thể có)
def receive_feedback(num_of_send_packet, l_sub_max, l_mW_max):
    feedback = np.zeros(shape=(NUM_OF_DEVICE, 2))

    for k in range(NUM_OF_DEVICE):
        l_sub_k = num_of_send_packet[k, 0]
        l_mW_k = num_of_send_packet[k, 1]

        feedback[k, 0] = min(l_sub_k, l_sub_max[k])
        feedback[k, 1] = min(l_mW_k, l_mW_max[k])

    return feedback


def compute_packet_loss_rate(frame_num, old_packet_loss_rate, received_packet_num, sent_packet_num):
    packet_loss_rate = np.zeros(shape=(NUM_OF_DEVICE, 2))
    for k in range(NUM_OF_DEVICE):
        packet_loss_rate[k,0] = env.packet_loss_rate(frame_num, old_packet_loss_rate[k,0],received_packet_num[k,0], sent_packet_num[k,0])
        packet_loss_rate[k,1] = env.packet_loss_rate(frame_num, old_packet_loss_rate[k,1], received_packet_num[k,1], sent_packet_num[k,1])

    return packet_loss_rate


# CREATE REWARD
# Initialize a reward table as a dictionary,
def initialize_reward(state, action):
    reward = {}
    return reward

# Update reward table
def update_reward(state, action, old_reward_table, num_of_send_packet, num_of_received_packet, frame_num):
    state_action = np.insert(state, 4, action, axis=1)
    state_action = tuple([tuple(row) for row in state_action])
    old_reward_value = 0
    if(state_action in old_reward_table):
        old_reward_value = old_reward_table[state_action]
    reward = compute_reward(state, num_of_send_packet, num_of_received_packet, old_reward_value, frame_num)
    old_reward_table.update({state_action: reward})
    return old_reward_table

# Compute reward of one pair of (state, action)
def compute_reward(state, num_of_send_packet, num_of_received_packet, old_reward_value, frame_num):
    sum = 0
    for k in range(NUM_OF_DEVICE):
        state_k = state[k]
        sum = sum + (num_of_received_packet[k, 0] + num_of_received_packet[k, 1])/(
            num_of_send_packet[k, 0] + num_of_send_packet[k, 1]) - (1 - state_k[0]) - (1-state_k[1])
    sum = ((frame_num - 1)*old_reward_value + sum)/frame_num
    return sum


# CREATE MODEL
# A Q-table is a dictionary with key=tuple(state.apend(action)), value = Q(state,action)
def initialize_Q_tables(first_state):
    Q_tables = []
    first_state = tuple([tuple(row) for row in first_state])
    for i in range(I):
        Q = {}
        add_new_state_to_table(Q,first_state)
        Q_tables.append(Q)
    return Q_tables


def add_2_Q_tables(Q1, Q2):
    q = Q1.copy()
    for state in Q2:
        if (state in q):
            for a in q[state]:
                q[state][a] += Q2[state][a]
        else:
            q.update({state: Q2[state].copy()})
    return q


def average_Q_table(Q_tables):
    res = {}
    for state in range(len(Q_tables)):
        res = add_2_Q_tables(res, Q_tables[state])
    for state in res:
        for action in res[state]:
            res[state][action] = res[state][action]/I
    return res


def compute_risk_adverse_Q(Q_tables, random_Q_index):
    Q_random = Q_tables[random_Q_index].copy()
    Q_average = average_Q_table(Q_tables)
    sum_sqr = {}
    minus_Q_average = {}
    for state in Q_average:
        for action in Q_average[state]:
            Q_average[state][action] = -Q_average[state][action]
        minus_Q_average.update({state: Q_average[state].copy()})

    for i in range(I):
        sub = {}
        sub = add_2_Q_tables(sub,Q_tables[i])
        sub = add_2_Q_tables(sub, minus_Q_average)
        for state in sub:
            for action in sub[state]:
                sub[state][action] *= sub[state][action]
        sum_sqr = add_2_Q_tables(sum_sqr, sub)

    for state in sum_sqr:
        for action in sum_sqr[state]:
            sum_sqr[state][action] = -sum_sqr[state][action]*LAMBDA_P/(I-1)
    
    res = add_2_Q_tables({},sum_sqr)
    res = add_2_Q_tables(res, Q_random)
    return res


def u(x):
    return -np.exp(BETA*x)

def add_new_state_to_table(table, state):
    state = tuple([tuple(row) for row in state])
    actions = {}
    for i in range(3):
        a = np.empty(env.NUM_OF_DEVICE)
        a[0] = i
        for j in range(3):
            a[1] = j
            for k in range(3):
                a[2] = k
                actions.update({tuple(a.copy()):0})
        table.update({state:actions})
    return table

COUNT = 0
def update_Q_table(Q_table, alpha, reward, state, action, next_state):
    state = tuple([tuple(row) for row in state])
    action = tuple(action)
    next_state = tuple([tuple(row) for row in next_state])

    # Find max(Q(s(t+1),a)
    max_Q = 0
    for a in Q_table[state]:
        if(Q_table[state][a]>max_Q):
            max_Q = Q_table[state][a]
    if(Q_table[state][action]!=0):
        global COUNT
        COUNT-=1
    Q_table[state][action] =  Q_table[state][action] + alpha[state][action] * (u(reward + GAMMA * max_Q - Q_table[state][action]) - X0)
    if(Q_table[state][action]!=0):
        COUNT+=1
    return Q_table


def initialize_V(first_state):
    V_tables = []
    for i in range(I):
        V = {}
        add_new_state_to_table(V,first_state)
        V_tables.append(V)
    return V_tables


def update_V(V, state, action):
    state = tuple([tuple(row) for row in state])
    action = tuple(action)
    if(state in V):
        V[state][action]+=1
    else:
        add_new_state_to_table(V,state)
        V[state][action] = 1

    return V


def initialize_alpha(first_state):
    return initialize_V(first_state)


def update_alpha(alpha, V, state, action):
    state = tuple([tuple(row) for row in state])
    action = tuple(action)
    if(state in alpha):
        alpha[state][action] = 1/V[state][action]
    else:
        add_new_state_to_table(alpha,state)
        alpha[state][action] = 1/V[state][action]

    return alpha

# Set up environment
# Complex channel coefficient
def generate_h_tilde(num_of_frame,mu=0, sigma=1):
    h_tilde = []
    h_tilde_sub = env.generate_h_tilde(mu, sigma, num_of_frame*NUM_OF_DEVICE*env.NUM_OF_SUB_CHANNEL)
    h_tilde_mW = env.generate_h_tilde(mu, sigma, num_of_frame*NUM_OF_DEVICE*env.NUM_OF_BEAM)
    
    for frame in range(num_of_frame):
        h_tilde_sub_t = np.empty(shape=(NUM_OF_DEVICE, env.NUM_OF_SUB_CHANNEL), dtype=complex)
        for k in range(NUM_OF_DEVICE):
            for n in range(env.NUM_OF_SUB_CHANNEL):
                h_tilde_sub_t[k, n] = h_tilde_sub[frame*NUM_OF_DEVICE *
                                                  env.NUM_OF_SUB_CHANNEL + k*env.NUM_OF_SUB_CHANNEL + n]

        h_tilde_mW_t = np.empty(shape=(NUM_OF_DEVICE, env.NUM_OF_BEAM), dtype=complex)
        for k in range(NUM_OF_DEVICE):
            for n in range(env.NUM_OF_BEAM):
                h_tilde_mW_t[k, n] = h_tilde_mW[frame*NUM_OF_DEVICE *
                                                env.NUM_OF_BEAM + k*env.NUM_OF_BEAM + n]
        h_tilde_t = [h_tilde_sub_t, h_tilde_mW_t]
        h_tilde.append(h_tilde_t)
    return h_tilde

# Achievable rate, phân phối các giá trị vận tốc thực tại device cho 10000 frame T
def compute_r(device_positions, h_tilde, alloxcation,frame):
    r = []
    r_sub = np.zeros(NUM_OF_DEVICE)
    r_mW = np.zeros(NUM_OF_DEVICE)
    h_tilde_sub = h_tilde[0]
    h_tilde_mW = h_tilde[1]
    for k in range(NUM_OF_DEVICE):
        sub_channel_index = allocation[0][k]
        mW_beam_index = allocation[1][k]
        if (sub_channel_index != -1):
            h_sub_k = env.compute_h_sub(device_positions, k, h_tilde_sub[k, sub_channel_index])
            r_sub[k] = env.r_sub(h_sub_k, device_index=k)
        if (mW_beam_index != -1):
            h_mW_k = env.compute_h_mW(device_positions, device_index=k, h_tilde=h_tilde_mW[k, mW_beam_index],frame=frame)
            r_mW[k] = env.r_mW(h_mW_k, device_index=k)

    r.append(r_sub)
    r.append(r_mW)
    return r


def compute_average_r(average_r, last_r, frame_num):
    res = average_r.copy()
    for k in range(NUM_OF_DEVICE):
        res[0][k] =(last_r[0][k] + res[0][k]*(frame_num-1))/frame_num 
        res[1][k] =(last_r[1][k] + res[1][k]*(frame_num-1))/frame_num 
    return res

# l_max = r*T/d, tính số gói tin nhận được thành công gửi từ AP đến device
def compute_l_max(r):
    l = np.floor(np.multiply(r, env.T/env.D))
    return l
































# TRAINING
# Read from old Q-tables

# Train with new data
device_positions = env.initialize_devices_pos()
state = initialize_state()
action = initialize_action()
reward = initialize_reward(state, action)
reward_value = 0
allocation = allocate(action)
Q_tables = initialize_Q_tables(state)
V = initialize_V(state)
alpha = initialize_alpha(state)
packet_loss_rate = np.zeros(shape=(NUM_OF_DEVICE, 2))

# Generate h_tilde for all frame
# h_tilde = generate_h_tilde(env.NUM_OF_FRAME+1)
h_tilde = generate_h_tilde(env.NUM_OF_FRAME+1)
h_tilde_t = h_tilde[0]
average_r = compute_r(device_positions, h_tilde_t, allocation=allocate(action),frame=1)

state_plot=[]
action_plot=[]
reward_plot=[]
number_of_sent_packet_plot=[]
number_of_received_packet_plot=[]
packet_loss_rate_plot=[]
rate_plot=[]

for frame in range(1, env.NUM_OF_FRAME+1):
    # Random Q-table
    H = np.random.randint(0, I)
    risk_adverse_Q = compute_risk_adverse_Q(Q_tables, H)

    # Update EPSILON
    EPSILON = EPSILON * LAMBDA

    # Set up environment
    h_tilde_t = h_tilde[frame]
    state_plot.append(state)

    # Select action
    action = choose_action(state, risk_adverse_Q)
    allocation = allocate(action)
    action_plot.append(action)

    # Perform action
    l_max_estimate = compute_l_max(average_r)
    l_sub_max_estimate = l_max_estimate[0]
    l_mW_max_estimate = l_max_estimate[1]
    number_of_send_packet = perform_action(action, l_sub_max_estimate, l_mW_max_estimate)
    number_of_sent_packet_plot.append(number_of_send_packet)
    

    # Get feedback
    r = compute_r(device_positions, h_tilde_t, allocation,frame)
    l_max = compute_l_max(r)
    l_sub_max = l_max[0]
    l_mW_max = l_max[1]
    rate_plot.append(r)

    number_of_received_packet = receive_feedback(number_of_send_packet, l_sub_max, l_mW_max)
    packet_loss_rate = compute_packet_loss_rate(frame, packet_loss_rate, number_of_received_packet, number_of_send_packet)
    packet_loss_rate_plot.append(packet_loss_rate)
    number_of_received_packet_plot.append(number_of_received_packet)
    average_r = compute_average_r(average_r, r, frame)

    # Compute reward
    # reward = update_reward(state, action, reward,number_of_send_packet, number_of_received_packet, frame)
    reward_value = compute_reward(state,number_of_send_packet,number_of_received_packet,reward_value,frame)
    reward_plot.append(reward_value)
    next_state = update_state(state, packet_loss_rate, number_of_received_packet)

    # Generate mask J
    J = np.random.poisson(1, I)

    for i in range(I):
        next_state_tuple = tuple([tuple(row) for row in next_state])
        if (J[i] == 1):
            V[i] = update_V(V[i],state,action)
            alpha[i] = update_alpha(alpha[i], V[i],state,action)
            Q_tables[i] = update_Q_table(Q_tables[i], alpha[i], reward_value, state, action, next_state)
        if(not (next_state_tuple in Q_tables[i])):
            add_new_state_to_table(Q_tables[i], next_state_tuple)
    state = next_state

    print('frame: ',frame)

IO.save(number_of_received_packet_plot,'number_of_received_packet')
IO.save(number_of_sent_packet_plot,'number_of_sent_packet')
IO.save(reward_plot,'reward')
IO.save(action_plot,'action')
IO.save(state_plot,'state')
IO.save(h_tilde,'h_tilde')
IO.save(device_positions,'device_positions')
IO.save(Q_tables,'Q_tables')
IO.save(reward,'all_reward')
IO.save(packet_loss_rate_plot,'packet_loss_rate')
IO.save(rate_plot,'rate')
print(COUNT)
Plot.plot_reward()
Plot.plot_interface_usage()
Plot.plot_sum_rate()
Plot.scatter_packet_loss_rate()