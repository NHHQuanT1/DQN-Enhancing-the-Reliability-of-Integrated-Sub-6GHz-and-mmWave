import environment as env
import numpy as np
import matplotlib.pyplot as plt

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
I = 2
LAMBDA_P = 0.5
LAMBDA = 0.995
X0 = 1
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

def initialize_action():
    action = np.random.randint(0, 3, NUM_DEVICES)
    return action

# def choose_action(state, Q_table):
#     #Epsilon-Greedy
#     p = np.random.rand()
#     action = initialize_action()
#     if(p < EPSILON):
#         return action
#     else:
#         max_Q = -np.inf
#         state = tuple([tuple(row) for row in state]) #use search in Q_table
#         action = tuple(action)
#         random_action = []
#         for a in Q_table[state]:
#             if(Q_table[state][a] >= max_Q):
#                 max_Q = Q_table[state][a]
#                 action = a
#                 if(max_Q == 0):
#                     random_action.append(action)
#         if(max_Q ==0):
#             action = random_action[np.random.randint(0, len(random_action))]

#         return action
    
def choose_action(state, Q_table):
    # Epsilon-Greedy
    p = np.random.rand()
    
    # Chuyển đổi trạng thái sang định dạng tuple
    state_tuple = tuple([tuple(row) for row in state])
    
    # Nếu trạng thái không có trong Q_table, thêm nó vào
    if state_tuple not in Q_table:
        add_new_state_to_table(Q_table, state)
    
    # Hành động ngẫu nhiên với xác suất epsilon
    if p < EPSILON:
        action = np.random.randint(0, 3, NUM_DEVICES)
        return action
    
    # Nếu không, chọn hành động tốt nhất
    else:
        max_Q = -np.inf
        best_action = tuple(np.random.randint(0, 3, NUM_DEVICES))  # Hành động mặc định
        random_actions = []
        
        for a in Q_table[state_tuple]:
            q_value = Q_table[state_tuple][a]
            
            if q_value > max_Q:
                max_Q = q_value
                best_action = a
                random_actions = [a]  # Đặt lại danh sách chỉ với hành động này
            elif q_value == max_Q:
                random_actions.append(a)
        
        # Nếu tất cả giá trị Q bằng không, chọn ngẫu nhiên
        if max_Q == 0 and random_actions:
            best_action = random_actions[np.random.randint(0, len(random_actions))]
        
        # Chuyển đổi hành động tuple trở lại mảng numpy
        return np.array(best_action)
    
#Create h for each frame (10000) of sub-6GHz,mmWave for each device
def create_h_base(num_of_frame, mean = 0, sigma = 1):
    h_base = []
    h_base_sub = env.gennerate_h_base(mean, sigma, num_of_frame*NUM_DEVICES*NUM_SUBCHANNELS)
    h_base_mW = env.gennerate_h_base(mean, sigma, num_of_frame*NUM_DEVICES*NUM_BEAMS)

    for frame in range(num_of_frame):
        h_base_sub_t = np.empty(shape=(NUM_DEVICES, NUM_SUBCHANNELS), dtype=complex)
        for k in range(NUM_DEVICES):
            for n in range(NUM_SUBCHANNELS):
                h_base_sub_t[k, n] = h_base_sub[frame*NUM_DEVICES*NUM_SUBCHANNELS + k*NUM_SUBCHANNELS + n] #index của subchannel n của device k tại frame t

        h_base_mW_t = np.empty(shape=(NUM_DEVICES, NUM_BEAMS), dtype=complex)
        for k in range(NUM_DEVICES):
            for n in range(NUM_BEAMS):
                h_base_mW_t[k, n] = h_base_mW[frame*NUM_DEVICES*NUM_BEAMS + k*NUM_BEAMS + n]
        
        h_base_t = [h_base_sub_t, h_base_mW_t]
        h_base.append(h_base_t)
    return h_base #tạo ra các phai mơ kênh
    
#Compute rate for each frame
def compute_r(device_positions, h_base, allocation, frame): #tính giá trị vận tốc trên từng giao diện tại frame thứ t
    r = []
    r_sub = np.zeros(NUM_DEVICES)
    r_mW = np.zeros(NUM_DEVICES)
    h_base_sub = h_base[0]
    # print(f"    h_base_sub: {h_base_sub}") 
    # h_base_mW = h_base[1]
    # print(f"    h_base_mW: {h_base_mW}") 

    for k in range(NUM_DEVICES):
        sub_channel_index = allocation[0][k]
        mW_beam_index = allocation[1][k]
        # print(f"    sub_channel_index: {sub_channel_index:.4f}") 
        # print(f"    mW_beam_index: {mW_beam_index:.4f}") 


        if(sub_channel_index != -1):
            h_sub_k = env.h_sub(device_positions, k, h_base_sub[k, sub_channel_index])
            # print(f"  h_sub-6GHz: {h_sub_k:.4f}") 
            r_sub[k] = env.r_sub(h_sub_k, device_index=k)
            # print(f"  Sub6 Calculated Rate r_sub[k]: {r_sub[k]}")
        if(mW_beam_index != -1):
            h_mW_k = env.h_mW(device_positions, k, frame)
            # print(f"  h_mW: {h_mW_k:.4f}") 
            r_mW[k] = env.r_mW(h_mW_k, device_index=k)
            # print(f"  mW Calculated Rate r_mW[k]: {r_mW[k]}")

        r.append(r_sub)
        r.append(r_mW)
    return r

#Compute number of success packets from AP each frame
def l_kv_success(r):
    l_kv_success = np.floor(np.multiply(r, T/D))
    return l_kv_success #sử dụng chung cho cả tính ước lượng và tính chính xác

#Compute average rate
def compute_average_rate(average_r, last_r, frame_num):
    avg_r = average_r.copy()
    for k in range(NUM_DEVICES):
        avg_r[0][k] = (last_r[0][k] + avg_r[0][k]*(frame_num - 1))/frame_num
        avg_r[1][k] = (last_r[1][k] + avg_r[1][k]*(frame_num - 1))/frame_num #toc do trung binh de uoc luong so goi tin gui

    return avg_r


#Xay dung ham phan bo cac goi tin tu hanh dong cua AP lua chon kenh truyen
def allocate(action):
    sub =[]
    mW = []
    for i in range(NUM_DEVICES):
        sub.append(-1)
        mW.append(-1)

    rand_sub = []
    rand_mW = []
    for i in range(NUM_SUBCHANNELS):
        rand_sub.append(i)     #danh sach chi so kenh 
    for i in range(NUM_BEAMS):
        rand_mW.append(i)     #danh sach chi so tia              
    
    for k in range(NUM_DEVICES):
        if(action[k] == 0):
            rand_index = np.random.randint(len(rand_sub))
            sub[k] = rand_sub[rand_index]
            rand_sub.pop(rand_index) #xoa kenh da chon ra khoi danh sach
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
    return allocate #trả về danh sách kênh, số thứ tự kênh thứ bao nhiêu (mỗi interface có 4 kênh) được chọn tương ứng cho device k đó 


# AP quyet dinh so goi tin truyen cho device
def perform_action(action, l_sub_max, l_mW_max): # cac goi nay chinh la goi tin uoc luong l_sub_max_estimate, l_mW_max_estimate vong lap for o model
    number_of_packet = np.zeros(shape=(NUM_DEVICES, 2))
    for k in range(NUM_DEVICES):
        l_sub_max_k = l_sub_max[k]
        l_mW_max_k = l_mW_max[k]
        if(action[k] == 0):
            number_of_packet[k, 0] = min(l_sub_max_k, MAX_PACKETS) #lấy giá trị nhỏ hơn giữa giá trị ước lượng với số gói tin tối đa L_k = 6
            number_of_packet[k, 1] = 0
        if(action[k] == 1):
            number_of_packet[k, 0] = 0
            number_of_packet[k, 1] = min(l_mW_max_k, MAX_PACKETS)
        if(action[k] == 2):
            if(l_mW_max_k < MAX_PACKETS):
                number_of_packet[k, 1] = min(l_mW_max_k, MAX_PACKETS)
                number_of_packet[k, 0] = min(l_sub_max_k, MAX_PACKETS - number_of_packet[k, 1])
            else:
                # number_of_packet[k, 1] = MAX_PACKETS
                # number_of_packet[k, 0] = 0
                number_of_packet[k, 1] = min(l_mW_max_k, MAX_PACKETS)
                number_of_packet[k, 0] = MAX_PACKETS - 1
    return number_of_packet #quyet dinh so goi tin gui ti tu AP toi device

# Xay dung ham nhan phan hoi ACK/NACK
def receive_feedback(packet_send, l_sub_max, l_mW_max): #l_sub_max, l_mW_max chinh la cac gia tri goi tin nhan duoc tai device
    feedback = np.zeros(shape=(NUM_DEVICES, 2))

    for k in range(NUM_DEVICES):
        l_sub_k = packet_send[k, 0]
        l_mW_k = packet_send[k, 1]

        feedback[k, 0] = min(l_sub_k, l_sub_max[k])
        feedback[k, 1] = min(l_mW_k, l_mW_max[k])

        # feedback[k, 0] = l_sub_max[k]
        # feedback[k, 1] = l_mW_max[k]
    return feedback

#Xay dung ham tinh ti le nhan goi tin that bai
def compute_packet_loss_rate(frame_num, old_packet_loss_rate, received_paket_num, sent_packet_num):
    plr = np.zeros(shape=(NUM_DEVICES, 2))
    for k in range(NUM_DEVICES):
        plr[k, 0] = env.packet_loss_rate(frame_num, old_packet_loss_rate[k, 0], received_paket_num[k, 0], sent_packet_num[k, 0])
        plr[k, 1] = env.packet_loss_rate(frame_num, old_packet_loss_rate[k, 1], received_paket_num[k, 1], sent_packet_num[k, 1])
        
    return plr

#CREATE REWARD
#Initialize reward
def initialize_reward(state, action):
    reward = {}
    return reward

#Update reward
def update_reward(state, action, old_reward_table, num_of_send_packet, num_of_received_packet, frame_num):
    state_action = np.insert(state, 4, action, axis=1)
    state_action = tuple([tuple(row) for row in state_action])
    old_reward_table = 0
    if(state_action in old_reward_table):
        old_reward_table = old_reward_table[state_action]
    reward = compute_reward(state, num_of_send_packet, num_of_received_packet, old_reward_table, frame_num)
    old_reward_table.update({state_action: reward})
    return old_reward_table

#Compute reward
# def compute_reward(state, num_of_send_packet, num_of_received_packet, old_reward_value, frame_num):
#     sum = 0
#     for k in range(NUM_DEVICES):
#         state_k = state[k]
#         sum = sum +((num_of_received_packet[k, 0] + num_of_received_packet[k, 1])/(
#             num_of_send_packet[k, 0] + num_of_send_packet[k, 1])) - (1 - state_k[0]) - (1 - state_k[1])
#     sum = ((frame_num - 1)*old_reward_value + sum)/frame_num
#     return sum
def compute_reward(state, num_of_send_packet, num_of_received_packet, old_reward_value, frame_num):
    sum = 0
    for k in range(NUM_DEVICES):
        # plr_state = plr_state[k]
        state_k = state[k]
        numerator = num_of_received_packet[k, 0] + num_of_received_packet[k, 1]
        denominator = num_of_send_packet[k, 0] + num_of_send_packet[k, 1]

        # !!! Vấn đề tiềm ẩn: Chia cho 0 !!!
        if denominator == 0:
             # Giả sử là 0.0 để phản ánh không có dữ liệu truyền đi.
             success_rate_k = 0.0
        else:
             success_rate_k = numerator / denominator

        plr_penalty_sub = (1 - state_k[0]) # Phạt = 1 nếu PLR Sub6 tệ (state[0]=0)
        plr_penalty_mW = (1 - state_k[1])  # Phạt = 1 nếu PLR mW tệ (state[1]=0)

        sum = sum + success_rate_k - plr_penalty_sub - plr_penalty_mW
        # 'sum' bây giờ chứa điểm số tức thời của frame này
    reward = ((frame_num - 1) * old_reward_value + sum) / frame_num
    # old_reward_value là giá trị trung bình được tính ở frame trước (frame_num - 1)

    return reward
#######################
#CREATE MODEL
# Tạo I bảng Q riêng biệt 
def initialize_Q_tables(first_state):
    Q_tables = []
    first_state = tuple([tuple(row) for row in first_state])
    for i in range(I):
        Q = {}
        add_new_state_to_table(Q, first_state)
        Q_tables.append(Q)
    return Q_tables

#create 2 Q_tables
# def sum_2_Q_tables(Q1, Q2): #tổng 2 state của 2 tables
#     q = Q1.copy()
#     for state in Q2:
#         if(state in q): #nếu trạng thái đó đã có ở Q1
#             for a in q[state]:
#                 q[state][a] += Q2[state][a] #cộng lại
#         else:
#             q.update({state: Q2[state].copy()}) #chưa có thì copy trạng thái từ Q2 sang
#     return q
def sum_2_Q_tables(Q1, Q2):  # tổng 2 state của 2 tables
    q = Q1.copy()
    for state in Q2:
        if state in q:  # nếu trạng thái đó đã có ở Q1
            for a in Q2[state]:  # Lặp qua các hành động trong Q2[state]
                if a not in q[state]:  # Kiểm tra xem hành động có trong q[state] chưa
                    q[state][a] = 0  # Nếu chưa, khởi tạo giá trị bằng 0
                q[state][a] += Q2[state][a]  # Sau đó cộng lại
        else:
            q.update({state: Q2[state].copy()})  # chưa có thì copy trạng thái từ Q2 sang
    return q

#Create average Q table function
# def average_Q_table(Q_tables):
#     res = {}
#     for state in range(len(Q_tables)):
#         res = add_2_Q_tables(res, Q_tables[state])
#     for state in res: 
#         for action in res[state]:
#             res[state][action] = res[state][action]/I
#     return res

def average_Q_table(Q_tables):
    res = {} #khởi tạo 1 dict rỗng
    for Q in Q_tables:
        for state in Q:
            # Tạo 1 Q_table chỉ có state
            Q_single_state = {state: Q[state]}
            # Cộng vào res
            res = sum_2_Q_tables(res, Q_single_state)

    # Sau khi cộng xong, chia trung bình
    for state in res: 
        for action in res[state]:
            res[state][action] /= I

    return res


def u(x):
    return -np.exp(BETA*x)

#Creata compute risk averge Q_tables function (tính bảng Q rủi ro)
def compute_risk_averse_Q(Q_tables, random_Q_index): 
    Q_random = Q_tables[random_Q_index].copy()
    Q_average = average_Q_table(Q_tables)
    sum_sqr = {}
    minus_Q_average = {}
    for state in Q_average:
        for action in Q_average[state]:
            Q_average[state][action] = -Q_average[state][action]
        minus_Q_average.update({state: Q_average[state].copy()}) #chuyển sang số đối của Q_average để tính Qi - Qavg 
    
    for i in range(I):
        sub = {}
        sub = sum_2_Q_tables(sub, Q_tables[i])
        sub = sum_2_Q_tables(sub, minus_Q_average) #tổng phương sai
        for state in sub:
            for action in sub[state]:
                sub[state][action] *= sub[state][action]
        sum_sqr = sum_2_Q_tables(sum_sqr, sub)
    
    for state in sum_sqr:
        for action in sum_sqr[state]:
            sum_sqr[state][action] = -(sum_sqr[state][action]*LAMBDA_P)/(I-1) #he so ne tranh rui ro

    res = sum_2_Q_tables({}, sum_sqr)
    # res = sum_2_Q_tables(res, Q_random)
    res = sum_2_Q_tables(Q_random, res)
    return res

# def add_new_state_to_table(table, state):
#     state = tuple([tuple(row) for row in state])
#     actions = {}
#     for i in range(3):
#         a = np.empty(NUM_DEVICES)
#         a[0] = i 
#         for j in range(3):
#             a[1] = j
#             for k in range(3):
#                 a[2] = k 
#                 actions.update({tuple(a.copy()):0})
#         table.update({state: actions})
#     return table
def add_new_state_to_table(table, state):
    state = tuple([tuple(row) for row in state])
    if state not in table:
        table[state] = {}  # Khởi tạo với từ điển hành động rỗng
    return table

COUNT = 0
# Create update Q_table function
# def update_Q_table(Q_table, alpha, reward, state, action, next_state):
#     state = tuple([tuple(row) for row in state])
#     action = tuple(action)
#     next_state = tuple([tuple(row) for row in next_state])

#     #Find max(Q(s(t+1), a))
#     max_Q = 0
#     for a in Q_table[state]:
#         if(Q_table[state][a] > max_Q):
#             max_Q = Q_table[state][a]
#     if(Q_table[state][action] != 0):
#         global COUNT
#         COUNT -= 1
#     Q_table[state][action] = Q_table[state][action] + alpha[state][action] * (u(reward + GAMMA * max_Q - Q_table[state][action]) - X0)
#     if(Q_table[state][action] != 0):
#         COUNT += 1
#     return Q_table
def update_Q_table(Q_table, alpha, reward, state, action, next_state):
    state = tuple([tuple(row) for row in state])
    action = tuple(action)
    next_state = tuple([tuple(row) for row in next_state])

    # Đảm bảo trạng thái tồn tại trong Q_table
    if state not in Q_table:
        Q_table[state] = {}
    if next_state not in Q_table:
        Q_table[next_state] = {}
    
    # Đảm bảo hành động tồn tại trong từ điển hành động của trạng thái
    if action not in Q_table[state]:
        Q_table[state][action] = 0
    
    # Tìm giá trị Q tối đa cho trạng thái tiếp theo
    max_Q = 0
    for a in Q_table[next_state]:
        if Q_table[next_state][a] > max_Q:
            max_Q = Q_table[next_state][a]
    
    # Theo dõi giá trị khác không bằng COUNT
    if Q_table[state][action] != 0:
        global COUNT
        COUNT -= 1
    
    # Cập nhật giá trị Q
    Q_table[state][action] = Q_table[state][action] + alpha[state][action] * (u(reward + GAMMA * max_Q - Q_table[state][action]) - X0)
    
    if Q_table[state][action] != 0:
        COUNT += 1
    
    return Q_table

#Khoi tao bang V
def initialize_V(first_state):
    V_tables = []
    for i in range(I):
        V =  {}
        add_new_state_to_table(V, first_state)
        V_tables.append(V)
    return V_tables

# def update_V(V, state, action):
#     state = tuple([tuple(row) for row in state])
#     action = tuple(action)
#     if(state in V): #nếu cặp trạng thái - hành động đó đã có ở V tăng giá trị V[state][action] lên 1
#         V[state][action] += 1
#     else:
#         add_new_state_to_table(V, state)
#         V[state][action] = 1

#     return V
def update_V(V, state, action):
    state = tuple([tuple(row) for row in state])
    action = tuple(action)
    
    # Kiểm tra trạng thái có tồn tại trong V không
    if state not in V:
        V[state] = {}
    
    # Kiểm tra hành động có tồn tại cho trạng thái này không
    if action not in V[state]:
        V[state][action] = 0
    
    # Tăng số lần truy cập
    V[state][action] += 1
    
    return V

#Khoi tao he so alpha
def initialize_alpha(first_state):
    return initialize_V(first_state)

# def update_alpha(alpha, V, state, action):
#     state = tuple([tuple(row) for row in state])
#     action = tuple(action)
#     if(state in alpha): #giảm tỉ lệ học cặp trạng thái - hành động đó
#         alpha[state][action] = 1/V[state][action]
#     else:
#         add_new_state_to_table(alpha, state)
#         alpha[state][action] = 1/V[state][action]
#     return alpha
def update_alpha(alpha, V, state, action):
    state = tuple([tuple(row) for row in state])
    action = tuple(action)
    
    # Đảm bảo trạng thái tồn tại trong alpha
    if state not in alpha:
        alpha[state] = {}
    
    # Đảm bảo hành động tồn tại cho trạng thái này
    if action not in alpha[state]:
        alpha[state][action] = 1.0  # Giá trị mặc định
    
    # Cập nhật alpha dựa trên số lần truy cập
    if state in V and action in V[state]:
        alpha[state][action] = 1.0 / V[state][action]  # Tỷ lệ học giảm khi truy cập nhiều hơn
    
    return alpha


########## TRAINING ############
device_positions = env.initialize_pos_of_devices()
# print(f"    device_positions: {device_positions}") 
state = initialize_state()
action = initialize_action()
reward = initialize_reward(state, action)
reward_value = 0.0
allocation = allocate(action)
Q_tables = initialize_Q_tables(state)
V = initialize_V(state)
alpha = initialize_alpha(state)
packet_loss_rate = np.zeros(shape=(NUM_DEVICES, 2))


#Generate h_base for each frame (100000)
h_base = create_h_base(NUM_OF_FRAME + 1)
h_base_t = h_base[0]
average_r = compute_r(device_positions, h_base_t, allocation=allocate(action),frame=1)

# state_plot=[]
# action_plot=[]
reward_plot=[]
# number_of_sent_packet_plot=[]
# number_of_received_packet_plot=[]
packet_loss_rate_plot=[]
rate_plot=[] #giá trị vận tốc

for frame in range(1, NUM_OF_FRAME + 1):
    # Random Q-table
    H = np.random.randint(0, I) #chọn ngẫu nhiên giá trị H từ 1 đến I (index Q = H-1)
    risk_adverse_Q = compute_risk_averse_Q(Q_tables, H)

    # Update EPSILON
    EPSILON = EPSILON * LAMBDA

    # Set up environment
    h_base_t = h_base[frame]
    h_base_sub_t = h_base_t[0] #giá trị này là giá trị đang ở số phức (của interface sub-6GHz)
    # print(f"Frame {frame}: h_base_sub_t = {h_base_t[0]}")  # In hệ số kênh Sub-6GHz
    # print(f"Frame {frame}: h_base_mW_t = {h_base_t[1]}")
    # state_plot.append(state)

    # Select action
    action = choose_action(state, risk_adverse_Q)
    allocation = allocate(action)
    # action_plot.append(action)

    # Perform action
    l_max_estimate = l_kv_success(average_r) #sử dụng hàm tính toán gói tin nhận được --> ước lượng gói tin gửi đi từ AP
    l_sub_max_estimate = l_max_estimate[0]
    l_mW_max_estimate = l_max_estimate[1]
    number_of_send_packet = perform_action(action, l_sub_max_estimate, l_mW_max_estimate)
    # number_of_sent_packet_plot.append(number_of_send_packet)

    

    # Get feedback
    r = compute_r(device_positions, h_base_t, allocation, frame) #vận tốc tại device
    rate_plot.append(r) #giá trị vận tốc

    print(f"Frame {frame}: Calculated Rate r at device = {r}")

    l_max = l_kv_success(r) #gói tin nhận được thành công
    print(f"Tong so goi tin nhan duoc thanh cong {l_max} tai frame {frame}")
    l_sub_max = l_max[0]
    print(f"So goi tin l_sub_max nhan duoc thanh cong {l_sub_max} tai frame {frame}")
    l_mW_max = l_max[1]
    print(f"So goi tin l_mW_max nhan duoc thanh cong {l_mW_max} tai frame {frame}")
    # rate_plot.append(r)

    number_of_received_packet = receive_feedback(number_of_send_packet, l_sub_max, l_mW_max)
    # number_of_received_packet = receive_feedback(number_of_send_packet, l_sub_max, l_mW_max)
    print(f"number_of_received_packet {number_of_received_packet} tai frame {frame}")

    packet_loss_rate = compute_packet_loss_rate(frame, packet_loss_rate, number_of_received_packet, number_of_send_packet)
    print(f"packet_loss_rate {packet_loss_rate} tai frame {frame}")
    packet_loss_rate_plot.append(packet_loss_rate)

    # packet_loss_rate_plot.append(packet_loss_rate)
    # number_of_received_packet_plot.append(number_of_received_packet)
    average_r = compute_average_rate(average_r, r, frame) #tính toán giá trị vận tốc trung bình để ước lượng gói tin
    # Compute reward
    # reward = update_reward(state, action, reward,number_of_send_packet, number_of_received_packet, frame)
    reward_value = compute_reward(state,number_of_send_packet,number_of_received_packet,reward_value,frame)
    # reward_value = compute_reward(plr_state, number_of_send_packet, number_of_received_packet,reward_value,frame)
    # reward_plot.append(reward_value)
    next_state = update_state(state, packet_loss_rate, number_of_received_packet) #number_of_received_packet chính là feedback
    reward_plot.append(reward_value)
    # print(f"reward_plot {reward_plot}") # In reward của frame hiện tại
     # --- IN RA PHẦN THƯỞNG MỖI FRAME ---
    print(f"Frame {frame}: Reward = {reward_value}") # In reward của frame hiện tại
    # print(f"Gia tri cua bien G: {env.G}")

    # Generate mask J
    J = np.random.poisson(1, I)

    for i in range(I):
        next_state_tuple = tuple([tuple(row) for row in next_state]) #chuyển đổi từ numpy sang tuple của các tuple
        if (J[i] == 1):
            V[i] = update_V(V[i],state,action)
            alpha[i] = update_alpha(alpha[i], V[i],state,action)
            Q_tables[i] = update_Q_table(Q_tables[i], alpha[i], reward_value, state, action, next_state)
        if(not (next_state_tuple in Q_tables[i])):
            add_new_state_to_table(Q_tables[i], next_state_tuple)
    state = next_state

    print('frame: ',frame)

# print("COUNT: {COUNT}")
# Vẽ đồ thị
plt.figure(figsize=(12, 6))
plt.plot(range(1, NUM_OF_FRAME + 1), reward_plot, label='Reward theo frame', color='green')
# Thêm tiêu đề và nhãn trục
plt.title('Biểu đồ Reward theo từng Frame')
plt.xlabel('Frame')
plt.ylabel('Reward')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#vẽ biểu đồ plr cho từng thiết bị ứng với các giao diện mạng 
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


# import numpy as np
# import os
# from datetime import datetime

# # Thư mục lưu file
# save_dir = "results"
# os.makedirs(save_dir, exist_ok=True)

# # Tên file kèm ngày giờ
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# file_path = os.path.join(save_dir, f"Training_results_{timestamp}.npz")

# # Lưu nhiều mảng với định dạng gốc (giữ nguyên shape)
# np.savez(file_path,
#          reward_plot=reward_plot,
#          packet_loss_rate_plot=packet_loss_rate_plot,
#          rate_plot=rate_plot)

# print(f"✅ Đã lưu kết quả vào file: {file_path}")


