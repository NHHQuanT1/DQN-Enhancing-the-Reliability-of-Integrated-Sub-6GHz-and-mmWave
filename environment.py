import numpy as np
import random
import matplotlib.pyplot as plt


# Thiết lập tham số
NUM_DEVICES = 3  # Số thiết bị (K=3, scenario 1)
NUM_SUBCHANNELS = 4  # Số subchannel Sub-6GHz (N)
NUM_BEAMS = 4  # Số beam mmWave (M)
MAX_PACKETS = 6  # Số gói tin tối đa mỗi frame (L_k(t))
PLR_MAX = 0.1  # Giới hạn PLR tối đa
NUM_ACTIONS = 3  # 3 hành động: 0 (Sub-6GHz), 1 (mmWave), 2 (cả hai)
STATE_SIZE = NUM_DEVICES * 4  # State: [u_sub, u_mw, omega_sub, omega_mw] cho mỗi thiết bị
GAMMA = 0.9  # Discount factor
# P_DBM = 5 #dbm
# SIGMA = -169 #dbm/Hz
# P = pow(10, P_DBM/10) * 1e-3
P_DBM  = pow(10, 5/10)*1e-3
SIGMA = pow(10, -169/10)*1e-3
I_SUB = I_MW = 0.0
W_SUB = 1e8/NUM_SUBCHANNELS
W_MW = 1e9
T = 1e-3
D = 8000
NUM_OF_FRAME = 10000
LOS_PATH_LOSS = np.random.normal(0, 5.8, NUM_OF_FRAME)
NLOS_PATH_LOSS = np.random.normal(0, 8.7, NUM_OF_FRAME)

AP_POSITION = (0, 0)

def distance_to_AP(pos_of_device):
    d = np.sqrt((pos_of_device[0] - AP_POSITION[0])**2 + (pos_of_device[1] - AP_POSITION[1])**2)/1000
    return d

def initialize_pos_of_devices():
    list_of_devices = []
    for i in range(NUM_DEVICES):
        #position device_1
        if(i == 0):
            x = 0
            y = 20
        #position device_2
        elif(i == 1):
            x = 20
            y = 0
        #position device_3
        elif(i == 2):
            x = -60
            y = -60
        #position other devices (from device_4 to device_10)
        elif(i == 3):
            x = -40
            y = -40

        elif(i == 4):
            x = 15
            y = -75

        elif(i == 5):
            x = -25
            y = -20

        elif(i == 6):
            x = -45
            y = 15

        elif(i == 7):
            x = 60
            y = 55

        elif(i == 8):
            x = 50
            y = 2

        elif(i == 9):
            x = 50
            y = -40
            
        elif(i == 10):
            x = -60
            y = 40
            
        elif(i == 11):
            x = 60
            y = 30

        else: 
            x = random.uniform(-80, 80)   
            y = random.uniform(-80, 80)   
        list_of_devices.append((x,y))        
    return list_of_devices

list_of_devices = initialize_pos_of_devices()

# def initialize_pos_of_devices(distribution="uniform"):
#     list_of_devices = []

#     for _ in range(NUM_DEVICES):
#         if distribution == "uniform":
#             x = random.uniform(-80, 80)
#             y = random.uniform(-80, 80)

#         elif distribution == "gaussian":
#             x = np.random.normal(loc=0.0, scale=30.0)
#             y = np.random.normal(loc=0.0, scale=30.0)
#             # Giới hạn tọa độ trong [-80, 80]
#             x = max(min(x, 80), -80)
#             y = max(min(y, 80), -80)

#         elif distribution == "poisson":
#             # Dịch từ [0, ∞) về [-80, 80] bằng cách lấy mean=40 và trừ 80
#             x = np.random.poisson(lam=40) - 80
#             y = np.random.poisson(lam=40) - 80
#             x = max(min(x, 80), -80)
#             y = max(min(y, 80), -80)

#         else:
#             raise ValueError("Unsupported distribution type. Use 'uniform', 'gaussian', or 'poisson'.")

#         list_of_devices.append((x, y))

#     return list_of_devices

# # Uniform
# # list_of_devices = initialize_pos_of_devices(distribution="uniform")

# # Gaussian
# # list_of_devices = initialize_pos_of_devices(distribution="gaussian")

# # Poisson
# list_of_devices = initialize_pos_of_devices(distribution="poisson")


#Caculator Path loss
#Path loss Sub_6GHz
def path_loss_sub(d):
    return 38.5 + 30*(np.log10(d * 1000))
#Los Path loss mmWave
def los_path_loss_mW(d, frame):
    shadowing = LOS_PATH_LOSS[frame - 1]
    # shadowing = LOS_PATH_LOSS[frame]
    return 61.4 + 20*(np.log10(d * 1000)) + shadowing
#NLos path loss mmWave
def nlos_path_loss_mW(d, frame):
    shadowing = NLOS_PATH_LOSS[frame - 1]
    # shadowing = NLOS_PATH_LOSS[frame]
    return 72 + 29.2*(np.log10(d * 1000)) + shadowing

#Gennerate coefficient h_base Raileigh 
def generate_h_base(mean, sigma, size):
    re = np.random.normal(mean, sigma, size)
    im = np.random.normal(mean, sigma, size)
    h_base = []
    for i in range(size):
        h_base.append(complex(re[i], im[i])/np.sqrt(2))
    return h_base #hệ số của phai mờ kênh Raileigh (đang là giá trị số phức)

#Creat h for sub-6GHz each device within frame_t
def h_sub(list_of_devices, device_index, h_base_sub):
    h = np.abs(h_base_sub * pow(10, -path_loss_sub(distance_to_AP(list_of_devices[device_index]))/20.0))**2
    return h

#Main transmit beam Gain G_b
def transmit_beam_gain(eta = 5*np.pi/180, beta = 0):
    epsilon = 0.1
    G = (2*np.pi - (2*np.pi - eta)*epsilon)/eta
    return G

#h for mmWave each device within frame_t
def h_mW(list_of_devices, device_index, frame, eta = 5*np.pi/180, beta = 0): #truyền vào vị trí các device, device k, frame
    #device blocked
    if(device_index == 1 or device_index == 5 or device_index == 12):
        path_loss = nlos_path_loss_mW(distance_to_AP(list_of_devices[device_index]), frame) # giá trị PL tại frame_num của beam device k
        h = transmit_beam_gain(eta, beta) * 1 * pow(10, -path_loss/10.0) * 0.1 # G_Rx^k=epsilon
    
    #other devices
    else: # G_Rx^k = G_b
        path_loss = los_path_loss_mW(distance_to_AP(list_of_devices[device_index]), frame)
        h = transmit_beam_gain(eta, beta) * 1 * pow(10, -path_loss/10.0) * transmit_beam_gain(eta, beta) # transmit_beam_gain(eta, beta) chinh la tinh gia tri G
    return h

#Caculator SINR of sub-6GHz
def compute_sinr_sub(h, device_index):
    power = h * P_DBM
    interference_plus_noise = I_SUB + W_SUB * SIGMA
    gamma_sub = power/interference_plus_noise
    return gamma_sub

#Caculator SINR of mmWave
def compute_sinr_mW(h, device_index):
    power = h * P_DBM
    interference_plus_noise = I_MW + W_MW * SIGMA
    gamma_mW = power/interference_plus_noise
    return gamma_mW

#Caculator r_sub
def r_sub(h, device_index):
    r_sub = W_SUB * np.log2(1 + compute_sinr_sub(h, device_index))
    return r_sub

#Caculator r_mW
def r_mW(h, device_index):
    r_mW = W_MW * np.log2(1 + compute_sinr_mW(h, device_index))
    return r_mW

#Caculator number of success packets device received
# def l_kv_success(r_sub, r_mW, action):
#     pass

#Caculator packet loss rate (l_kv: so goi tin quyet dinh gui boi AP)
def packet_loss_rate(t, old_packet_loss_rate, omega_kv, l_kv):
    if(l_kv == 0): #số gói tin gửi đi = 0
        packet_loss_rate = ((t-1)/t)*old_packet_loss_rate
        return packet_loss_rate
    elif(l_kv > 0):
        packet_loss_rate = (1/t)*((t-1)*old_packet_loss_rate + (1 - omega_kv/l_kv))
        return packet_loss_rate



# device_positions = initialize_pos_of_devices()

# h_base = gennerate_h_base(0,1,3)
# h_sub_v = h_sub(device_positions,0,h_base[0])
# print(h_sub_v)

print('vi tri cua device', list_of_devices)