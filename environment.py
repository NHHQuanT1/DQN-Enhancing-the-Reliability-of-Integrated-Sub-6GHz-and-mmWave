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
BATCH_SIZE = 16
GAMMA = 0.9  # Discount factor
EPS_START = 0.5  # Khởi đầu epsilon
EPS_END = 0.05  # Kết thúc epsilon
EPS_DECAY = 0.995  # Decay factor
TARGET_UPDATE = 10  # Cập nhật mạng target mỗi 10 bước
MEMORY_SIZE = 10000  # Kích thước bộ nhớ replay
NUM_EPISODES = 1  # Số episode huấn luyện

P_DBM = 5 #dbm
# P_DBM  = pow(10, 5/10)*1e-3
# P = pow(10, P_DBM/10) * 1e-3
SIGMA = -169 #dbm
# SIGMA = pow(10, -169/10)*1e-3
I_SUB = I_MW = 0
W_SUB = 1e8/NUM_SUBCHANNELS
W_MW = 1e9
T = 1e-3
D = 40000
NUM_OF_FRAME = 10000
LOS_PATH_LOSS = np.random.normal(0, 5.8, NUM_OF_FRAME)
NLOS_PATH_LOSS = np.random.normal(0, 8.7, NUM_OF_FRAME)

AP_POSITION = (0, 0)

def distance_to_AP(pos_of_device):
    d = np.sqrt((pos_of_device[0] - AP_POSITION[0])**2 + (pos_of_device[1] - AP_POSITION[1])**2)
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
            x = -55
            y = -60
        #position other devices
        else: 
            x = random.uniform(-80, 80)   
            y = random.uniform(-80, 80)   
        list_of_devices.append((x,y))        
    return list_of_devices

list_of_devices = initialize_pos_of_devices()

#Caculator Path loss
#Path loss Sub_6GHz
def path_loss_sub(d):
    return 38.5 + 30*(np.log10(d))
#Los Path loss mmWave
def los_path_loss_mW(d, frame):
    shadowing = LOS_PATH_LOSS[frame - 1]
    return 61.4 + 20*(np.log10(d)) + shadowing
#NLos path loss mmWave
def nlos_path_loss_mW(d, frame):
    shadowing = NLOS_PATH_LOSS[frame - 1]
    return 72 + 29.2*(np.log10(d)) + shadowing

#Gennerate coefficient h_base Raileigh
def gennerate_h_base(mean, sigma, size):
    re = np.random.normal(mean, sigma, size)
    im = np.random.normal(mean, sigma, size)
    h_base = []
    for i in range(size):
        h_base.append(complex(re[i], im[i])/np.sqrt(2))
    return h_base

#Creat h for sub-6GHz each device within frame_t
def h_sub(list_of_devices, device_index, h_base):
    h = np.abs(h_base * pow(10, -path_loss_sub(distance_to_AP(list_of_devices[device_index]))/20.0)**2)
    return h

#Main transmit beam Gain G_b
def transmit_beam_gain(eta = 5*np.pi/180, beta = 0):
    epsilon = 0.1
    G = (2*np.pi - (2*np.pi - eta)*epsilon)/eta
    return G

#h for mmWave each device within frame_t
def h_mW(list_of_devices, device_index, h_base, frame, G):
    #device blocked
    if(device_index == 1):
        path_loss = nlos_path_loss_mW(distance_to_AP(list_of_devices[device_index]), frame)
        h = G * (h_base * pow(10, -path_loss/20.0)) * G
    
    #other devices
    else:
        path_loss = los_path_loss_mW(distance_to_AP(list_of_devices[device_index]), frame)
        h = G * (h_base * pow(10, -path_loss/20)) * G
    return h

#Caculator SINR of sub-6GHz
def sinr_sub(h, device_index):
    power = h * P_DBM
    interference_plus_noise = I_SUB + W_SUB * SIGMA
    gamma_sub = power/interference_plus_noise
    return gamma_sub

#Caculator SINR of mmWave
def sinr_mW(h, device_index):
    power = h * P_DBM
    interference_plus_noise = I_MW + W_MW * SIGMA
    gamma_mW = power/interference_plus_noise
    return gamma_mW

#Caculator r_sub
def r_sub(gamma_sub, device_index):
    r_sub = W_SUB * np.log2(1 + sinr_sub(gamma_sub, device_index))
    return r_sub

#Caculator r_mW
def r_mW(gamma_mW, device_index):
    r_mW = W_MW * np.log2(1 + sinr_mW(gamma_mW, device_index))
    return r_mW

#Caculator number of success packets device received
# def l_kv_success(r_sub, r_mW, action):
#     pass

#Caculator packet loss rate (l_kv: so goi tin quyet dinh gui boi AP)
def packet_loss_rate(t, old_packet_loss_rate, omega_kv, l_kv):
    if(l_kv == 0):
        packet_loss_rate = ((t-1)/t)*old_packet_loss_rate
        return packet_loss_rate
    elif(l_kv > 0):
        packet_loss_rate = (1/t)*((t-1)*old_packet_loss_rate + (1 - omega_kv/l_kv))
        return packet_loss_rate





