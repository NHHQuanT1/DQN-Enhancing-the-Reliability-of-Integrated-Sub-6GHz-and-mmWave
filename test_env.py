import numpy as np
import random as rd
import matplotlib.pyplot as plt

# the considered space has a width of 150 meters, a length of 150 meters
length = 150
width = 150

# Number of APs
NUM_OF_AP = 1
# Number of Devices
NUM_OF_DEVICE = 3
# Number of Sub-6Ghz channels N and mmWave beam M
NUM_OF_SUB_CHANNEL = 4
NUM_OF_BEAM = 4
# Transmit Power P_sub = P_mW = P ~ 5dBm
P = pow(10, 5/10)*1e-3
# Noise Power sigma^2 ~ -169dBm/Hz
SIGMA_SQR = pow(10, -169/10)*1e-3
# Bandwidth Sub6-GHz = 100MHz, W_mW = 1GHz
# Bandwidth per subchannel W_sub = 100MHz/number of sub channel
W_SUB = 1e8/NUM_OF_SUB_CHANNEL
W_MW = 1e9
# Frame Duration T_s
T = 1e-3
# Packet size D = 8000 bit
D = 8000
# Number of frame
NUM_OF_FRAME = 10000
# LoS Path loss - mmWave
LOS_PATH_LOSS = np.random.normal(0,5.8,NUM_OF_FRAME)
# NLoS Path loss - mmWave
NLOS_PATH_LOSS = np.random.normal(0,8.7,NUM_OF_FRAME) 

# initialize position of AP.
# the AP was located at the central of the area
# the position of each AP is the constant
AP_POSITION = (0, 0)

# the function calculates the distance to the nearest AP
def distance_to_AP(pos_of_device):
    distance = np.sqrt((pos_of_device[0] - AP_POSITION[0])**2 + (pos_of_device[1] - AP_POSITION[1])**2)
    return distance

    
# initialize device's postion with random value
# after initializing any device's position, check the distance from that device to the nearest AP,
# if the distance is satisfied, store it into the array list_of_devices.
def initialize_devices_pos():
    list_of_devices = []

    for i in range (NUM_OF_DEVICE):
        # Distance from Device #1 to AP and Device #2 to AP is equal
        if(i==0):
            x = 0
            y = 20
        elif(i==1):
            x = 20
            y = 0
        
        # Distance from Device #3 to AP is greater than from Device #1 and #2 
        elif(i==2):
            x = -55
            y = -60

        else:
            x = rd.uniform(-length/2,length/2)
            y = rd.uniform(-width/2,width/2)

        list_of_devices.append((x,y))
    return list_of_devices


# The list contains positions of devices
list_of_devices = initialize_devices_pos()

# Path loss model
def path_loss_sub(distance):
    return 38.5 + 30*(np.log10(distance))


def path_loss_mW_los(distance,frame):
    X = LOS_PATH_LOSS[frame-1]
    return 61.4 + 20*(np.log10(distance))+X


def path_loss_mw_nlos(distance,frame):
    X = NLOS_PATH_LOSS[frame-1]
    return 72 + 29.2*(np.log10(distance))+X


# Main Transmit Beam Gain G_b
def G(eta=5*np.pi/180, beta=0):
    epsilon = 0.01
    return (2*np.pi-(2*np.pi-eta)*epsilon)/(eta)


# Channel coefficient h=h_tilde* 10^(-pathloss/20)
# h_tilde = (a + b*i)/sqrt(2)
# in which a and b is random value from a Normal distribution
#hàm tạo ra h_ngã theo mô hình phai mờ kênh lớn và nhỏ
def generate_h_tilde(mu, sigma, amount):
    re = np.random.normal(mu, sigma, amount)
    im = np.random.normal(mu, sigma, amount)
    h_tilde = []
    for i in range(amount):
        h_tilde.append(complex(re[i], im[i])/np.sqrt(2))
    return h_tilde


def compute_h_sub(list_of_devices, device_index, h_tilde):
    h = np.abs(h_tilde* pow(10, -path_loss_sub(distance_to_AP(list_of_devices[device_index]))/20.0))**2
    return h


def compute_h_mW(list_of_devices, device_index, h_tilde, frame,eta=5*np.pi/180, beta=0):
    h = 0
    # device blocked by obstacle
    if (device_index == 1 or device_index == 5):
        path_loss = path_loss_mw_nlos(distance_to_AP(list_of_devices[device_index]),frame)
        h = np.abs(G(eta, beta)*h_tilde* pow(10, -path_loss/20)*0.01)**2  # G_Rx^k=epsilon
    # device not blocked
    else:
        path_loss = path_loss_mW_los(distance_to_AP(list_of_devices[device_index]),frame)
        h = np.abs((G(eta, beta)**2)*h_tilde * pow(10, -path_loss/20))**2  # G_Rx^k = G_b

    return h


# gamma_sub(h,k,n) (t) is the Signal to Interference-plus-Noise Ratio (SINR) from AP to device k on subchannel n with channel coefficient h
def gamma_sub(h, device_index):
    power = h*P
    interference_plus_noise = W_SUB*SIGMA_SQR
    # for b in range(NUM_OF_AP):
    #     if(b!=AP_index):
    #         interference_plus_noise += pow(abs(h[device_index,channel_index]),2)*P
    return power/interference_plus_noise

# gamma_mW(k,m) (t) is the Signal to Interference-plus-Noise Ratio (SINR) from AP to device k on beam m with channel coeffiction h
def gamma_mW(h, device_index):
    power = h*P
    interference_plus_noise = W_MW*SIGMA_SQR
    # for b in range(NUM_OF_AP):
    #     if(b!=AP_index):
    #         interference_plus_noise += pow(abs(h[device_index,beam_index]),2)*P
    return power/interference_plus_noise


# achievable data rate r_bkf (t) for the link between
# AP b, device k and for application f using bandwidth Wf at scheduling frame t
def r_sub(h, device_index):
    return W_SUB*np.log2(1+gamma_sub(h, device_index))


def r_mW(h, device_index):
    return W_MW*np.log2(1+gamma_mW(h, device_index))


def packet_loss_rate(t, old_packet_loss_rate, omega_kv, l_kv):
    if (l_kv == 0):
        packet_loss_rate = ((t-1)/t)*old_packet_loss_rate
        return packet_loss_rate
    elif (l_kv > 0):
        packet_loss_rate = (1/t)*((t-1)*old_packet_loss_rate + (1-omega_kv/l_kv))
        return packet_loss_rate