import numpy as np
import matplotlib.pyplot as plt
import environment as env

def plot_packet_loss_rate(device):
    plr = IO.load('packet_loss_rate')
    plrsub=[]
    plrmw=[]
    for i in range(len(plr)):
        plrsub.append(plr[i][device-1,0])
        plrmw.append(plr[i][device-1,1])
    import matplotlib.pyplot as plt
    plt.plot(plrsub,label='sub')
    plt.plot(plrmw,label='mw')
    plt.legend()
    plt.title(f'Packet loss rate of device {device}')
    plt.show()

def scatter_packet_loss_rate(device='all'):
    received = IO.load('number_of_received_packet')
    sent = IO.load('number_of_sent_packet')

    x = np.arange(len(received))

    if(device!= 'all'):
        plr = []
        for i in range(len(received)):
            plr.append(1-(received[i][device-1,0]+received[i][device-1,1])/(sent[i][device-1,0]+sent[i][device-1,1]))
            
        plt.scatter(x=x, y=plr)
        plt.title(f'Packet loss rate of device {device}')

    else:
        plr = np.zeros(shape=(env.NUM_OF_DEVICE,len(received)))
        for i in range(len(received)):
            for k in range(env.NUM_OF_DEVICE):
                plr[k,i]=(1-(received[i][k,0]+received[i][k,1])/(sent[i][k,0]+sent[i][k,1]))

        for k in range(env.NUM_OF_DEVICE):
            plt.scatter(x=x,y=plr[k],label = f'Device {k+1}')
        plt.title(f'Packet loss rate of all devices') 

    plt.xlabel('Frame')
    plt.ylabel('Packet loss rate')
    plt.legend()
    plt.show()

def plot_reward():
    reward = IO.load('reward')
    # p = []
    # for i in range(len(reward)):
    #     p.append(np.mean(reward[0:i]))
    plt.title('Reward')
    plt.xlabel('Frame')
    plt.ylabel('Reward')
    plt.plot(reward)
    plt.show()

def plot_position():
    ap_pos = env.AP_POSITION
    device_pos = IO.load('device_positions')
    plt.title("AP and devices Position")
    plt.scatter(ap_pos[0], ap_pos[1], color = 'r',marker = 's',label = 'AP')
    for i in range(len(device_pos)):
        if(i == 1 or i == 5):
            plt.scatter(1/2*device_pos[i][0]+1/2*ap_pos[0],1/2*device_pos[i][1]+1/2*ap_pos[1], color = 'black',label = 'Obstacle',marker = 'd')
        plt.scatter(device_pos[i][0],device_pos[i][1], color = 'b')
        plt.text(device_pos[i][0]-0.4,device_pos[i][1]+0.8,f"D{i+1}",fontsize=12)
        
    plt.xlim([-env.width/2,env.width/2])
    plt.ylim([-env.length/2,env.length/2])
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()

def scatter_action(device=1):
    action = IO.load('action')
    plot = []
    for i in range(len(action)):
        plot.append(action[i][device-1])

    fig,ax = plt.subplots()
    x = np.arange(len(action))
    ax.scatter(x,y=plot)
    ax.set_title(f'Action of device {device}')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Action')
    plt.show()

def plot_interface_usage():
    action = IO.load('action')
    usage = np.zeros(shape=(env.NUM_OF_DEVICE,3))
    
    for i in range(len(action)):
        for j in range(env.NUM_OF_DEVICE):
            usage[j][int(action[i][j])]+=1
    usage = np.divide(usage,len(action)/100)
    usage = usage.transpose()
    fig,ax = plt.subplots(layout='constrained')
    x = np.arange(env.NUM_OF_DEVICE)
    width = 0.2
    multiplier = 0
    interfaces = ['Sub-6GHz','mmWave','Both']
    labels = [f'Device {i+1}' for i in range(env.NUM_OF_DEVICE)]

    for u in usage:
        offset = width*multiplier
        rects = ax.bar(x+offset,u,width)
        ax.bar_label(rects,padding=3)
        multiplier+=1
    ax.set_ylabel('Ratio [%]')
    ax.set_xticks(x+width,labels)
    ax.set_title('Interface Usage')
    plt.legend(interfaces,loc='upper right',ncols=3)
    plt.show()
                
def plot_rate(device=1):
    rate = IO.load('rate')
    sub = []
    mW = []
    for i in range(len(rate)):
        if(i==0):
            sub.append(rate[i][0][device-1])
            mW.append(rate[i][1][device-1])

        else:
            sub.append(1/i*((i-1)*sub[i-1] + rate[i][0][device-1]))
            mW.append(1/i*((i-1)*mW[i-1] + rate[i][1][device-1]))

    plt.plot(sub,label='Rate over Sub6-GHz')
    plt.plot(mW,label='Rate over mmWave')
    plt.xlabel('Frame')
    plt.ylabel('Rate')
    plt.title(f'Average rate of device {device}')
    plt.legend()
    plt.show()

def plot_sum_rate():
    rate = IO.load('rate')
    sub = []
    mW = []
    avg_sub = []
    avg_mW = []
    for i in range(len(rate)):
        sumsub = 0
        summW = 0
        for k in range(env.NUM_OF_DEVICE):
            sumsub+= rate[i][0][k]
            summW+= rate[i][1][k]
        sumsub/=env.NUM_OF_DEVICE
        summW/=env.NUM_OF_DEVICE

        if(i==0):
            sub.append(sumsub)
            mW.append(summW)
        else:
            sub.append(1/i* (sub[i-1]*(i-1) + sumsub))
            mW.append(1/i* (mW[i-1]*(i-1) + summW))

    plt.plot(sub,label = 'Rate over Sub6-GHz')
    plt.plot(mW,label='Rate over mmWave')
    plt.xlabel('Frame')
    plt.ylabel('Rate')
    plt.title(f'Average sum rate of all device')
    plt.legend()
    plt.show()