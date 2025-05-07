import numpy as np
import os
from datetime import datetime

file_path = "/Users/hongquannguyenhiep/HQ7/HUST/2024.2/ĐATN/demo/implement_DQN/results/metrics_20250504_175126.npz"
data = np.load(file_path)
reward_plot = data['reward_plot']
packet_loss_rate_plot = data['packet_loss_rate_plot']
rate_plot = data['rate_plot']

# print(reward_plot.shape)
# print(packet_loss_rate_plot.shape)
# print(rate_plot.shape)

print("reward_plot (first 10 elements):", reward_plot[:10])
print("packet_loss_rate_plot (first 10 elements):", packet_loss_rate_plot[:10])
print("rate_plot (first 10 elements):", rate_plot[:10])
