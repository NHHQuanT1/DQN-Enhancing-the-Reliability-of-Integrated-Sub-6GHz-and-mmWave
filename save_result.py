from datetime import datetime
import pickle
import main 
import os


save_dir = "results"  

# Tạo tên file có thời gian để dễ phân biệt
filename = f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
save_path = os.path.join(save_dir, filename)

save_data = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'device_positions': main.device_positions,
    'Q_tables': main.Q_tables,
    'V': main.V,
    'alpha': main.alpha,
    'packet_loss_rate': main.packet_loss_rate,
    'h_base': main.h_base
}

with open(save_path, 'wb') as f:
    pickle.dump(save_data, f)

with open(save_path, 'rb') as f:
    saved_data = pickle.load(f)

print("Dữ liệu được lưu vào:", saved_data['timestamp'])
