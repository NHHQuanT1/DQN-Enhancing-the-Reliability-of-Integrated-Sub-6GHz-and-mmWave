from datetime import datetime
import pickle
import main 
import os

save_dir = "results"   #Tạo folder trước

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



def save_tunable_parameters_txt(tunable_parameters, save_dir='tunable_para_test_03'):
    # Tạo tên file dựa trên thời gian hiện tại
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"tunable_parameters_{current_time}.txt"

    # Đường dẫn đầy đủ
    full_path = os.path.join(save_dir, filename)

    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(save_dir, exist_ok=True)

    # Ghi vào file
    with open(full_path, 'w') as f:
        for key, value in tunable_parameters.items():
            f.write(f'{key}: {value}\n')

    print(f"[INFO] Hyperparameters saved to {full_path}")
