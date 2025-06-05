from datetime import datetime
import pickle
import os
import test_05 as core
import numpy as np

# save_dir = "results"   #Tạo folder trước

# # Tạo tên file có thời gian để dễ phân biệt
# filename = f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
# save_path = os.path.join(save_dir, filename)

# save_data = {
#     'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#     'device_positions': main_01.device_positions,
#     'Q_tables': main_01.Q_tables,
#     'V': main_01.V,
#     'alpha': main_01.alpha,
#     'packet_loss_rate': main_01.packet_loss_rate,
#     'h_base': main_01.h_base
# }

# with open(save_path, 'wb') as f:
#     pickle.dump(save_data, f)

# with open(save_path, 'rb') as f:
#     saved_data = pickle.load(f)

# print("Dữ liệu được lưu vào:", saved_data['timestamp'])



def save_tunable_parameters_txt(I, NUM_DEVICES, tunable_parameters, save_dir='tunable_para_test_03'):
    # Tạo tên file dựa trên thời gian hiện tại
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{I}Q_{NUM_DEVICES}D_tunable_parameters_{current_time}.txt"

    # Đường dẫn đầy đủ
    full_path = os.path.join(save_dir, filename)

    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(save_dir, exist_ok=True)

    # Ghi vào file
    with open(full_path, 'w') as f:
        for key, value in tunable_parameters.items():
            f.write(f'{key}: {value}\n')

    print(f"[INFO] Tunable parameters saved to {full_path}")


#### HÀM LƯU GIÁ TRỊ H_BASE ####
def save_or_load_h_base(I, NUM_DEVICES, num_frames, storage_dir='data/h_base_storage'):
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(storage_dir, exist_ok=True)

    # Tạo tên file kèm thời gian
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    key_word = 'test_05'
    filename = f"h_base_{I}Q_{NUM_DEVICES}D_{key_word}.npz"
    full_path = os.path.join(storage_dir, filename)

    # Kiểm tra xem file đã tồn tại chưa
    if os.path.exists(full_path):
        # Tải h_base từ file
        with np.load(full_path, allow_pickle=True) as data:
            h_base = data['h_base'].tolist()
        print(f"[INFO] Đã tải h_base từ {full_path}")
    else:
        # Tạo mới h_base
        h_base = core.create_h_base(num_frames)
        # Lưu h_base vào file
        np.savez(full_path, h_base=np.array(h_base, dtype=object))
        print(f"[INFO] Đã tạo và lưu h_base vào {full_path}")

    return h_base
