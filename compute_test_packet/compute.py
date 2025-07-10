import numpy as np

with open("send.txt") as f:
    raw_data = f.read()

# Bỏ phần "number_of_send_packet:" nếu có
if "number_of_send_packet:" in raw_data:
    raw_data = raw_data.split("number_of_send_packet:")[1].strip()

# Gán array = np.array để tránh lỗi
array = np.array

# Thực thi biểu thức
number_of_send_packet = eval(raw_data)

# ✅ Giờ bạn có thể xử lý tiếp như bình thường:
summed_array = np.sum(number_of_send_packet, axis=0)
row_sums = np.sum(summed_array, axis=1)
print(row_sums)
