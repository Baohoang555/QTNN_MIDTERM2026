import numpy as np

# Số lần mô phỏng n = 10^5 cho cả 2 câu [cite: 1125, 1133]
n = 10**5

print("--- Câu 1: Tính xác suất P(-3 <= Z <= 1) với Z ~ N(-2, 4) ---")
# 1. Sinh n biến Z tuân theo N(-2, 4). Tham số scale là độ lệch chuẩn (căn của 4 = 2) [cite: 1120, 1127]
z = np.random.normal(loc=-2, scale=2, size=n)

# 2. Tạo hàm chỉ tiêu: kiểm tra các giá trị rơi vào đoạn [-3, 1] [cite: 1128]
indicator_1 = (z >= -3) & (z <= 1)

# 3. Tính tần suất (xấp xỉ xác suất) và sai số chuẩn [cite: 1129, 1130]
est_1 = np.mean(indicator_1)
se_1 = np.sqrt(est_1 * (1 - est_1) / n)

print(f"Xác suất xấp xỉ: {est_1:.5f}")
print(f"Sai số chuẩn: {se_1:.5f}\n")


print("--- Câu 2: Tính xác suất P(N >= 3) với N ~ Poi(2.5) ---")
# 1. Sinh n biến N_poi tuân theo phân phối Poisson với lambda = 2.5 [cite: 1132, 1134]
N_poi = np.random.poisson(lam=2.5, size=n)

# 2. Tạo hàm chỉ tiêu: kiểm tra điều kiện N >= 3 [cite: 1135]
indicator_2 = (N_poi >= 3)

# 3. Tính tần suất (xấp xỉ xác suất) và sai số chuẩn [cite: 1136, 1137]
est_2 = np.mean(indicator_2)
se_2 = np.sqrt(est_2 * (1 - est_2) / n)

print(f"Xác suất xấp xỉ: {est_2:.5f}")
print(f"Sai số chuẩn: {se_2:.5f}")
