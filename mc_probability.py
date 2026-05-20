import numpy as np
import matplotlib.pyplot as plt

# PHẦN 1: TÍNH TOÁN XẤP XỈ MONTE CARLO

# Cố định random seed để kết quả có thể lặp lại
np.random.seed(42)

# Số lượng mô phỏng n = 10^5
n = 10**5

# Sinh n biến ngẫu nhiên từ phân phối Exp(0.5)
# Lưu ý: Hàm exponential của numpy sử dụng tham số scale = 1/lambda
lam = 0.5
scale_val = 1 / lam
x = np.random.exponential(scale=scale_val, size=n)

# Hàm chỉ tiêu: Trả về True nếu 1 < X <= 4, ngược lại là False
indicator = (x > 1) & (x <= 4)

# Tính xấp xỉ xác suất (tỉ lệ các giá trị thỏa mãn điều kiện)
est = np.mean(indicator)

# Tính sai số chuẩn (Standard Error)
se = np.sqrt(est * (1 - est) / n)

# In kết quả ra màn hình bằng tiếng Việt
print(f"Xác suất xấp xỉ: {est:.5f}")
print(f"Sai số chuẩn: {se:.5f}")


# PHẦN 2: TRỰC QUAN HÓA DỮ LIỆU

# Tính toán xác suất tích lũy qua từng bước mô phỏng để vẽ đường hội tụ
cumulative_est = np.cumsum(indicator) / np.arange(1, n + 1)

# Khởi tạo khung vẽ biểu đồ lớn chứa 2 biểu đồ con
plt.figure(figsize=(14, 6))

# ---- Biểu đồ 1: Sự hội tụ của phương pháp Monte Carlo ----
plt.subplot(1, 2, 1)
plt.plot(cumulative_est, color='blue', alpha=0.7)

# Giá trị lý thuyết tính bằng công thức tích phân: e^(-0.5*1) - e^(-0.5*4) ≈ 0.47120
plt.axhline(y=0.47120, color='red', linestyle='--', label='Giá trị lý thuyết (~0.47120)')
plt.title('Sự hội tụ của xác suất theo số lượng mô phỏng (n)')
plt.xlabel('Số lượng mô phỏng (n)')
plt.ylabel('Xác suất xấp xỉ')
plt.legend()
plt.grid(True, alpha=0.3)

# ---- Biểu đồ 2: Phân phối Exp(0.5) và vùng xác suất cần tính ----
plt.subplot(1, 2, 2)
# Vẽ histogram của mẫu dữ liệu đã sinh
count, bins, ignored = plt.hist(x, bins=100, density=True, alpha=0.6, color='gray')

# KHẮC PHỤC LỖI: Tính điểm trung tâm của mỗi bin để tạo mảng có độ dài 100 (khớp với mảng count)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# Bôi màu vùng thỏa mãn điều kiện (1 < X <= 4) bằng bin_centers
plt.fill_between(bin_centers, 0, count, where=(bin_centers > 1) & (bin_centers <= 4), color='orange', alpha=0.7, label='Vùng P(1 < X <= 4)')

# Vẽ thêm đường cong hàm mật độ xác suất (PDF) lý thuyết để đối chiếu
# PDF của phân phối mũ: f(x) = lambda * e^(-lambda * x)
pdf_x = np.linspace(0, max(bins), 1000)
pdf_y = lam * np.exp(-lam * pdf_x)
plt.plot(pdf_x, pdf_y, color='black', linewidth=2, label='Hàm mật độ lý thuyết (PDF)')

plt.title('Phân phối Exp(0.5) và vùng xác suất P(1 < X <= 4)')
plt.xlabel('Giá trị X')
plt.ylabel('Mật độ xác suất')
plt.xlim(0, 10)  # Giới hạn trục x để dễ nhìn hơn
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()