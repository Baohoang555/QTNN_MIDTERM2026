# 🎲 Mô Phỏng Monte Carlo – Ví dụ 3.5
Xấp xỉ xác suất bằng phương pháp **Mô phỏng Monte Carlo cổ điển** — triển khai bằng Python cho Ví dụ 3.5 trong giáo trình *Mô phỏng Monte Carlo*.

---

## 📌 Bài toán

| Câu | Yêu cầu | Phân phối |
|-----|---------|-----------|
| 1 | Tính P(−3 ≤ Z ≤ 1) | Z ~ N(μ = −2, σ² = 4) |
| 2 | Tính P(N ≥ 3) | N ~ Poi(λ = 2.5) |

---

## 🔢 Cơ sở lý thuyết

Phương pháp dựa trên **Luật số lớn**: xác suất của một biến cố A bằng kỳ vọng của hàm chỉ tiêu tương ứng:

$$\mathbb{P}(X \in A) = \mathbb{E}[\mathbf{1}_A(X)] \approx \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}_A(X_i)$$

Các bước thực hiện:

1. **Sinh mẫu** — Tạo *n* biến ngẫu nhiên độc lập theo đúng phân phối đề bài.
2. **Áp dụng hàm chỉ tiêu** — Kiểm tra mỗi giá trị có thuộc biến cố A hay không (trả về 0 hoặc 1).
3. **Tính tần suất** — Lấy trung bình mảng chỉ tiêu → xấp xỉ xác suất.
4. **Đánh giá sai số** — Tính sai số chuẩn để đo độ tin cậy của ước lượng.

---

## 🧠 Giải thích code

### Câu 1 — Phân phối chuẩn N(−2, 4)

```python
# scale = độ lệch chuẩn = sqrt(phương sai) = sqrt(4) = 2
z = np.random.normal(loc=-2, scale=2, size=n)

indicator_1 = (z >= -3) & (z <= 1)   # Hàm chỉ tiêu: 1_{[-3,1]}(z)
est_1 = np.mean(indicator_1)          # Xác suất xấp xỉ
se_1  = np.sqrt(est_1 * (1-est_1) / n)
```

> ⚠️ **Lưu ý:** Tham số `scale` trong NumPy nhận **độ lệch chuẩn**, không phải phương sai.
> Phương sai = 4 → `scale = √4 = 2`.

### Câu 2 — Phân phối Poisson Poi(2.5)

```python
N_poi = np.random.poisson(lam=2.5, size=n)

indicator_2 = (N_poi >= 3)            # Hàm chỉ tiêu: 1_{[3,∞)}(N)
est_2 = np.mean(indicator_2)
se_2  = np.sqrt(est_2 * (1-est_2) / n)
```

### Công thức sai số chuẩn

$$SE = \sqrt{\frac{\hat{p}(1 - \hat{p})}{n}}$$

SE càng nhỏ → ước lượng càng đáng tin cậy. Với n = 10⁵, SE thường dao động quanh **0.001**.

---


## 📊 Kết quả mẫu

```
--- Câu 1: Tính xác suất P(-3 <= Z <= 1) với Z ~ N(-2, 4) ---
Xác suất xấp xỉ: 0.53270
Sai số chuẩn:   0.00158

--- Câu 2: Tính xác suất P(N >= 3) với N ~ Poi(2.5) ---
Xác suất xấp xỉ: 0.45640
Sai số chuẩn:   0.00157
```
