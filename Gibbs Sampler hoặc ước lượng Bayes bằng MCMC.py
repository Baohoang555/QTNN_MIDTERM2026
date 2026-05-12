"""
MCMC TOÀN DIỆN: Gibbs Sampler + Metropolis-Hastings + Chẩn đoán hội tụ + Ước lượng Bayes
======================================================================================
Gồm 5 phần:
  1. Gibbs Sampler – Chuỗi nhị phân (bài toán gốc + chẩn đoán)
  2. Nhiều chain song song + Gelman-Rubin R̂  (công thức đúng)
  3. Autocorrelation & Effective Sample Size (ESS)
  4. So sánh Gibbs vs. Metropolis-Hastings   (ghi chú không gian khác nhau)
  5. Ước lượng Bayes: posterior Beta(θ|data) với prior Beta  (likelihood chuẩn hóa đúng)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
from math import comb as _comb, floor as _floor
from numpy import trapz
from scipy.stats import beta as beta_dist

# ─── FORMAT ────────────────────────────────────────────────────────────
np.random.seed(42)
plt.rcParams.update({
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False
})
COLORS = {
    'gibbs'     : '#2563EB',
    'mh'        : '#DC2626',
    'theory'    : '#D97706',
    'chain'     : ['#2563EB', '#16A34A', '#DC2626', '#9333EA'],
    'posterior' : '#0F6E56',
    'prior'     : '#888780',
    'likelihood': '#993C1D',
}


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 1 – GIBBS SAMPLER TRÊN CHUỖI NHỊ PHÂN
# ══════════════════════════════════════════════════════════════════════════════
#
# Bài toán: lấy mẫu đồng đều từ tập S = {x ∈ {0,1}^N : không có 1 kề nhau}.
#
# Phân phối mục tiêu: π đồng đều trên S, tức π(x) = 1/|S| với mọi x ∈ S.
#
# Gibbs Sampler: tại mỗi bước, chọn vị trí i ngẫu nhiên, cập nhật x_i
# theo phân phối điều kiện đầy đủ p(x_i | x_{-i}):
#
#   - Nếu x_{i-1}=1 HOẶC x_{i+1}=1  →  x_i = 0  (xác suất 1, bắt buộc)
#   - Ngược lại                       →  x_i = 0 hoặc 1 với xác suất 1/2
#
# Lý do: với π đồng đều, p(x_i=1 | x_{-i}) = 1/2 khi hàng xóm đều = 0,
# và = 0 khi có hàng xóm = 1 (để duy trì tính hợp lệ).
#
# Định lý Ergodic (Định lý 2.44 trong bài giảng) đảm bảo:
#   (1/M) Σ f(X_m) → E_π[f(X)]  h.c.c. khi M → ∞
#
# E[#số 1] CHÍNH XÁC:
#   Số chuỗi hợp lệ độ dài N có đúng k số 1 = C(N-k+1, k)
#   Tổng số chuỗi hợp lệ = F(N+2)  (số Fibonacci thứ N+2, F(1)=F(2)=1)
#   E[X] = Σ_{k=0}^{⌊(N+1)/2⌋} k · C(N-k+1, k) / F(N+2)
#
# Lưu ý: công thức xấp xỉ (N+1)/3 chỉ đúng khi N → ∞.
# ─────────────────────────────────────────────────────────────────────────────

N       = 100   # độ dài chuỗi
M       = 3000  # số mẫu Monte Carlo
BURN_IN = 500   # số sweep burn-in (mỗi sweep = N bước Gibbs)


def init_chain(N):
    """Khởi tạo chuỗi hợp lệ: 1,0,1,0,..."""
    x = np.zeros(N, dtype=int)
    x[::2] = 1
    return x


def gibbs_step(x):
    """Một bước Gibbs: chọn vị trí i ngẫu nhiên, cập nhật theo full conditional."""
    i     = np.random.randint(len(x))
    left  = (i > 0)        and (x[i - 1] == 1)
    right = (i < len(x)-1) and (x[i + 1] == 1)
    x[i]  = 0 if (left or right) else np.random.randint(2)
    return x


def _fibonacci_Np2(N):
    """
    Tính F(N+2) = số chuỗi nhị phân độ dài N không có 1 kề nhau.
    Dãy Fibonacci: F(1)=1, F(2)=1, F(3)=2, F(4)=3, F(5)=5, ...
    Có thể kiểm chứng: N=1 → F(3)=2 = {0,1} ✓
                       N=2 → F(4)=3 = {00,01,10} ✓
                       N=3 → F(5)=5 = {000,001,010,100,101} ✓
    """
    a, b = 1, 1                   # F(1), F(2)
    for _ in range(N):            # sau N vòng: a=F(N+1), b=F(N+2)
        a, b = b, a + b
    return b                      # = F(N+2)  ✓


def e_theory_exact(N):
    """
    E[#số 1] chính xác = Σ_{k=0}^{⌊(N+1)/2⌋} k · C(N-k+1, k) / F(N+2)
    """
    total   = _fibonacci_Np2(N)
    numerator = sum(
        k * _comb(N - k + 1, k)
        for k in range(_floor((N + 1) / 2) + 1)
    )
    return numerator / total


def run_gibbs(N, M, burn_in, seed=0):
    """Chạy Gibbs Sampler, trả về mảng M mẫu (số lượng 1 trong mỗi chuỗi)."""
    np.random.seed(seed)
    x = init_chain(N)
    # Burn-in
    for _ in range(burn_in * N):
        gibbs_step(x)
    # Thu thập mẫu (mỗi mẫu = sau 1 sweep = N bước Gibbs)
    samples = np.empty(M, dtype=int)
    for m in range(M):
        for _ in range(N):
            gibbs_step(x)
        samples[m] = x.sum()
    return samples


print("Đang chạy Phần 1 – Gibbs Sampler đơn...")
samples_gibbs = run_gibbs(N, M, BURN_IN, seed=42)

E_theory = e_theory_exact(N)
E_hat    = samples_gibbs.mean()
SE       = samples_gibbs.std() / np.sqrt(M)

print(f"  E_hat    = {E_hat:.4f}")
print(f"  Lý thuyết= {E_theory:.4f}  (xấp xỉ (N+1)/3 = {(N+1)/3:.4f})")
print(f"  SE       = {SE:.4f}")
print(f"  95% CI   : [{E_hat - 1.96*SE:.4f}, {E_hat + 1.96*SE:.4f}]")


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 2 – NHIỀU CHAIN SONG SONG + GELMAN-RUBIN R̂
# ══════════════════════════════════════════════════════════════════════════════
#
# Gelman-Rubin R̂ đo mức độ hội tụ bằng cách so sánh:
#   B = phương sai GIỮA các chain (between-chain variance)
#   W = phương sai TRONG từng chain (within-chain variance)
#
# Công thức chuẩn (Gelman & Rubin 1992, Brooks & Gelman 1998):
#
#   B = n/(m-1) · Σ_{j=1}^{m} (θ̄_j - θ̄)²        (B ≥ 0)
#   W = (1/m) · Σ_{j=1}^{m} s_j²                   (s_j² = var nội chain j)
#
#   V̂ = (n-1)/n · W  +  (m+1)/(m·n) · B
#
#   R̂ = sqrt(V̂ / W)
#
# Giải thích:
#   - Nếu tất cả chain đã hội tụ về cùng phân phối: B ≈ W → R̂ ≈ 1
#   - Nếu chain chưa mix: B >> W → R̂ >> 1
#   - Ngưỡng thực tế: R̂ < 1.1 → coi như đã hội tụ
#
# Hệ số (m+1)/(m·n) thay vì 1/n là điều chỉnh vì chúng ta ước lượng B
# từ m chain hữu hạn (không phải từ vô số chain lý tưởng).
# ─────────────────────────────────────────────────────────────────────────────

def gelman_rubin(chains):
    """
    Tính Gelman-Rubin R̂.
    chains: list of 1-D numpy arrays, mỗi array là 1 chain cùng độ dài.
    Trả về: R_hat, B (between), W (within)
    """
    m           = len(chains)
    n           = len(chains[0])
    chain_means = np.array([c.mean()      for c in chains])   # θ̄_j
    chain_vars  = np.array([c.var(ddof=1) for c in chains])   # s_j²
    grand_mean  = chain_means.mean()                           # θ̄

    B     = n / (m - 1) * np.sum((chain_means - grand_mean)**2)
    W     = chain_vars.mean()
    V_hat = (n - 1) / n * W  +  (m + 1) / (m * n) * B
    R_hat = np.sqrt(V_hat / W)
    return R_hat, B, W


print("\nĐang chạy Phần 2 – 4 chains song song...")
N_CHAINS = 4
chains = [run_gibbs(N, M, BURN_IN, seed=k * 100) for k in range(N_CHAINS)]

R_hat, B_GR, W_GR = gelman_rubin(chains)
print(f"  Gelman-Rubin R̂ = {R_hat:.4f}  "
      f"({'✓ hội tụ (< 1.1)' if R_hat < 1.1 else '✗ chưa hội tụ'})")


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 3 – AUTOCORRELATION & EFFECTIVE SAMPLE SIZE (ESS)
# ══════════════════════════════════════════════════════════════════════════════
#
# Mẫu MCMC không độc lập → phương sai thực tế lớn hơn phương sai i.i.d.
#
# Autocorrelation tại lag k:
#   ρ(k) = Cov(X_t, X_{t+k}) / Var(X_t)
#
# Effective Sample Size (Kass et al. 1998):
#   ESS = M / (1 + 2 Σ_{k=1}^{K} ρ(k))
#
# Quy tắc cắt (cutoff) — heuristic "initial positive sequence" (Geyer 1992):
#   Cộng dồn các cặp ρ(2k) + ρ(2k+1); dừng khi cặp đó âm lần đầu tiên.
#   Cách này ổn định hơn cắt tại lag đầu tiên ACF < 0.
# ─────────────────────────────────────────────────────────────────────────────

def autocorrelation(x, max_lag=50):
    """Tính ACF tại các lag 0, 1, ..., max_lag."""
    n   = len(x)
    xc  = x - x.mean()                           # center
    d   = np.dot(xc, xc)                         # Var không chuẩn hóa
    acf = np.array([
        np.dot(xc[:n - k], xc[k:]) / d
        for k in range(max_lag + 1)
    ])
    return acf


def ess(samples, max_lag=500):
    """
    Effective Sample Size theo công thức chuẩn.
    Dùng heuristic "initial positive sequence" (Geyer 1992) để chọn cutoff:
      - Xét các cặp Γ_k = ρ(2k) + ρ(2k+1)
      - Cộng dồn các Γ_k dương liên tiếp từ k=0; dừng khi Γ_k ≤ 0
    """
    n   = len(samples)
    lag = min(max_lag, n - 1)
    acf = autocorrelation(samples, max_lag=lag)   # acf[0]=1, acf[k]=ρ(k)

    # Geyer initial positive sequence
    rho_sum = 0.0
    for k in range(1, lag // 2 + 1):
        pair = acf[2*k - 1] + acf[2*k]           # Γ_k = ρ(2k-1) + ρ(2k)
        if pair <= 0:
            break
        rho_sum += pair

    ess_val = n / (1 + 2 * rho_sum)
    return max(ess_val, 1.0)


acf_vals = autocorrelation(samples_gibbs, max_lag=80)
ess_val  = ess(samples_gibbs)
print(f"\n  ESS Gibbs = {ess_val:.1f}  "
      f"(trên M={M} mẫu → efficiency = {ess_val/M*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 4 – SO SÁNH GIBBS VS. METROPOLIS-HASTINGS
# ══════════════════════════════════════════════════════════════════════════════
#
#  QUAN TRỌNG – HAI KHÔNG GIAN TRẠNG THÁI KHÁC NHAU:
#
#   Gibbs (P1): không gian S = {x ∈ {0,1}^N : không có 1 kề nhau}
#               → lấy mẫu chuỗi x đầy đủ, phân phối đồng đều trên S
#
#   MH   (P4): không gian K = {0, 1, ..., ⌊(N+1)/2⌋}
#               → lấy mẫu số nguyên k = #(số 1 trong chuỗi)
#               → phân phối mục tiêu π(k) ∝ C(N-k+1, k)
#                 (= số chuỗi hợp lệ có đúng k số 1)
#
#   Cả hai đều ước lượng E[#số 1] = E_π[k], nhưng:
#   - Gibbs cho phép phân tích toàn bộ chuỗi (vị trí từng số 1)
#   - MH nhanh hơn nhưng chỉ cho thông tin về tổng số 1
#   → So sánh ESS mang tính MINH HỌA THUẬT TOÁN, không hoàn toàn tương đương.
#
# MH (Metropolis thuần túy vì Q đối xứng):
#   Proposal: k' = k ± 1  (xác suất bằng nhau → Q đối xứng)
#   Hàm chấp nhận: a(k, k') = min{1, π(k')/π(k)} = min{1, C(N-k'+1,k')/C(N-k+1,k)}
# ─────────────────────────────────────────────────────────────────────────────

def log_pi_k(k, N):
    """
    log π(k) ∝ log C(N-k+1, k) = log(số chuỗi hợp lệ có đúng k số 1).
    Công thức: số cách xếp k số 1 vào N vị trí sao cho không có 2 số kề nhau
               = C(N-k+1, k)  (bài toán đặt k vật vào N-k+1 khe trống).
    """
    if k < 0 or k > _floor((N + 1) / 2):
        return -np.inf
    return np.log(_comb(N - k + 1, k) + 1e-300)


def run_mh_discrete(N, M, burn_in, seed=0):
    """
    MH trên không gian K = {0,...,⌊(N+1)/2⌋}.
    Proposal: k' = k + choice([-1, +1])  (đối xứng → Metropolis thuần túy).
    """
    np.random.seed(seed)
    k        = N // 4      # khởi tạo gần vùng xác suất cao
    samples  = np.empty(M, dtype=int)
    n_accept = 0

    for step in range(burn_in + M):
        k_prop    = k + np.random.choice([-1, 1])
        log_ratio = log_pi_k(k_prop, N) - log_pi_k(k, N)
        if np.log(np.random.rand()) < log_ratio:
            k = k_prop
            if step >= burn_in:
                n_accept += 1
        if step >= burn_in:
            samples[step - burn_in] = k

    accept_rate = n_accept / M
    return samples, accept_rate


print("\nĐang chạy Phần 4 – Metropolis-Hastings (trên không gian k)...")
samples_mh, mh_accept = run_mh_discrete(N, M, burn_in=BURN_IN, seed=42)
acf_mh  = autocorrelation(samples_mh, max_lag=80)
ess_mh  = ess(samples_mh)

print(f"  MH acceptance rate = {mh_accept*100:.1f}%")
print(f"  ESS MH = {ess_mh:.1f}  "
      f"(efficiency = {ess_mh/M*100:.1f}%)")
print(f"  [Lưu ý] MH lấy mẫu k ∈ ℤ; Gibbs lấy mẫu x ∈ {{0,1}}^N")
print(f"          So sánh ESS mang tính minh họa thuật toán.")


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 5 – ƯỚC LƯỢNG BAYES
# ══════════════════════════════════════════════════════════════════════════════
#
# Bài toán: tung đồng xu n lần, quan sát h lần ngửa.
# Muốn suy luận về xác suất ngửa θ ∈ (0,1).
#
# Mô hình:
#   Prior:      θ ~ Beta(α₀, β₀)
#   Likelihood: h | θ ~ Binomial(n, θ)
#   Posterior:  θ | h ~ Beta(α₀+h, β₀+n-h)   [conjugate prior]
#
# log-posterior (đến sai khác hằng số):
#   log p(θ|h) = (α₀+h-1)·log θ + (β₀+n-h-1)·log(1-θ) + const
#              = log-likelihood + log-prior
#
# MCMC dùng Metropolis-Hastings trên θ ∈ (0,1):
#   Proposal: θ' = θ + Normal(0, σ)
#   Hàm chấp nhận: a = min{1, p(θ'|h)/p(θ|h)} = min{1, exp(log_p(θ') - log_p(θ))}
#
# Ưu điểm quan trọng (Chú ý 3.13 bài giảng thầy):
#   MCMC KHÔNG CẦN hằng số chuẩn hóa C = ∫ likelihood × prior dθ
#   vì C triệt tiêu trong tỷ số p(θ'|h)/p(θ|h).
# ─────────────────────────────────────────────────────────────────────────────

print("\nĐang chạy Phần 5 – Ước lượng Bayes + MCMC...")

n_obs  = 50    # tổng lần tung
h_obs  = 32    # số lần ngửa
alpha0 = 2.0   # tham số prior
beta0  = 2.0

# Posterior giải tích (conjugate) — dùng để kiểm chứng
alpha_post     = alpha0 + h_obs
beta_post      = beta0  + (n_obs - h_obs)
theta_mean_exact = alpha_post / (alpha_post + beta_post)
theta_mode_exact = (alpha_post - 1) / (alpha_post + beta_post - 2)


def log_posterior(theta, h, n, a0, b0):
    """
    log p(θ|data) = log-likelihood + log-prior  (bỏ hằng số chuẩn hóa)
    Hợp lệ vì hằng số triệt tiêu trong tỷ số MH.
    """
    if theta <= 0 or theta >= 1:
        return -np.inf
    log_lik   = h * np.log(theta) + (n - h) * np.log(1 - theta)
    log_prior = (a0 - 1) * np.log(theta) + (b0 - 1) * np.log(1 - theta)
    return log_lik + log_prior


M_bayes    = 5000
burn_bayes = 500
sigma_prop = 0.05   # std của proposal Gaussian

np.random.seed(99)
theta_curr    = 0.5
theta_samples = np.empty(M_bayes)
accepts       = 0

for i in range(burn_bayes + M_bayes):
    theta_prop = theta_curr + np.random.normal(0, sigma_prop)
    log_r = (log_posterior(theta_prop, h_obs, n_obs, alpha0, beta0)
           - log_posterior(theta_curr, h_obs, n_obs, alpha0, beta0))
    if np.log(np.random.rand()) < log_r:
        theta_curr = theta_prop
        if i >= burn_bayes:
            accepts += 1
    if i >= burn_bayes:
        theta_samples[i - burn_bayes] = theta_curr

bayes_accept   = accepts / M_bayes
theta_mean_mcmc= theta_samples.mean()
theta_ci_lo, theta_ci_hi = np.percentile(theta_samples, [2.5, 97.5])

print(f"  E[θ|data] giải tích = {theta_mean_exact:.4f}")
print(f"  E[θ|data] MCMC      = {theta_mean_mcmc:.4f}")
print(f"  95% Credible Interval: [{theta_ci_lo:.4f}, {theta_ci_hi:.4f}]")
print(f"  MH acceptance rate  = {bayes_accept*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# VẼ TOÀN BỘ (5 hàng × 3 cột)
# ══════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(18, 26))
fig.suptitle(
    'MCMC Toàn Diện: Gibbs Sampler, Metropolis-Hastings & Ước Lượng Bayes',
    fontsize=16, fontweight='bold', y=0.995
)
gs   = gridspec.GridSpec(5, 3, figure=fig, hspace=0.55, wspace=0.35)
LAGS = np.arange(81)

# ─── Hàng 1: Phần 1 (Gibbs đơn) ─────────────────────────────────────────────

# 1a. Histogram phân phối số lượng 1
ax = fig.add_subplot(gs[0, 0])
freq = Counter(samples_gibbs)
keys = sorted(freq.keys())
vals = [freq[k] / M for k in keys]
ax.bar(keys, vals, color=COLORS['gibbs'], alpha=0.7, edgecolor='white', width=0.9)
ax.axvline(E_hat,    color='red',            lw=2, ls='--',
           label=f'MC ≈ {E_hat:.2f}')
ax.axvline(E_theory, color=COLORS['theory'], lw=2, ls=':',
           label=f'Lý thuyết = {E_theory:.2f}')
ax.set_xlabel('Số lượng chữ số 1')
ax.set_ylabel('Tần suất tương đối')
ax.set_title('[P1] Phân phối số lượng 1\n(Gibbs Sampler)')
ax.legend(fontsize=9)

# 1b. Hội tụ running mean  ── [S1] dùng running variance O(M) ──
ax = fig.add_subplot(gs[0, 1])
ns       = np.arange(1, M + 1)
cumsum   = np.cumsum(samples_gibbs)
cumsum2  = np.cumsum(samples_gibbs.astype(float)**2)
run_mean = cumsum / ns
run_var  = cumsum2 / ns - run_mean**2          # E[X²] - (E[X])²
run_se   = np.sqrt(np.maximum(run_var, 0) / ns)  # SE = σ/√n

ax.plot(run_mean, color=COLORS['gibbs'], lw=1.5, label='Running mean')
ax.axhline(E_theory, color=COLORS['theory'], lw=1.5, ls='--',
           label=f'Lý thuyết = {E_theory:.2f}')
ax.fill_between(range(M),
                run_mean - 1.96 * run_se,
                run_mean + 1.96 * run_se,
                alpha=0.2, color=COLORS['gibbs'], label='95% CI')
ax.set_xlabel('Số mẫu')
ax.set_ylabel('E[#1]')
ax.set_title('[P1] Hội tụ ước lượng Monte Carlo\n(running mean ± 1.96·SE)')
ax.legend(fontsize=9)

# 1c. Trace plot chuỗi đơn
ax = fig.add_subplot(gs[0, 2])
ax.plot(samples_gibbs[:500], color=COLORS['gibbs'], lw=0.8, alpha=0.8)
ax.axhline(E_theory, color=COLORS['theory'], lw=1.5, ls='--',
           label=f'Lý thuyết = {E_theory:.2f}')
ax.set_xlabel('Bước MCMC')
ax.set_ylabel('#số 1')
ax.set_title('[P1] Trace plot\n(500 mẫu đầu sau burn-in)')
ax.legend(fontsize=9)

# ─── Hàng 2: Phần 2 (Multiple chains + Gelman-Rubin) ────────────────────────

# 2a. Trace plot 4 chains
ax = fig.add_subplot(gs[1, 0:2])
for k in range(N_CHAINS):
    ax.plot(chains[k][:400], color=COLORS['chain'][k], lw=0.9,
            alpha=0.75, label=f'Chain {k+1}')
ax.axhline(E_theory, color=COLORS['theory'], lw=1.5, ls='--', label='Lý thuyết')
ax.set_xlabel('Số mẫu')
ax.set_ylabel('#số 1')
ax.set_title(f'[P2] 4 chains song song  |  R̂ = {R_hat:.4f} '
             f'({"< 1.1 → hội tụ ✓" if R_hat < 1.1 else "> 1.1 → chưa hội tụ"})')
ax.legend(fontsize=8, ncol=5)

# 2b. R̂ theo cỡ mẫu tăng dần
ax = fig.add_subplot(gs[1, 2])
checkpoints = np.arange(50, M + 1, 50)
rhat_vals   = [gelman_rubin([c[:cp] for c in chains])[0] for cp in checkpoints]
ax.plot(checkpoints, rhat_vals, color='#9333EA', lw=1.5)
ax.axhline(1.1, color='red',  lw=1.5, ls='--', label='Ngưỡng 1.1')
ax.axhline(1.0, color='gray', lw=1.0, ls=':')
ax.set_xlabel('Số mẫu')
ax.set_ylabel('R̂')
ax.set_title('[P2] Gelman-Rubin R̂\ntheo cỡ mẫu')
ax.legend(fontsize=9)
ax.set_ylim(bottom=0.95)

# ─── Hàng 3: Phần 3 (ACF + ESS) ─────────────────────────────────────────────

# 3a. ACF – Gibbs
ax = fig.add_subplot(gs[2, 0])
ax.bar(LAGS, acf_vals, color=COLORS['gibbs'], alpha=0.7, width=0.8)
ax.axhline(0,                   color='black', lw=0.8)
ax.axhline( 1.96/np.sqrt(M),   color='red',   lw=1, ls='--', label='±1.96/√M')
ax.axhline(-1.96/np.sqrt(M),   color='red',   lw=1, ls='--')
ax.set_xlabel('Lag')
ax.set_ylabel('ACF')
ax.set_title(f'[P3] Autocorrelation – Gibbs\nESS = {ess_val:.0f} / {M}  '
             f'(efficiency = {ess_val/M*100:.1f}%)')
ax.legend(fontsize=9)

# 3b. ACF – MH
ax = fig.add_subplot(gs[2, 1])
ax.bar(LAGS, acf_mh, color=COLORS['mh'], alpha=0.7, width=0.8)
ax.axhline(0,                   color='black', lw=0.8)
ax.axhline( 1.96/np.sqrt(M),   color='blue',  lw=1, ls='--', label='±1.96/√M')
ax.axhline(-1.96/np.sqrt(M),   color='blue',  lw=1, ls='--')
ax.set_xlabel('Lag')
ax.set_ylabel('ACF')
ax.set_title(f'[P3] Autocorrelation – MH (trên k)\nESS = {ess_mh:.0f} / {M}  '
             f'(efficiency = {ess_mh/M*100:.1f}%)')
ax.legend(fontsize=9)

# 3c. ESS bar chart
ax = fig.add_subplot(gs[2, 2])
methods  = ['Gibbs\n(trên x)', 'MH\n(trên k)']
ess_list = [ess_val, ess_mh]
bars = ax.bar(methods, ess_list,
              color=[COLORS['gibbs'], COLORS['mh']],
              alpha=0.8, width=0.5)
ax.axhline(M, color='gray', lw=1, ls='--', label=f'M={M} (lý tưởng i.i.d.)')
for b, v in zip(bars, ess_list):
    ax.text(b.get_x() + b.get_width()/2, v + 20,
            f'{v:.0f}', ha='center', va='bottom', fontweight='bold')
ax.set_ylabel('Effective Sample Size')
ax.set_title('[P3] ESS so sánh\n(khác không gian – xem P4 ghi chú)')
ax.legend(fontsize=9)

# ─── Hàng 4: Phần 4 (So sánh Gibbs vs MH) ───────────────────────────────────

# 4a. Phân phối so sánh
ax = fig.add_subplot(gs[3, 0])
bins = range(min(samples_gibbs.min(), samples_mh.min()),
             max(samples_gibbs.max(), samples_mh.max()) + 2)
ax.hist(samples_gibbs, bins=bins, density=True, alpha=0.55,
        color=COLORS['gibbs'], label='Gibbs (x ∈ {0,1}^N)', edgecolor='white')
ax.hist(samples_mh,    bins=bins, density=True, alpha=0.55,
        color=COLORS['mh'],    label='MH (k ∈ ℤ)',          edgecolor='white')
ax.axvline(E_theory, color=COLORS['theory'], lw=2, ls='--',
           label=f'Lý thuyết = {E_theory:.2f}')
ax.set_xlabel('#số 1')
ax.set_ylabel('Mật độ')
ax.set_title('[P4] Phân phối ước lượng: Gibbs vs. MH')
ax.legend(fontsize=9)

# 4b. Hội tụ running mean cả hai
ax = fig.add_subplot(gs[3, 1])
run_mh_mean = np.cumsum(samples_mh) / np.arange(1, M + 1)
ax.plot(run_mean,    color=COLORS['gibbs'], lw=1.5, label='Gibbs')
ax.plot(run_mh_mean, color=COLORS['mh'],   lw=1.5, alpha=0.8, label='MH')
ax.axhline(E_theory, color=COLORS['theory'], lw=1.5, ls='--', label='Lý thuyết')
ax.set_xlabel('Số mẫu')
ax.set_ylabel('E[#1]')
ax.set_title('[P4] Hội tụ running mean: Gibbs vs. MH')
ax.legend(fontsize=9)

# 4c. Bảng tóm tắt
ax = fig.add_subplot(gs[3, 2])
ax.axis('off')
summary = [
    ['Chỉ số',          'Gibbs',                       'MH'],
    ['Không gian',      'x ∈ {0,1}^N',                'k ∈ ℤ'],
    ['E[#1]',          f'{E_hat:.3f}',                f'{samples_mh.mean():.3f}'],
    ['SE',             f'{SE:.4f}',
                       f'{samples_mh.std()/np.sqrt(M):.4f}'],
    ['ESS',            f'{ess_val:.0f}',              f'{ess_mh:.0f}'],
    ['ESS/M (%)',      f'{ess_val/M*100:.1f}%',       f'{ess_mh/M*100:.1f}%'],
    ['Accept rate',    'N/A (Gibbs)',                 f'{mh_accept*100:.1f}%'],
    ['Lý thuyết',      f'{E_theory:.3f}',             f'{E_theory:.3f}'],
]
tbl = ax.table(cellText=summary[1:], colLabels=summary[0],
               cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.auto_set_column_width([0, 1, 2])
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#DBEAFE')
        cell.set_text_props(fontweight='bold')
ax.set_title('[P4] Bảng so sánh\n(ghi chú: hai không gian khác nhau)', fontsize=10)

# ─── Hàng 5: Phần 5 (Ước lượng Bayes) ───────────────────────────────────────

theta_grid = np.linspace(0.001, 0.999, 500)
prior_pdf  = beta_dist.pdf(theta_grid, alpha0, beta0)
post_pdf   = beta_dist.pdf(theta_grid, alpha_post, beta_post)

# ── [S2] Chuẩn hóa likelihood bằng trapezoid rule ──
like_unnorm = theta_grid**h_obs * (1 - theta_grid)**(n_obs - h_obs)
like_norm   = like_unnorm / trapz(like_unnorm, theta_grid)   # ∫ likelihood dθ = 1

# 5a. Prior / Likelihood / Posterior giải tích
ax = fig.add_subplot(gs[4, 0])
ax.plot(theta_grid, prior_pdf, color=COLORS['prior'],      lw=2,
        label=f'Prior Beta({alpha0},{beta0})')
ax.plot(theta_grid, like_norm, color=COLORS['likelihood'], lw=2, ls='-.',
        label='Likelihood (chuẩn hóa)')
ax.plot(theta_grid, post_pdf,  color=COLORS['posterior'],  lw=2.5,
        label=f'Posterior Beta({alpha_post},{beta_post})')
ax.axvline(theta_mean_exact, color=COLORS['posterior'], lw=1.5, ls='--',
           label=f'E[θ|data] = {theta_mean_exact:.3f}')
ax.axvline(h_obs/n_obs,      color=COLORS['likelihood'], lw=1.5, ls=':',
           label=f'MLE = {h_obs/n_obs:.3f}')
ax.set_xlabel('θ')
ax.set_ylabel('Mật độ xác suất')
ax.set_title(f'[P5] Bayes giải tích\n(n={n_obs}, h={h_obs}; '
             f'likelihood chuẩn hóa bằng trapezoid)')
ax.legend(fontsize=8)

# 5b. MCMC Posterior histogram vs. giải tích
ax = fig.add_subplot(gs[4, 1])
ax.hist(theta_samples, bins=60, density=True, alpha=0.6,
        color=COLORS['posterior'], edgecolor='white', label='Mẫu MCMC')
ax.plot(theta_grid, post_pdf, color='black', lw=2, label='Posterior giải tích')
ax.axvline(theta_mean_mcmc, color='red', lw=1.5, ls='--',
           label=f'E[θ|data] MCMC = {theta_mean_mcmc:.3f}')
ax.axvspan(theta_ci_lo, theta_ci_hi, alpha=0.15, color='red',
           label=f'95% CI [{theta_ci_lo:.3f}, {theta_ci_hi:.3f}]')
ax.set_xlabel('θ')
ax.set_ylabel('Mật độ')
ax.set_title(f'[P5] MCMC vs. Posterior giải tích\n'
             f'(accept rate = {bayes_accept*100:.1f}%; '
             f'sai số = {abs(theta_mean_mcmc - theta_mean_exact):.4f})')
ax.legend(fontsize=8)

# 5c. Trace + running mean của θ
ax  = fig.add_subplot(gs[4, 2])
ax2 = ax.twinx()
ax.plot(theta_samples[:1000], color=COLORS['posterior'],
        lw=0.7, alpha=0.6, label='Trace θ')
run_theta = np.cumsum(theta_samples) / np.arange(1, M_bayes + 1)
ax2.plot(run_theta, color='black', lw=1.5, label='Running mean')
ax2.axhline(theta_mean_exact, color='red', lw=1.5, ls='--',
            label=f'Đúng = {theta_mean_exact:.3f}')
ax.set_xlabel('Bước MCMC')
ax.set_ylabel('θ (trace)', color=COLORS['posterior'])
ax2.set_ylabel('E[θ|data]', color='black')
ax.set_title('[P5] Trace & hội tụ θ\n(Bayes MCMC)')
lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8)

plt.savefig('gibbs_mcmc_full.png', dpi=150, bbox_inches='tight')
print("\nĐã lưu: gibbs_mcmc_full.png")
plt.show()

# ─── Tóm tắt kết quả ─────────────────────────────────────────────────────────
print("\n" + "═" * 60)
print("TÓM TẮT KẾT QUẢ")
print("═" * 60)

print(f"\n[P1] Gibbs Sampler – Chuỗi nhị phân (N={N}, M={M})")
print(f"  E[#1] MC     = {E_hat:.4f}")
print(f"  Lý thuyết    = {E_theory:.4f}  (xấp xỉ (N+1)/3 = {(N+1)/3:.4f})")
print(f"  SE           = {SE:.4f}")
print(f"  95% CI       : [{E_hat - 1.96*SE:.4f}, {E_hat + 1.96*SE:.4f}]")

print(f"\n[P2] Hội tụ đa chain (K={N_CHAINS} chains)")
print(f"  Gelman-Rubin R̂ = {R_hat:.4f}  "
      f"({'✓ hội tụ (< 1.1)' if R_hat < 1.1 else '✗ chưa hội tụ'})")

print(f"\n[P3] Hiệu quả lấy mẫu (ESS theo Geyer 1992)")
print(f"  Gibbs ESS = {ess_val:.1f}  ({ess_val/M*100:.1f}% efficiency)")
print(f"  MH    ESS = {ess_mh:.1f}  ({ess_mh/M*100:.1f}% efficiency)")

print(f"\n[P4] So sánh Gibbs vs. MH")
print(f"  |E_Gibbs - lý thuyết| = {abs(E_hat - E_theory):.4f}")
print(f"  |E_MH    - lý thuyết| = {abs(samples_mh.mean() - E_theory):.4f}")
print(f"  ⚠ Lưu ý: Gibbs lấy mẫu x ∈ {{0,1}}^N; MH lấy mẫu k ∈ ℤ")

print(f"\n[P5] Ước lượng Bayes (n={n_obs} lần tung, h={h_obs} lần ngửa)")
print(f"  Prior: Beta({alpha0},{beta0})")
print(f"  Posterior giải tích: Beta({alpha_post},{beta_post})")
print(f"  E[θ|data] giải tích = {theta_mean_exact:.4f}")
print(f"  E[θ|data] MCMC      = {theta_mean_mcmc:.4f}")
print(f"  Sai số MCMC         = {abs(theta_mean_mcmc - theta_mean_exact):.4f}")
print(f"  95% Credible Interval: [{theta_ci_lo:.4f}, {theta_ci_hi:.4f}]")
print(f"  MH acceptance rate  = {bayes_accept*100:.1f}%")
print("═" * 60)