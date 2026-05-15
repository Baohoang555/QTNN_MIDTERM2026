# CÁC THƯ VIỆN CẦN THIẾT
import random
import math
import string
from collections import defaultdict 
# ====================================================================================
# KHỞI TẠO CÁC BIẾN VÀ HÀM
# ====================================================================================
# BIẾN DỮ LIỆU MÃ HÓA CẦN ĐƯỢC GIẢI MÃ
DU_LIEU_MA_HOA = """
gsadgaredgen ad wedeoil ioe woeit ntubcladw clsgyn ad the kif sm thit
glinn sm thadyeon khs hive ceed erugiter ts ydsk dsthadw sm the thesof sm posci-calataen thit thesof ts khagh the bsnt wlsoasun sczegtn sm hubid oeneiogh ioe
adrecter mso the bsnt wlsoasun sm alluntoitasdn - erwio illed pse the buoreon ad the oue bsowue
"""
# CHUYỂN TẤT CẢ CÁC CHỮ TRONG DỮ LIỆU MÃ HÓA THÀNH CHỮ THƯỜNG
DU_LIEU_MA_HOA = DU_LIEU_MA_HOA.lower()
# ====================================================================================
# BIẾN DỮ LIỆU MẪU DỰA VÀO ĐỂ GIẢI MÃ BIẾN DỮ LIỆU MÃ HÓA
DU_LIEU_MAU = """
it is a truth universally acknowledged that a single man in possession of a good fortune
must be in want of a wife however little known the feelings or views of such a man may
be on his first entering a neighbourhood this truth is so well fixed in the minds of the
surrounding families that he is considered the rightful property of some one or other of
their daughters

the quick brown fox jumps over the lazy dog
the theory of probabilities is the most important subject in mathematics and statistics
the most glorious of illustrations are often found in the study of random processes
coincidences in general are great stumbling blocks in the way of that class of thinkers
who have been educated to know nothing of the theory of probabilities

beyond the hills and valleys where silent rivers wandered through ancient forests there lived
many curious minds who sought wisdom in books and conversation every evening the old
scholars gathered beneath the fading light of lanterns to discuss philosophy science and
the mysteries of human nature some believed that destiny governed every action while
others argued that chance alone shaped the fortunes of mankind the children listened with
great attention learning that knowledge was not merely the collection of facts but the art
of understanding the world with patience imagination and reason and through these shared
stories their small village slowly became known as a place of remarkable thought and
unexpected discovery
"""
# CHUYỂN TẤT CẢ CÁC CHỮ TRONG DỮ LIỆU MẪU HÓA THÀNH CHỮ THƯỜNG
DU_LIEU_MAU = DU_LIEU_MAU.lower()
# ====================================================================================
# BIẾN BẢNG CHỮ CÁI ĐỂ LIỆT KÊ 26 CHỮ CÁI VÀ " ", TỔNG 27 KÝ TỰ
BANG_CHU_CAI = list(string.ascii_lowercase + " ")
# ====================================================================================
# HÀM LÀM SẠCH DỮ LIỆU, LOẠI BỎ CÁC KÝ TỰ KHÔNG CÓ TRONG BIẾN BANG_CHU_CAI, THAY THẾ CÁC KÝ TỰ XUỐNG DÒNG "\N" THÀNH KÝ TỰ " "
def HAM_LAM_SACH(DU_LIEU):
    TAP_DU_LIEU = []
    for ch in DU_LIEU:
        if ch in BANG_CHU_CAI:
            TAP_DU_LIEU.append(ch) #LỆNH .append DÙNG ĐỂ ĐIỀN KÝ TỰ VÀO TẬP
        elif ch == "\n":
            TAP_DU_LIEU.append(" ")
    return "".join(TAP_DU_LIEU)
# LÀM SẠCH CÁC DỮ LIỆU ĐẦU VÀO
DU_LIEU_MA_HOA = HAM_LAM_SACH(DU_LIEU_MA_HOA)
DU_LIEU_MAU = HAM_LAM_SACH(DU_LIEU_MAU)
# ====================================================================================
# HÀM TÍNH ĐIỂM DỮ LIỆU ĐƯỢC GIẢI MÃ THEO ĐIỂM MẪU DỮ LIỆU. ĐIỂM NÀY TÍNH TRÊN HÀM LOG
def DIEM_DU_LIEU(DU_LIEU, KHOA, DIEM_MAU):
    DIEM_DU_LIEU = 0.0
    for i in range(len(DU_LIEU) - 1):
        a = KHOA.get(DU_LIEU[i], DU_LIEU[i])
        b = KHOA.get(DU_LIEU[i + 1], DU_LIEU[i + 1])
        DIEM_DU_LIEU += math.log(1 + DIEM_MAU[(a, b)]) # 1 + DIEM_MAU[(a, b)] để tránh trường hợp log(0)
    return DIEM_DU_LIEU
# ====================================================================================
# SỐ BƯỚC CHẠY CHƯƠNG TRÌNH
SO_BUOC = 100000
# ====================================================================================
# KHOẢNG CÁCH MỖI LẦN IN ĐỂ KIỂM TRA KẾT QUẢ
KHOANG_CACH_IN = 5000
# ====================================================================================
# ĐIỂM CỦA DỮ LIỆU MẪU BAN ĐẦU
DIEM_MAU = defaultdict(int)
for i in range(len(DU_LIEU_MAU) - 1):
    a = DU_LIEU_MAU[i]
    b = DU_LIEU_MAU[i + 1]
    if a in BANG_CHU_CAI and b in BANG_CHU_CAI:
        DIEM_MAU[(a, b)] += 1
# ====================================================================================        
KHOA_HIEN_TAI = {ch: ch for ch in BANG_CHU_CAI}
# ====================================================================================
# ĐIỂM CỦA DỮ LIỆU MÃ HÓA HIỆN TẠI
DIEM_HIEN_TAI = DIEM_DU_LIEU(
    DU_LIEU_MA_HOA,
    KHOA_HIEN_TAI,
    DIEM_MAU
)
# ====================================================================================
KHOA_TOT_NHAT = KHOA_HIEN_TAI.copy()
# ====================================================================================
DIEM_TOT_NHAT = DIEM_HIEN_TAI
# ====================================================================================
# VÒNG LẶP MCMC
for BUOC in range(1, SO_BUOC + 1):
    # BƯỚC 1: TẠO PROPOSAL KEY
    KHOA_DE_XUAT = KHOA_HIEN_TAI.copy()
    a, b = random.sample(BANG_CHU_CAI, 2)
    KHOA_DE_XUAT[a], KHOA_DE_XUAT[b] = KHOA_DE_XUAT[b], KHOA_DE_XUAT[a]
    # BƯỚC 2: TÍNH SCORE MỚI
    DIEM_DE_XUAT = DIEM_DU_LIEU(
        DU_LIEU_MA_HOA,
        KHOA_DE_XUAT,
        DIEM_MAU
    )
    # BƯỚC 3: XÁC XUẤT CHẤP NHẬN
    log_alpha = DIEM_DE_XUAT - DIEM_HIEN_TAI
    if log_alpha >= 0:
        alpha = 1.0
    else:
        alpha = math.exp(log_alpha)
    # BƯỚC 4: ACCEPT / REJECT
    u = random.random()
    if u <= alpha:
        KHOA_HIEN_TAI = KHOA_DE_XUAT.copy()
        DIEM_HIEN_TAI = DIEM_DE_XUAT
    # BƯỚC 6: LƯU BEST
    if DIEM_HIEN_TAI > DIEM_TOT_NHAT:
        KHOA_TOT_NHAT = KHOA_HIEN_TAI.copy()
        DIEM_TOT_NHAT = DIEM_HIEN_TAI
    # BƯỚC 7: IN KẾT QUẢ SAU SỐ BƯỚC NHẤT ĐỊNH
    if BUOC % KHOANG_CACH_IN == 0:
        print("\n" + "=" * 70)
        print("Bước thứ:", BUOC)
        print(
            "Điểm hiện tại:",
            round(DIEM_HIEN_TAI, 2)
        )
        print(
            "Điểm tốt nhất:",
            round(DIEM_TOT_NHAT, 2)
        )
        print("-" * 70)
        # GIẢI MÃ TẠM THỜI
        DU_LIEU_TAM_THOI = []
        for ch in DU_LIEU_MA_HOA:
            DU_LIEU_TAM_THOI.append(
                KHOA_TOT_NHAT.get(ch, ch)
            )
        DU_LIEU_TAM_THOI = "".join(DU_LIEU_TAM_THOI)
        print(DU_LIEU_TAM_THOI[:800])
# ====================================================================================
# KẾT QUẢ CUỐI
print("\n" + "=" * 70)
print("DỮ LIỆU ĐƯỢC GIẢI MÃ")
print("=" * 70)
DU_LIEU_DUOC_GIAI_MA = []
for ch in DU_LIEU_MA_HOA:
    DU_LIEU_DUOC_GIAI_MA.append(
        KHOA_TOT_NHAT.get(ch, ch)
    )
DU_LIEU_DUOC_GIAI_MA = "".join(DU_LIEU_DUOC_GIAI_MA)
print(DU_LIEU_DUOC_GIAI_MA)
