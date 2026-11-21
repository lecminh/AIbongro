"""
Launcher tự động: Luôn kiểm tra và tải bản AIbongro.py mới nhất từ GitHub trước khi chạy.
Sử dụng: python run.py
"""
import os
import sys
import subprocess
import urllib.request

RAW_URL = "https://raw.githubusercontent.com/lecminh/AIbongro/main/AIbongro.py"
LOCAL_FILE = "AIbongro.py"

print("=" * 64)
print("  AI BÓNG RỔ FIBA - OBS LIVE TRACKING (AUTO UPDATER)")
print("=" * 64)
print("[Auto-Update] Đang kiểm tra và tải bản code mới nhất từ GitHub...")

try:
    urllib.request.urlretrieve(RAW_URL, LOCAL_FILE)
    print("[Auto-Update] Đã tải bản mới nhất thành công!\n")
except Exception as e:
    print(f"[Cảnh báo] Không thể kết nối GitHub ({e}). Sẽ chạy bản hiện có trên máy.\n")

if not os.path.exists(LOCAL_FILE):
    print(f"[Lỗi] Không tìm thấy file {LOCAL_FILE}!")
    sys.exit(1)

subprocess.run([sys.executable, LOCAL_FILE])
