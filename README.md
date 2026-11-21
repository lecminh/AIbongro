# AI Bóng Rổ FIBA — OBS Live Tracking (YOLOv8 & Gemini AI)

Hệ thống thống kê bóng rổ chuẩn FIBA Box Score trực tiếp từ luồng OBS Studio kết hợp mô hình thị giác máy tính YOLOv8 (hitbox cầu thủ + vệt tracking quả bóng) và Gemini AI (nhận diện số áo & sự kiện trận đấu).

## 🚀 Cách chạy & Tự động cập nhật

### Cách 1: Dùng Launcher tự cập nhật (Không cần Git - Khuyên dùng)
Chỉ cần chạy:
```bash
python run.py
```
File `run.py` sẽ **tự động tải bản code mới nhất** từ GitHub về máy trước khi mở app. Bạn không bao giờ phải tải hay ghi đè file thủ công.

### Cách 2: Dùng Git (Nếu đã clone repo)
Click đúp chuột vào file `chay_app.bat` hoặc gõ:
```bash
git pull origin main
python AIbongro.py
```

---

## 📦 Cài đặt thư viện (Chỉ làm lần đầu)

```bash
pip install -r requirements.txt
```

---

## ⚙️ Hướng dẫn kết nối với OBS Studio

1. Mở **OBS Studio**, thêm nguồn video / camera trận đấu vào Scene.
2. Bấm nút **Start Virtual Camera** ở góc phải dưới của OBS.
3. Chạy `python run.py`:
   - Bước 1: Nhập số camera ảo OBS (thường là `0` hoặc `1` hoặc `2`).
   - Bước 2: Nhập 2 Gemini API Key (Key chính và Key dự bị).
   - Bước 3: Nhập thông tin 2 đội (Tên đội, mã viết tắt, màu áo).
   - Bước 4: Nhấn Enter để AI quét số áo và nhập tên cầu thủ.
4. Trình duyệt sẽ tự động mở tại `http://127.0.0.1:5000` hiển thị luồng video có hitbox + bảng FIBA Box Score tự đồng bộ mỗi 2 giây.
