import os
import glob
import time
import queue
import threading
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import cv2
import numpy as np
from typing import List, Optional, Literal
from dataclasses import dataclass
from flask import Flask, Response
from ultralytics import YOLO
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# =====================================================================
# PHẦN 1: SCHEMA GEMINI API & BỘ QUẢN LÝ CHUYỂN ĐỔI API 1 -> API 2
# =====================================================================

class RosterScanResult(BaseModel):
    team_a_numbers: List[int] = Field(description="Danh sách các số áo cầu thủ Đội A nhìn thấy được trên sân (không trùng lặp)")
    team_b_numbers: List[int] = Field(description="Danh sách các số áo cầu thủ Đội B nhìn thấy được trên sân (không trùng lặp)")

EventType = Literal[
    "2PT_MADE", "2PT_MISSED",
    "3PT_MADE", "3PT_MISSED",
    "FT_MADE", "FT_MISSED",
    "OREB", "DREB",
    "ASSIST", "TURNOVER", "STEAL", "BLOCK",
    "FOUL_PERSONAL", "FOUL_DRAWN",
    "SUBSTITUTION_IN", "SUBSTITUTION_OUT"
]

class BasketballPlayEvent(BaseModel):
    team_code: str = Field(description="Mã đội bóng dựa theo màu áo")
    jersey_number: int = Field(description="Số áo cầu thủ thực hiện hành động")
    event_type: EventType = Field(description="Loại sự kiện thống kê chuẩn FIBA")
    is_in_paint: bool = Field(default=False, description="Điểm ghi trong khu vực hình thang (Points in the Paint)")
    is_fast_break: bool = Field(default=False, description="Điểm phản công nhanh (Fast Break Points)")
    is_second_chance: bool = Field(default=False, description="Điểm cơ hội hai sau OREB (Second Chance Points)")
    is_off_turnover: bool = Field(default=False, description="Điểm ghi ngay sau khi đối phương mất bóng (Points from Turnovers)")

class ClipAnalysisResult(BaseModel):
    active_players_team_a: List[int] = Field(description="Danh sách số áo cầu thủ Đội A đang có mặt trên sân")
    active_players_team_b: List[int] = Field(description="Danh sách số áo cầu thủ Đội B đang có mặt trên sân")
    events: List[BasketballPlayEvent] = Field(description="Danh sách các sự kiện bóng rổ diễn ra trong đoạn video")


class GeminiFailoverManager:
    """
    Quản lý 2 Gemini API Key. Khi key đang dùng báo hết hạn mức (429 / RESOURCE_EXHAUSTED / Quota),
    tự chuyển sang key còn lại và gửi lại request. Chuyển 2 chiều: key 1 hồi quota sẽ được dùng lại khi key 2 hết.
    """
    MODEL = "gemini-3.1-flash-lite"

    def __init__(self, api_key_1: str, api_key_2: str):
        opts = types.HttpOptions(timeout=30000)
        self.clients = [genai.Client(api_key=api_key_1, http_options=opts), genai.Client(api_key=api_key_2, http_options=opts)]
        self.active_idx = 0
        self.lock = threading.Lock()

    @property
    def active_label(self) -> str:
        return f"API #{self.active_idx + 1} ({'CHINH' if self.active_idx == 0 else 'DU BI'})"

    def _is_quota_or_key_error(self, err: Exception) -> bool:
        msg = str(err).lower()
        triggers = [
            "429", "resource_exhausted", "quota", "rate limit",
            "exhausted", "exceeded", "api_key_invalid", "403"
        ]
        return any(t in msg for t in triggers)

    def _switch_from(self, idx: int, reason: str):
        with self.lock:
            if self.active_idx == idx:
                self.active_idx = 1 - idx
                print(f"\n[FAILOVER] API {idx + 1} gặp giới hạn ({reason[:60]}). Chuyển sang API {self.active_idx + 1}!")

    def _generate(self, contents: list, schema):
        for attempt in range(2):
            idx = self.active_idx
            try:
                response = self.clients[idx].models.generate_content(
                    model=self.MODEL,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=0.1,
                    ),
                )
                return response.parsed
            except Exception as e:
                if attempt == 0 and self._is_quota_or_key_error(e):
                    self._switch_from(idx, str(e))
                    continue
                raise

    def analyze_images_roster(self, jpeg_bytes_list: List[bytes], prompt: str) -> Optional[RosterScanResult]:
        parts = [types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in jpeg_bytes_list]
        return self._generate(parts + [prompt], RosterScanResult)

    def analyze_video_clip(self, video_input, prompt: str) -> Optional[ClipAnalysisResult]:
        if isinstance(video_input, str):
            with open(video_input, "rb") as f:
                video_bytes = f.read()
        else:
            video_bytes = video_input
        part = types.Part.from_bytes(data=video_bytes, mime_type="video/mp4")
        return self._generate([part, prompt], ClipAnalysisResult)


# =====================================================================
# PHẦN 2: BỘ TÍNH TOÁN SỐ LIỆU CHUẨN FIBA BOX SCORE
# =====================================================================

@dataclass
class PlayerStats:
    no: str
    name: str
    is_starter: bool = True
    is_captain: bool = False
    on_court: bool = True
    seconds_played: float = 0.0
    fg2_m: int = 0; fg2_a: int = 0
    fg3_m: int = 0; fg3_a: int = 0
    ft_m: int = 0;  ft_a: int = 0
    or_reb: int = 0; dr_reb: int = 0
    assists: int = 0; turnovers: int = 0; steals: int = 0; blocks: int = 0
    pf: int = 0; fd: int = 0; plus_minus: int = 0

    @property
    def fg_m(self) -> int: return self.fg2_m + self.fg3_m
    @property
    def fg_a(self) -> int: return self.fg2_a + self.fg3_a
    @property
    def tot_reb(self) -> int: return self.or_reb + self.dr_reb
    @property
    def pts(self) -> int: return (self.fg2_m * 2) + (self.fg3_m * 3) + self.ft_m

    @property
    def min_formatted(self) -> str:
        if self.seconds_played == 0 and not self.is_starter:
            return "DNP"
        mins = int(self.seconds_played // 60)
        secs = int(self.seconds_played % 60)
        return f"{mins:02d}:{secs:02d}"

    @staticmethod
    def calc_pct(made: int, attempted: int) -> str:
        return "0.0" if attempted == 0 else f"{(made / attempted) * 100:.1f}"


class FIBABoxScoreEngine:
    def __init__(self, team_a_code: str, team_a_name: str, team_a_color: str,
                 team_b_code: str, team_b_name: str, team_b_color: str,
                 match_format: str = "5x5"):
        self.match_format = match_format
        self.max_on_court = 3 if "3" in match_format else 5
        self.lock = threading.Lock()
        self.teams = {
            team_a_code: {"name": team_a_name, "color": team_a_color, "players": {}, "score": 0},
            team_b_code: {"name": team_b_name, "color": team_b_color, "players": {}, "score": 0}
        }
        self.team_codes = [team_a_code, team_b_code]
        self.adv_stats = {
            code: {
                "pts_from_to": 0, "pts_in_paint_m": 0, "pts_in_paint_a": 0, "pts_in_paint_pts": 0,
                "second_chance_pts": 0, "fast_break_pts": 0, "bench_pts": 0,
                "biggest_lead": 0, "biggest_lead_score": "0-0",
                "biggest_run": 0, "biggest_run_score": "0-0",
                "time_with_lead_sec": 0.0
            } for code in self.team_codes
        }
        self.lead_changes = 0
        self.times_tied = 0
        self.current_leader: Optional[str] = None
        self.current_run_team: Optional[str] = None
        self.current_run_pts: int = 0

    def register_player(self, team_code: str, no: int, name: str, is_starter: bool = True, is_captain: bool = False):
        with self.lock:
            self.teams[team_code]["players"][no] = PlayerStats(
                no=str(no), name=name.upper(), is_starter=is_starter, is_captain=is_captain, on_court=is_starter
            )

    def get_roster_summary_for_prompt(self) -> str:
        lines = []
        for code in self.team_codes:
            t = self.teams[code]
            if t["players"]:
                p_list = ", ".join([f"#{no} ({p.name})" for no, p in sorted(t["players"].items())])
            else:
                p_list = "Chưa có danh sách đăng ký cụ thể"
            lines.append(f"- Đội {code} ({t['name']} - Màu áo: {t['color']}): {p_list}")
        return "\n".join(lines)

    def _get_player(self, team_code: str, no: int) -> PlayerStats:
        players = self.teams[team_code]["players"]
        if no not in players:
            players[no] = PlayerStats(no=str(no), name=f"CẦU THỦ #{no}", is_starter=False, on_court=True)
        return players[no]

    def update_time_played(self, elapsed_seconds: float, active_a: List[int], active_b: List[int]):
        with self.lock:
            for code, active in ((self.team_codes[0], active_a), (self.team_codes[1], active_b)):
                if active:  # Gemini không thấy ai thì giữ đội hình cũ
                    team_players = self.teams[code]["players"]
                    has_registered_roster = len(team_players) >= self.max_on_court

                    valid_active = []
                    for no in active:
                        if no in team_players:
                            valid_active.append(no)
                        elif not has_registered_roster and len(valid_active) < self.max_on_court:
                            # Chỉ tự động thêm nếu đội hình chưa đủ số lượng trên sân
                            self._get_player(code, no)
                            valid_active.append(no)

                    # Giới hạn số cầu thủ trên sân không vượt quá thể thức (3 cho 3x3, 5 cho 5x5)
                    valid_active = valid_active[:self.max_on_court]

                    if valid_active:
                        for no, p in team_players.items():
                            p.on_court = (no in valid_active)

                for p in self.teams[code]["players"].values():
                    if p.on_court:
                        p.seconds_played += elapsed_seconds

            s_a = self.teams[self.team_codes[0]]["score"]
            s_b = self.teams[self.team_codes[1]]["score"]
            if s_a > s_b:
                self.adv_stats[self.team_codes[0]]["time_with_lead_sec"] += elapsed_seconds
            elif s_b > s_a:
                self.adv_stats[self.team_codes[1]]["time_with_lead_sec"] += elapsed_seconds

    def record_event(self, ev: BasketballPlayEvent):
        with self.lock:
            if ev.team_code not in self.teams:
                return
            team = self.teams[ev.team_code]
            opp_code = self.team_codes[1] if ev.team_code == self.team_codes[0] else self.team_codes[0]

            # Kiểm tra whitelist: nếu đội đã có danh sách đăng ký và số áo lạ không thuộc danh sách -> bỏ qua
            has_registered_roster = len(team["players"]) >= self.max_on_court
            if has_registered_roster and ev.jersey_number not in team["players"]:
                return

            p = self._get_player(ev.team_code, ev.jersey_number)
            pts = 0

            if ev.event_type == "2PT_MADE":
                p.fg2_m += 1; p.fg2_a += 1; pts = 2
                if ev.is_in_paint:
                    self.adv_stats[ev.team_code]["pts_in_paint_m"] += 1
                    self.adv_stats[ev.team_code]["pts_in_paint_a"] += 1
                    self.adv_stats[ev.team_code]["pts_in_paint_pts"] += 2
            elif ev.event_type == "2PT_MISSED":
                p.fg2_a += 1
                if ev.is_in_paint:
                    self.adv_stats[ev.team_code]["pts_in_paint_a"] += 1
            elif ev.event_type == "3PT_MADE":
                p.fg3_m += 1; p.fg3_a += 1; pts = 3
            elif ev.event_type == "3PT_MISSED":
                p.fg3_a += 1
            elif ev.event_type == "FT_MADE":
                p.ft_m += 1; p.ft_a += 1; pts = 1
            elif ev.event_type == "FT_MISSED":
                p.ft_a += 1
            elif ev.event_type == "OREB": p.or_reb += 1
            elif ev.event_type == "DREB": p.dr_reb += 1
            elif ev.event_type == "ASSIST": p.assists += 1
            elif ev.event_type == "TURNOVER": p.turnovers += 1
            elif ev.event_type == "STEAL": p.steals += 1
            elif ev.event_type == "BLOCK": p.blocks += 1
            elif ev.event_type == "FOUL_PERSONAL": p.pf += 1
            elif ev.event_type == "FOUL_DRAWN": p.fd += 1
            elif ev.event_type == "SUBSTITUTION_IN": p.on_court = True
            elif ev.event_type == "SUBSTITUTION_OUT": p.on_court = False

            if pts > 0:
                team["score"] += pts
                for pl in team["players"].values():
                    if pl.on_court: pl.plus_minus += pts
                for pl in self.teams[opp_code]["players"].values():
                    if pl.on_court: pl.plus_minus -= pts

                if not p.is_starter: self.adv_stats[ev.team_code]["bench_pts"] += pts
                if ev.is_off_turnover: self.adv_stats[ev.team_code]["pts_from_to"] += pts
                if ev.is_second_chance: self.adv_stats[ev.team_code]["second_chance_pts"] += pts
                if ev.is_fast_break: self.adv_stats[ev.team_code]["fast_break_pts"] += pts

                score_str = f"{self.teams[self.team_codes[0]]['score']}-{self.teams[self.team_codes[1]]['score']}"
                if self.current_run_team == ev.team_code:
                    self.current_run_pts += pts
                else:
                    self.current_run_team = ev.team_code
                    self.current_run_pts = pts

                if self.current_run_pts > self.adv_stats[ev.team_code]["biggest_run"]:
                    self.adv_stats[ev.team_code]["biggest_run"] = self.current_run_pts
                    self.adv_stats[ev.team_code]["biggest_run_score"] = score_str

                diff = team["score"] - self.teams[opp_code]["score"]
                if diff > self.adv_stats[ev.team_code]["biggest_lead"]:
                    self.adv_stats[ev.team_code]["biggest_lead"] = diff
                    self.adv_stats[ev.team_code]["biggest_lead_score"] = score_str

                if team["score"] == self.teams[opp_code]["score"]:
                    self.times_tied += 1
                elif diff > 0 and self.current_leader != ev.team_code:
                    if self.current_leader is not None:
                        self.lead_changes += 1
                    self.current_leader = ev.team_code


# =====================================================================
# PHẦN 3: BỘ ĐỌC LUỒNG OBS & XỬ LÝ HITBOX + GEMINI FAILOVER
# =====================================================================

def open_obs_capture(source) -> cv2.VideoCapture:
    """Mở luồng OBS: hỗ trợ cả OBS Virtual Camera (chỉ số int) và RTSP/UDP/HTTP URL"""
    if isinstance(source, int) and os.name == "nt":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
    return cv2.VideoCapture(source)


class AsyncOBSReader:
    def __init__(self, source):
        self.source = source
        self.cap = open_obs_capture(source)
        self.latest_frame: Optional[np.ndarray] = None
        self.running = True
        self.connected = self.cap.isOpened()
        self.lock = threading.Lock()

        if self.connected:
            threading.Thread(target=self._read_loop, daemon=True).start()

    def _read_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame
            else:
                time.sleep(0.02)

    def get_frame(self, target_size=(960, 540)) -> np.ndarray:
        with self.lock:
            if self.latest_frame is not None:
                return cv2.resize(self.latest_frame, target_size)
        blank = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        cv2.putText(blank, "DANG CHO TIN HIEU TU OBS STUDIO...", (180, target_size[1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return blank

    def stop(self):
        self.running = False
        self.cap.release()


class OBSVisionSystem:
    def __init__(self, obs_reader: AsyncOBSReader, engine: FIBABoxScoreEngine, gemini_mgr: GeminiFailoverManager):
        self.obs = obs_reader
        self.engine = engine
        self.gemini = gemini_mgr
        self.running = True

        self.latest_jpeg: Optional[bytes] = None
        for leftover in glob.glob("temp_obs_clip_*.mp4"):
            try:
                os.remove(leftover)
            except Exception:
                pass
        self.pool = ThreadPoolExecutor(max_workers=3)
        self.clip_queue = queue.Queue(maxsize=6)
        self.gemini_status = "Đang theo dõi luồng OBS..."
        self.ball_trail = deque(maxlen=12)

        # Đa giác ranh giới sân đấu (Court ROI) theo tọa độ tỷ lệ (x_ratio, y_ratio: 0.0 -> 1.0)
        # Giúp tự động loại trừ quay phim ngồi bệt ở góc, khán giả, HLV và VĐV dự bị đứng ngoài biên
        self.court_roi_polygon = [
            (0.04, 0.20),  # Top-left (góc trên bên trái đường biên)
            (0.96, 0.20),  # Top-right (góc trên bên phải đường biên)
            (0.98, 0.93),  # Bottom-right (góc dưới bên phải)
            (0.02, 0.93),  # Bottom-left (góc dưới bên trái)
        ]
        self.use_court_roi = True

        print("[YOLO] Đang khởi tạo mô hình nhận diện Hitbox cầu thủ (yolov8n.pt)...")
        self.yolo_model = YOLO("yolov8n.pt")

    def scan_jerseys_pre_match(self) -> tuple[List[int], List[int]]:
        """Chụp 3 khung hình HD từ luồng OBS gửi lên Gemini để đọc số áo trước trận"""
        print("\n[AI SCAN] Đang chụp 3 khung hình HD từ OBS (trong 1.5 giây) để đọc số áo...")
        jpeg_list = []
        for _ in range(3):
            frame = self.obs.get_frame((1280, 720))
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if ok:
                jpeg_list.append(buf.tobytes())
            time.sleep(0.5)

        c1, c2 = self.engine.team_codes[0], self.engine.team_codes[1]
        t1, t2 = self.engine.teams[c1], self.engine.teams[c2]
        fmt = self.engine.match_format
        max_p = self.engine.max_on_court

        prompt = f"""
        Đây là các khung hình chụp trực tiếp từ luồng OBS ghi hình trận bóng rổ thể thức {fmt} (Bóng rổ {fmt}).
        Hãy quan sát kỹ áo thi đấu của các cầu thủ trên sân để đọc SỐ ÁO của từng đội:
        - Đội A ({t1['name']} - {c1}): mặc áo màu {t1['color']}.
        - Đội B ({t2['name']} - {c2}): mặc áo màu {t2['color']}.

        QUY TẮC PHÂN LOẠI & LOẠI TRỪ BẮT BUỘC:
        1. BỎ QUA TRỌNG TÀI: Người mặc áo xám FIBA hoặc sọc đen-trắng, quần dài thể thao màu đen, đeo còi trước ngực. TUYỆT ĐỐI KHÔNG coi trọng tài là cầu thủ.
        2. BỎ QUA NGƯỜI NGOÀI SÂN: Huấn luyện viên, cầu thủ dự bị ngồi ngoài biên, nhân viên quay phim, ban tổ chức và khán giả.
        3. CHỈ đọc số áo của các vận động viên đang đứng/khởi động/thi đấu trên sân thi đấu mang đúng màu áo của 2 đội trên.
        Mỗi đội tối đa {max_p + 1} số áo (cho 3x3) hoặc 12 số áo (cho 5x5). Trả về danh sách số nguyên không trùng lặp.
        """
        try:
            print(f"[AI SCAN] Đang phân tích số áo bằng {self.gemini.active_label}...")
            res = self.gemini.analyze_images_roster(jpeg_list, prompt)
            if res:
                return sorted(list(set(res.team_a_numbers))), sorted(list(set(res.team_b_numbers)))
        except Exception as e:
            print(f"[Lỗi AI Scan] Không thể quét số áo (cả 2 API đều lỗi): {e}")

        return [], []

    def _draw_hitboxes(self, frame: np.ndarray) -> tuple[np.ndarray, int]:
        # Track cả người (cls 0) và quả bóng rổ (cls 32 - sports ball).
        # Conf 0.18 cho phép bắt bóng nhỏ đang chuyển động nhanh mà không bị bỏ sót.
        results = self.yolo_model.track(frame, persist=True, classes=[0, 32], conf=0.18, verbose=False)
        player_count = 0
        ball_detected = False
        ball_data = None

        h, w = frame.shape[:2]
        court_poly_pts = np.array([
            [int(px * w), int(py * h)] for px, py in self.court_roi_polygon
        ], dtype=np.int32)

        # Vẽ viền ranh giới sân tinh tế (mờ nhẹ để người xem nhận biết vùng thi đấu hợp lệ)
        if self.use_court_roi:
            cv2.polylines(frame, [court_poly_pts], isClosed=True, color=(60, 140, 90), thickness=1, lineType=cv2.LINE_AA)

        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else 0

                if cls_id == 0 and conf >= 0.35:
                    # Kiểm tra vị trí bàn chân ((x1+x2)//2, y2) có nằm trong ranh giới sân không
                    foot_pt = (float((x1 + x2) // 2), float(y2))
                    is_in_court = True
                    if self.use_court_roi:
                        is_in_court = cv2.pointPolygonTest(court_poly_pts, foot_pt, False) >= 0

                    if is_in_court:
                        player_count += 1
                        color = (0, 255, 128)
                        label = f"Player #{track_id} ({conf:.2f})"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
                        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                        cv2.putText(frame, label, (x1 + 2, y1 - 3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1)
                    else:
                        # Người ngoài sân (cameraman ngồi bệt, VĐV dự bị, HLV, khán giả)
                        dim_color = (80, 85, 95)
                        label = f"Outside #{track_id}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), dim_color, 1)
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 4, y1), dim_color, -1)
                        cv2.putText(frame, label, (x1 + 2, y1 - 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 205, 215), 1)

                elif cls_id == 32 and conf >= 0.18:
                    ball_detected = True
                    ball_data = (x1, y1, x2, y2, conf)

        # Xử lý tracking & vẽ vệt di chuyển quả bóng (Ball Trail Tracking)
        if ball_detected and ball_data is not None:
            bx1, by1, bx2, by2, bconf = ball_data
            bcx, bcy = (bx1 + bx2) // 2, (by1 + by2) // 2
            br = max(8, (bx2 - bx1) // 2)
            self.ball_trail.append((bcx, bcy))

            # Vẽ vệt đuôi di chuyển của bóng (Motion Trail)
            trail_len = len(self.ball_trail)
            for i in range(1, trail_len):
                thick = max(1, int((i / trail_len) * 4))
                cv2.line(frame, self.ball_trail[i - 1], self.ball_trail[i], (0, 140, 255), thick)

            # Vẽ tâm ngắm vòng tròn quanh bóng
            cv2.circle(frame, (bcx, bcy), br + 4, (0, 140, 255), 2)
            cv2.circle(frame, (bcx, bcy), 3, (0, 255, 255), -1)
            cv2.putText(frame, f"BALL {bconf:.2f}", (bcx - 24, by1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1)
        elif len(self.ball_trail) > 0:
            # Khi bóng khuất tạm thời (motion blur), làm mờ dần vệt cũ
            self.ball_trail.popleft()
            trail_len = len(self.ball_trail)
            for i in range(1, trail_len):
                thick = max(1, int((i / trail_len) * 3))
                cv2.line(frame, self.ball_trail[i - 1], self.ball_trail[i], (0, 100, 220), thick)

        # Thanh trạng thái HUD chuẩn không bị lỗi dấu ? trong OpenCV
        import unicodedata
        clean_status = "".join(c for c in unicodedata.normalize('NFKD', self.gemini_status) if not unicodedata.combining(c)).replace('đ', 'd').replace('Đ', 'D')
        ball_label = "LOCKED" if ball_detected else "SEARCHING"

        cv2.rectangle(frame, (10, 10), (660, 82), (15, 17, 23), -1)
        cv2.putText(frame, f"OBS LIVE | On-Court: {player_count} ({self.engine.match_format}) | Ball: {ball_label} | {self.gemini.active_label}", (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 255, 200), 2)
        cv2.putText(frame, f"AI: {clean_status[:60]}", (20, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 215, 225), 1)
        return frame, player_count

    def _analyze_clip(self, clip_path: str) -> Optional[ClipAnalysisResult]:
        roster_text = self.engine.get_roster_summary_for_prompt()
        fmt = self.engine.match_format
        max_p = self.engine.max_on_court
        prompt = f"""
        Bạn là trọng tài bàn thống kê FIBA chuyên nghiệp cho trận đấu bóng rổ thể thức {fmt} (Bóng rổ {fmt}).
        Đoạn video này được ghi trực tiếp từ luồng OBS Studio (nếu OBS ghép song song 2 góc máy cùng lúc, CHỈ GHI NHẬN 1 LẦN DUY NHẤT cho mỗi pha bóng thực tế).

        Danh sách đội hình chính thức đã đăng ký:
        {roster_text}

        QUY TẮC LOẠI TRỪ & PHÂN BIỆT ĐỐI TƯỢNG (RẤT QUAN TRỌNG):
        1. TRỌNG TÀI (Referees): Thường mặc áo xám FIBA hoặc áo sọc đen-trắng, quần dài thể thao màu đen, đeo còi trước ngực.
           -> TUYỆT ĐỐI KHÔNG COI TRỌNG TÀI LÀ CẦU THỦ.
           -> KHÔNG đưa trọng tài vào active_players và KHÔNG ghi nhận bất kỳ sự kiện nào cho trọng tài.
        2. VẬN ĐỘNG VIÊN DỰ BỊ, BAN HUẤN LUYỆN, QUAY PHIM & KHÁN GIẢ:
           - Những người đứng/ngồi bên ngoài đường biên (sideline/baseline), ngoài rào chắn, hoặc sau bảng rổ.
           -> TUYỆT ĐỐI KHÔNG đưa vào danh sách active_players và KHÔNG ghi nhận sự kiện thi đấu.
        3. CẦU THỦ ĐANG THI ĐẤU TRÊN SÂN (active_players_team_a, active_players_team_b):
           - CHỈ bao gồm các cầu thủ đang thực sự thi đấu trong ranh giới sân bóng.
           - Giới hạn: Mỗi đội TỐI ĐA {max_p} CẦU THỦ trên sân cùng lúc (Thể thức {fmt}: tổng cộng tối đa {max_p * 2} cầu thủ trên sân).
           - Đối chiếu chính xác số áo theo danh sách đăng ký ở trên.
        4. SỰ KIỆN THI ĐẤU:
           - Ghi nhận: 2PT_MADE/MISSED, 3PT_MADE/MISSED, FT_MADE/MISSED, OREB, DREB, ASSIST, TURNOVER, STEAL, BLOCK, FOUL_PERSONAL, FOUL_DRAWN, SUBSTITUTION_IN/OUT.
           - 1.0 giây đầu video là phần lặp lại cuối clip trước (chỉ để lấy ngữ cảnh): KHÔNG ghi nhận sự kiện kết thúc trong khoảng này.
           - Pha bóng bị cắt ở cuối video chưa thấy kết quả thì bỏ qua (clip sau sẽ ghi nhận).
        """
        return self.gemini.analyze_video_clip(clip_path, prompt)

    def gemini_worker_loop(self):
        while self.running:
            try:
                future, clip_path, duration_sec = self.clip_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                self.gemini_status = f"Đang chờ Gemini ({self.clip_queue.qsize() + 1} clip) qua {self.gemini.active_label}..."
                result = future.result()
                if result:
                    self.engine.update_time_played(
                        duration_sec, result.active_players_team_a, result.active_players_team_b
                    )
                    for ev in result.events:
                        self.engine.record_event(ev)
                    self.gemini_status = f"Đã cập nhật {len(result.events)} sự kiện ({self.gemini.active_label})"
            except Exception as e:
                self.gemini_status = f"Cả 2 API đều lỗi: {str(e)[:35]}"
                print(f"[LỖI GEMINI] {e}")
            finally:
                if os.path.exists(clip_path):
                    os.remove(clip_path)
                self.clip_queue.task_done()

    def tracking_and_recording_loop(self):
        target_fps = 15.0
        frame_interval = 1.0 / target_fps
        frames_per_clip = int(target_fps * 8)  # 8 giây / clip
        frame_size = (960, 540)

        clip_idx = 0
        writer = None
        temp_clip_path = ""
        recorded_frames = 0
        active_frames = 0

        while self.running:
            t_start = time.time()
            raw = self.obs.get_frame(frame_size)

            # Vẽ Hitbox cầu thủ để truyền lên Chrome
            annotated, player_count = self._draw_hitboxes(raw.copy())
            ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok:
                self.latest_jpeg = buf.tobytes()

            # Ghi clip gốc (không dính chữ HUD/Hitbox) gửi lên Gemini để đọc số áo rõ nhất
            if writer is None:
                temp_clip_path = f"temp_obs_clip_{clip_idx}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(temp_clip_path, fourcc, target_fps, frame_size)
                recorded_frames = 0
                active_frames = 0
                clip_idx += 1

            writer.write(raw)
            recorded_frames += 1
            if player_count > 0:
                active_frames += 1

            if recorded_frames >= frames_per_clip:
                writer.release()
                writer = None
                if active_frames > (frames_per_clip * 0.2):
                    if self.clip_queue.full():
                        print("[CẢNH BÁO] Gemini xử lý chậm hơn tốc độ ghi, bỏ 1 clip.")
                    else:
                        future = self.pool.submit(self._analyze_clip, temp_clip_path)
                        self.clip_queue.put((future, temp_clip_path, recorded_frames / target_fps))
                elif os.path.exists(temp_clip_path):
                    os.remove(temp_clip_path)

            elapsed = time.time() - t_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

        if writer:
            writer.release()


# =====================================================================
# PHẦN 4: GIAO DIỆN WEB TRÊN CHROME (OBS HITBOX + FIBA BOX SCORE)
# =====================================================================

app = Flask(__name__)
vision_system: Optional[OBSVisionSystem] = None
boxscore_engine: Optional[FIBABoxScoreEngine] = None

def generate_boxscore_tables_html(engine: FIBABoxScoreEngine) -> str:
    with engine.lock:
        def render_team(code: str) -> str:
            team = engine.teams[code]
            rows = ""
            tot_fg_m = tot_fg_a = tot_2p_m = tot_2p_a = tot_3p_m = tot_3p_a = 0
            tot_ft_m = tot_ft_a = tot_or = tot_dr = tot_reb = 0
            tot_as = tot_to = tot_st = tot_bs = tot_pf = tot_fd = tot_pts = 0
            tot_sec = 0.0

            for no, p in sorted(team["players"].items()):
                prefix = "<span class='star-badge'>★</span> " if p.is_starter else "<span class='bench-dot'>•</span> "
                cap = " <span class='cap-badge'>C</span>" if p.is_captain else ""
                if p.min_formatted == "DNP":
                    rows += f"<tr class='dnp-row'><td>{prefix}{p.no}</td><td class='left'>{p.name}{cap}</td><td colspan='19' class='left dnp-label'>DNP - Không thi đấu</td></tr>"
                    continue

                tot_sec += p.seconds_played
                tot_fg_m += p.fg_m; tot_fg_a += p.fg_a
                tot_2p_m += p.fg2_m; tot_2p_a += p.fg2_a
                tot_3p_m += p.fg3_m; tot_3p_a += p.fg3_a
                tot_ft_m += p.ft_m; tot_ft_a += p.ft_a
                tot_or += p.or_reb; tot_dr += p.dr_reb; tot_reb += p.tot_reb
                tot_as += p.assists; tot_to += p.turnovers; tot_st += p.steals; tot_bs += p.blocks
                tot_pf += p.pf; tot_fd += p.fd; tot_pts += p.pts

                # Định dạng màu cho +/-
                pm_cls = "pm-pos" if p.plus_minus > 0 else ("pm-neg" if p.plus_minus < 0 else "pm-zero")
                pm_sign = "+" if p.plus_minus > 0 else ""

                rows += f"""
                <tr>
                    <td class="col-no">{prefix}<b>#{p.no}</b></td>
                    <td class="left col-name">{p.name}{cap}</td>
                    <td class="col-min">{p.min_formatted}</td>
                    <td>{p.fg_m}/{p.fg_a}</td><td class="col-pct">{p.calc_pct(p.fg_m, p.fg_a)}%</td>
                    <td>{p.fg2_m}/{p.fg2_a}</td><td class="col-pct">{p.calc_pct(p.fg2_m, p.fg2_a)}%</td>
                    <td>{p.fg3_m}/{p.fg3_a}</td><td class="col-pct">{p.calc_pct(p.fg3_m, p.fg3_a)}%</td>
                    <td>{p.ft_m}/{p.ft_a}</td><td class="col-pct">{p.calc_pct(p.ft_m, p.ft_a)}%</td>
                    <td>{p.or_reb}</td><td>{p.dr_reb}</td><td class="col-totreb"><b>{p.tot_reb}</b></td>
                    <td>{p.assists}</td><td>{p.turnovers}</td><td>{p.steals}</td><td>{p.blocks}</td>
                    <td>{p.pf}</td><td>{p.fd}</td>
                    <td><span class="pm-pill {pm_cls}">{pm_sign}{p.plus_minus}</span></td>
                    <td class="col-pts"><b>{p.pts}</b></td>
                </tr>"""

            tot_min = f"{int(tot_sec // 60):02d}:{int(tot_sec % 60):02d}"
            opp = engine.team_codes[1] if code == engine.team_codes[0] else engine.team_codes[0]
            pm = team["score"] - engine.teams[opp]["score"]
            tot_pm_cls = "pm-pos" if pm > 0 else ("pm-neg" if pm < 0 else "pm-zero")
            tot_pm_sign = "+" if pm > 0 else ""

            return f"""
            <div class="team-section">
                <div class="team-header-bar">
                    <div class="team-title-wrap">
                        <span class="team-badge-pill">{code}</span>
                        <span class="team-header-name">{team['name']}</span>
                    </div>
                    <div class="team-color-indicator">Màu áo: <b>{team['color']}</b></div>
                </div>
                <div class="table-responsive">
                    <table class="fiba-table">
                        <thead>
                            <tr class="th-group">
                                <th rowspan="2" class="col-no">No</th>
                                <th rowspan="2" class="left col-name">Cầu thủ</th>
                                <th rowspan="2" class="col-min">Min</th>
                                <th colspan="2">Field Goals</th>
                                <th colspan="2">2 Points</th>
                                <th colspan="2">3 Points</th>
                                <th colspan="2">Free Throws</th>
                                <th colspan="3">Rebounds</th>
                                <th rowspan="2">AS</th>
                                <th rowspan="2">TO</th>
                                <th rowspan="2">ST</th>
                                <th rowspan="2">BS</th>
                                <th colspan="2">Fouls</th>
                                <th rowspan="2">+/-</th>
                                <th rowspan="2" class="col-pts">PTS</th>
                            </tr>
                            <tr class="th-sub">
                                <th>M/A</th><th>%</th>
                                <th>M/A</th><th>%</th>
                                <th>M/A</th><th>%</th>
                                <th>M/A</th><th>%</th>
                                <th>OR</th><th>DR</th><th>TOT</th>
                                <th>PF</th><th>FD</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows}
                            <tr class="totals-row">
                                <td colspan="2" class="left"><b>TỔNG CỘNG</b></td>
                                <td class="col-min">{tot_min}</td>
                                <td>{tot_fg_m}/{tot_fg_a}</td><td class="col-pct">{PlayerStats.calc_pct(tot_fg_m, tot_fg_a)}%</td>
                                <td>{tot_2p_m}/{tot_2p_a}</td><td class="col-pct">{PlayerStats.calc_pct(tot_2p_m, tot_2p_a)}%</td>
                                <td>{tot_3p_m}/{tot_3p_a}</td><td class="col-pct">{PlayerStats.calc_pct(tot_3p_m, tot_3p_a)}%</td>
                                <td>{tot_ft_m}/{tot_ft_a}</td><td class="col-pct">{PlayerStats.calc_pct(tot_ft_m, tot_ft_a)}%</td>
                                <td>{tot_or}</td><td>{tot_dr}</td><td class="col-totreb"><b>{tot_reb}</b></td>
                                <td>{tot_as}</td><td>{tot_to}</td><td>{tot_st}</td><td>{tot_bs}</td>
                                <td>{tot_pf}</td><td>{tot_fd}</td>
                                <td><span class="pm-pill {tot_pm_cls}">{tot_pm_sign}{pm}</span></td>
                                <td class="col-pts"><b>{tot_pts}</b></td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>"""

        c1, c2 = engine.team_codes[0], engine.team_codes[1]
        t1, t2 = engine.teams[c1], engine.teams[c2]
        a1, a2 = engine.adv_stats[c1], engine.adv_stats[c2]

        def fmt_paint(adv):
            pct = PlayerStats.calc_pct(adv["pts_in_paint_m"], adv["pts_in_paint_a"])
            return f"{adv['pts_in_paint_pts']} pts ({adv['pts_in_paint_m']}/{adv['pts_in_paint_a']}) {pct}%"

        def fmt_time(sec):
            return f"{int(sec // 60):02d}:{int(sec % 60):02d}"

        return f"""
        <!-- SCOREBOARD BROADCAST BANNER -->
        <div class="score-banner-card">
            <div class="banner-top-meta">
                <span class="live-pill"><span class="pulse-dot"></span> FIBA LIVE {engine.match_format.upper()} TRACKER</span>
                <span class="banner-sub-text">BẢNG ĐIỂM TRỰC TIẾP TỪ OBS VISION AI</span>
            </div>
            <div class="scoreboard-grid">
                <div class="team-side team-side-a">
                    <div class="team-meta-box">
                        <span class="team-code-tag">{c1}</span>
                        <div class="team-full-name">{t1['name']}</div>
                        <div class="team-jersey-tag">Áo: {t1['color']}</div>
                    </div>
                    <div class="score-digit">{t1['score']}</div>
                </div>
                <div class="score-center-divider">
                    <div class="vs-text">VS</div>
                    <div class="lead-change-badge">Lead Changes: {engine.lead_changes}</div>
                </div>
                <div class="team-side team-side-b">
                    <div class="score-digit">{t2['score']}</div>
                    <div class="team-meta-box meta-right">
                        <span class="team-code-tag">{c2}</span>
                        <div class="team-full-name">{t2['name']}</div>
                        <div class="team-jersey-tag">Áo: {t2['color']}</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- TEAM TABLES -->
        {render_team(c1)}
        {render_team(c2)}

        <!-- ADVANCED STATS & GAME FLOW -->
        <div class="summary-grid">
            <div class="summary-card">
                <div class="card-header-bar">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20V10M18 20V4M6 20v-4"/></svg>
                    <span>CHỈ SỐ NÂNG CAO (ADVANCED STATS)</span>
                </div>
                <table class="summary-table">
                    <thead><tr><th class="left">Hạng mục</th><th>{c1}</th><th>{c2}</th></tr></thead>
                    <tbody>
                        <tr><td class="left">Points from Turnovers</td><td><b>{a1['pts_from_to']}</b></td><td><b>{a2['pts_from_to']}</b></td></tr>
                        <tr><td class="left">Points in the Paint</td><td>{fmt_paint(a1)}</td><td>{fmt_paint(a2)}</td></tr>
                        <tr><td class="left">Second Chance Points</td><td><b>{a1['second_chance_pts']}</b></td><td><b>{a2['second_chance_pts']}</b></td></tr>
                        <tr><td class="left">Fast Break Points</td><td><b>{a1['fast_break_pts']}</b></td><td><b>{a2['fast_break_pts']}</b></td></tr>
                        <tr><td class="left">Bench Points</td><td><b>{a1['bench_pts']}</b></td><td><b>{a2['bench_pts']}</b></td></tr>
                    </tbody>
                </table>
            </div>

            <div class="summary-card">
                <div class="card-header-bar">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                    <span>DIỄN BIẾN TRẬN ĐẤU (GAME FLOW)</span>
                </div>
                <table class="summary-table">
                    <thead><tr><th class="left">Chỉ số</th><th>{c1}</th><th>{c2}</th></tr></thead>
                    <tbody>
                        <tr><td class="left">Biggest Lead</td><td><b>{a1['biggest_lead']}</b> <span class="score-sub">({a1['biggest_lead_score']})</span></td><td><b>{a2['biggest_lead']}</b> <span class="score-sub">({a2['biggest_lead_score']})</span></td></tr>
                        <tr><td class="left">Scoring Run lớn nhất</td><td><b>{a1['biggest_run']}</b> pts <span class="score-sub">({a1['biggest_run_score']})</span></td><td><b>{a2['biggest_run']}</b> pts <span class="score-sub">({a2['biggest_run_score']})</span></td></tr>
                        <tr><td class="left">Số lần đổi dẫn (Lead Changes)</td><td colspan="2" class="center-highlight">{engine.lead_changes} lần</td></tr>
                        <tr><td class="left">Số lần hòa điểm (Times Tied)</td><td colspan="2" class="center-highlight">{engine.times_tied} lần</td></tr>
                        <tr><td class="left">Thời gian dẫn điểm</td><td><b>{fmt_time(a1['time_with_lead_sec'])}</b></td><td><b>{fmt_time(a2['time_with_lead_sec'])}</b></td></tr>
                    </tbody>
                </table>
            </div>
        </div>
        """

@app.route("/")
def index():
    html = """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>FIBA Box Score & OBS Live AI Tracking</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:ital,wght@0,600;0,700;0,800;1,700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-main: #0a0d14;
                --bg-card: #121824;
                --bg-card-subtle: #182030;
                --border-color: rgba(255, 255, 255, 0.08);
                --text-main: #f1f5f9;
                --text-muted: #94a3b8;
                --accent-amber: #f59e0b;
                --accent-emerald: #10b981;
                --accent-rose: #f43f5e;
                --accent-cyan: #06b6d4;
                --accent-blue: #3b82f6;
            }

            * { box-sizing: border-box; }
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                font-size: 13px;
                margin: 0;
                padding: 20px;
                background: var(--bg-main);
                color: var(--text-main);
                -webkit-font-smoothing: antialiased;
            }

            /* CONTAINER WRAPPER */
            .main-shell {
                max-width: 1280px;
                margin: 0 auto;
                display: flex;
                flex-direction: column;
                gap: 20px;
            }

            /* OBS CAM CARD */
            .cam-card {
                background: var(--bg-card);
                border: 1px solid var(--border-color);
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 10px 30px -10px rgba(0,0,0,0.6);
            }

            .cam-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 12px 18px;
                background: #0f141f;
                border-bottom: 1px solid var(--border-color);
            }

            .cam-title {
                display: flex;
                align-items: center;
                gap: 8px;
                font-family: 'Barlow Condensed', sans-serif;
                font-size: 18px;
                font-weight: 700;
                letter-spacing: 0.8px;
                color: #e2e8f0;
            }

            .cam-badges {
                display: flex;
                align-items: center;
                gap: 8px;
            }

            .badge {
                padding: 4px 10px;
                border-radius: 6px;
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 0.5px;
                text-transform: uppercase;
            }
            .badge-yolo { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
            .badge-ai { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }

            .obs-feed-container {
                position: relative;
                max-width: 960px;
                margin: 16px auto;
                background: #000;
                border-radius: 8px;
                overflow: hidden;
                border: 1px solid rgba(255, 255, 255, 0.12);
                box-shadow: 0 4px 20px rgba(0,0,0,0.8);
            }

            .obs-feed-container img {
                display: block;
                width: 100%;
                height: auto;
                aspect-ratio: 16 / 9;
                object-fit: contain;
            }

            .cam-footer {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 12px 20px;
                background: #0f141f;
                border-top: 1px solid var(--border-color);
            }

            .sync-indicator {
                display: flex;
                align-items: center;
                gap: 6px;
                color: var(--text-muted);
                font-size: 12px;
            }

            .pulse-dot {
                display: inline-block;
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #ef4444;
                box-shadow: 0 0 8px #ef4444;
                animation: pulse 1.5s infinite;
            }

            @keyframes pulse {
                0% { opacity: 1; transform: scale(1); }
                50% { opacity: 0.4; transform: scale(1.15); }
                100% { opacity: 1; transform: scale(1); }
            }

            .btn-print {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 8px 18px;
                background: #2563eb;
                color: #fff;
                border: none;
                border-radius: 6px;
                font-family: 'Inter', sans-serif;
                font-size: 12px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s;
            }
            .btn-print:hover { background: #1d4ed8; transform: translateY(-1px); }

            /* SCORE BANNER */
            .score-banner-card {
                background: linear-gradient(135deg, #161e2e 0%, #101622 100%);
                border: 1px solid var(--border-color);
                border-radius: 12px;
                padding: 18px 24px;
                box-shadow: 0 8px 24px -8px rgba(0,0,0,0.5);
            }

            .banner-top-meta {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 12px;
            }

            .live-pill {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                background: rgba(239, 68, 68, 0.15);
                color: #f87171;
                border: 1px solid rgba(239, 68, 68, 0.3);
                padding: 4px 10px;
                border-radius: 20px;
                font-family: 'Barlow Condensed', sans-serif;
                font-size: 13px;
                font-weight: 700;
                letter-spacing: 0.8px;
            }

            .banner-sub-text {
                font-size: 11px;
                color: var(--text-muted);
                letter-spacing: 0.5px;
            }

            .scoreboard-grid {
                display: grid;
                grid-template-columns: 1fr auto 1fr;
                align-items: center;
                gap: 20px;
            }

            .team-side {
                display: flex;
                align-items: center;
                gap: 18px;
            }
            .team-side-a { justify-content: flex-end; }
            .team-side-b { justify-content: flex-start; }

            .team-meta-box {
                display: flex;
                flex-direction: column;
                gap: 2px;
            }
            .team-meta-box.meta-right { align-items: flex-start; text-align: left; }
            .team-side-a .team-meta-box { align-items: flex-end; text-align: right; }

            .team-code-tag {
                display: inline-block;
                background: rgba(255, 255, 255, 0.1);
                color: #e2e8f0;
                font-family: 'Barlow Condensed', sans-serif;
                font-size: 12px;
                font-weight: 700;
                padding: 2px 8px;
                border-radius: 4px;
                width: fit-content;
            }

            .team-full-name {
                font-family: 'Barlow Condensed', sans-serif;
                font-size: 26px;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                color: #ffffff;
            }

            .team-jersey-tag {
                font-size: 11px;
                color: var(--text-muted);
            }

            .score-digit {
                font-family: 'Barlow Condensed', sans-serif;
                font-size: 64px;
                font-weight: 800;
                line-height: 1;
                color: var(--accent-amber);
                text-shadow: 0 0 20px rgba(245, 158, 11, 0.25);
                font-variant-numeric: tabular-nums;
            }

            .score-center-divider {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 4px;
                padding: 0 10px;
            }

            .vs-text {
                font-family: 'Barlow Condensed', sans-serif;
                font-size: 22px;
                font-weight: 800;
                color: #64748b;
                letter-spacing: 2px;
            }

            .lead-change-badge {
                font-size: 11px;
                color: var(--text-muted);
                background: rgba(255, 255, 255, 0.05);
                padding: 2px 8px;
                border-radius: 4px;
                white-space: nowrap;
            }

            /* TEAM BOX SCORE SECTIONS */
            .team-section {
                background: var(--bg-card);
                border: 1px solid var(--border-color);
                border-radius: 12px;
                overflow: hidden;
                margin-top: 16px;
                box-shadow: 0 4px 16px rgba(0,0,0,0.3);
            }

            .team-header-bar {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 12px 18px;
                background: #151d2c;
                border-bottom: 1px solid var(--border-color);
            }

            .team-title-wrap {
                display: flex;
                align-items: center;
                gap: 10px;
            }

            .team-badge-pill {
                background: var(--accent-blue);
                color: #fff;
                font-family: 'Barlow Condensed', sans-serif;
                font-size: 14px;
                font-weight: 700;
                padding: 3px 10px;
                border-radius: 4px;
            }

            .team-header-name {
                font-family: 'Barlow Condensed', sans-serif;
                font-size: 20px;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                color: #fff;
            }

            .team-color-indicator {
                font-size: 12px;
                color: var(--text-muted);
            }

            .table-responsive {
                overflow-x: auto;
                width: 100%;
            }

            .fiba-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 12px;
                text-align: center;
                font-variant-numeric: tabular-nums;
            }

            .fiba-table th, .fiba-table td {
                padding: 6px 8px;
                border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            }

            .fiba-table thead tr.th-group th {
                background: #182234;
                color: #94a3b8;
                font-size: 11px;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            }

            .fiba-table thead tr.th-sub th {
                background: #141c2b;
                color: #64748b;
                font-size: 10px;
                font-weight: 600;
                text-transform: uppercase;
            }

            .fiba-table tbody tr:hover {
                background: rgba(255, 255, 255, 0.03);
            }

            .fiba-table .left { text-align: left; }
            .col-no { width: 50px; font-weight: 600; }
            .col-name { min-width: 140px; font-weight: 600; color: #fff; }
            .col-min { font-weight: 500; color: #cbd5e1; }
            .col-pct { color: #94a3b8; font-size: 11px; }
            .col-totreb { color: #38bdf8; }
            .col-pts { font-size: 13px; color: var(--accent-amber); }

            .star-badge { color: #facc15; font-size: 11px; }
            .bench-dot { color: #64748b; font-size: 14px; }
            .cap-badge {
                display: inline-block;
                background: #eab308;
                color: #000;
                font-size: 9px;
                font-weight: 800;
                padding: 1px 4px;
                border-radius: 3px;
                vertical-align: middle;
            }

            .pm-pill {
                display: inline-block;
                padding: 1px 6px;
                border-radius: 4px;
                font-weight: 700;
                font-size: 11px;
            }
            .pm-pos { background: rgba(16, 185, 129, 0.15); color: #34d399; }
            .pm-neg { background: rgba(244, 63, 94, 0.15); color: #fb7185; }
            .pm-zero { color: #64748b; }

            .totals-row {
                background: #182234 !important;
                font-weight: 700;
                border-top: 2px solid rgba(255, 255, 255, 0.15);
            }
            .totals-row td { color: #fff; padding: 8px; }

            .dnp-row td { color: #64748b !important; }
            .dnp-label { font-style: italic; font-size: 11px; }

            /* SUMMARY GRID */
            .summary-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 16px;
                margin-top: 16px;
            }

            .summary-card {
                background: var(--bg-card);
                border: 1px solid var(--border-color);
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 4px 16px rgba(0,0,0,0.3);
            }

            .card-header-bar {
                display: flex;
                align-items: center;
                gap: 8px;
                padding: 10px 16px;
                background: #151d2c;
                border-bottom: 1px solid var(--border-color);
                font-family: 'Barlow Condensed', sans-serif;
                font-size: 15px;
                font-weight: 700;
                letter-spacing: 0.6px;
                color: #e2e8f0;
            }

            .summary-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 12px;
                text-align: center;
                font-variant-numeric: tabular-nums;
            }

            .summary-table th, .summary-table td {
                padding: 8px 12px;
                border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            }

            .summary-table th {
                background: #141c2b;
                color: #94a3b8;
                font-size: 11px;
                text-transform: uppercase;
            }

            .summary-table .left { text-align: left; color: #cbd5e1; }
            .score-sub { color: #94a3b8; font-size: 11px; font-weight: normal; }
            .center-highlight { font-weight: 700; color: #38bdf8; }

            /* PRINT STYLES */
            @media print {
                body { background: #fff !important; color: #000 !important; font-size: 11px; padding: 0; }
                .cam-card { display: none !important; }
                .main-shell { max-width: 100% !important; }
                .score-banner-card { background: #fff !important; border: 2px solid #000 !important; color: #000 !important; box-shadow: none !important; }
                .team-full-name { color: #000 !important; }
                .score-digit { color: #000 !important; text-shadow: none !important; }
                .team-section { background: #fff !important; border: 1px solid #000 !important; box-shadow: none !important; }
                .team-header-bar { background: #eee !important; border-bottom: 1px solid #000 !important; color: #000 !important; }
                .team-header-name { color: #000 !important; }
                .fiba-table { border: 1px solid #000 !important; font-size: 10px; }
                .fiba-table th, .fiba-table td { border: 1px solid #333 !important; color: #000 !important; padding: 3px 4px; }
                .fiba-table thead tr.th-group th { background: #eee !important; color: #000 !important; }
                .fiba-table thead tr.th-sub th { background: #f5f5f5 !important; color: #000 !important; }
                .totals-row { background: #eee !important; }
                .totals-row td { color: #000 !important; font-weight: bold; }
                .col-name { color: #000 !important; }
                .col-pts { color: #000 !important; font-weight: bold; }
                .summary-card { background: #fff !important; border: 1px solid #000 !important; box-shadow: none !important; }
                .card-header-bar { background: #eee !important; color: #000 !important; }
                .summary-table th, .summary-table td { border: 1px solid #333 !important; color: #000 !important; }
            }
        </style>
    </head>
    <body>
        <div class="main-shell">
            <!-- CAMERA & TRACKING VIEWER -->
            <div class="cam-card">
                <div class="cam-header">
                    <div class="cam-title">
                        <span class="pulse-dot"></span>
                        <span>OBS STUDIO • HITBOX & BALL TRACKING</span>
                    </div>
                    <div class="cam-badges">
                        <span class="badge badge-yolo">YOLOv8 TRACKING</span>
                        <span class="badge badge-ai">GEMINI 3.1 FLASH-LITE</span>
                    </div>
                </div>
                <div class="obs-feed-container">
                    <img src="/video_feed" alt="Đang kết nối luồng OBS Studio..." />
                </div>
                <div class="cam-footer">
                    <div class="sync-indicator">
                        <span class="pulse-dot"></span> Tự động đồng bộ số liệu mỗi 2s
                    </div>
                    <button class="btn-print" onclick="window.print()">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 9V2h12v7M6 18H4a2 2 0 01-2-2v-5a2 2 0 012-2h16a2 2 0 012 2v5a2 2 0 01-2 2h-2"/><rect x="6" y="14" width="12" height="8"/></svg>
                        In / Xuất PDF Biên Bản FIBA
                    </button>
                </div>
            </div>

            <!-- LIVE FIBA BOX SCORE CONTAINER -->
            <div id="boxscore-container">Đang tải dữ liệu trực tiếp từ OBS...</div>
        </div>

        <script>
            function refreshBoxScore() {
                fetch('/boxscore_data').then(r => r.text()).then(html => {
                    document.getElementById('boxscore-container').innerHTML = html;
                }).catch(err => console.error("Lỗi cập nhật box score:", err));
            }
            setInterval(refreshBoxScore, 2000);
            refreshBoxScore();
        </script>
    </body>
    </html>
    """
    return html

@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            if vision_system and vision_system.latest_jpeg:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + vision_system.latest_jpeg + b"\r\n")
            time.sleep(0.04)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/boxscore_data")
def boxscore_data():
    return generate_boxscore_tables_html(boxscore_engine) if boxscore_engine else ""


# =====================================================================
# PHẦN 5: KHÓA KHỞI ĐỘNG CONSOLE (BẮT BUỘC ĐIỀN ĐỦ OBS + 2 API KEY)
# =====================================================================

def parse_obs_input(raw_str: str):
    val = raw_str.strip()
    if val.isdigit():
        return int(val)  # OBS Virtual Camera index (VD: 1 hoặc 2)
    return val           # OBS Stream URL (VD: rtsp://127.0.0.1:8554/live hoặc udp://...)


def collect_required_config() -> tuple[object, str, str]:
    """Bắt buộc điền đủ và kiểm tra kết nối Luồng OBS + API 1 + API 2 mới cho phép chạy"""
    while True:
        print("\n--- BƯỚC 1: CẤU HÌNH BẮT BUỘC (PHẢI ĐIỀN ĐỦ MỚI BẮT ĐẦU CHẠY) ---")

        # 1. Điền và kiểm tra luồng OBS
        obs_source = None
        while obs_source is None:
            raw_obs = input(">> [1/3] Nhập luồng OBS (Số Virtual Cam VD: 1, 2 | hoặc link rtsp://, udp://): ").strip()
            if not raw_obs:
                print("   [!] Không được bỏ trống luồng OBS. Vui lòng nhập lại.")
                continue

            candidate = parse_obs_input(raw_obs)
            print(f"   [*] Đang kiểm tra tín hiệu từ OBS ({candidate})...")
            test_cap = open_obs_capture(candidate)
            if test_cap.isOpened():
                ret, _ = test_cap.read()
                test_cap.release()
                if ret:
                    print("   [OK] Đã nhận được khung hình trực tiếp từ OBS!")
                    obs_source = candidate
                else:
                    print("   [!] Đã mở được cổng OBS nhưng chưa có khung hình. Hãy kiểm tra nút 'Start Virtual Camera' hoặc 'Start Streaming' trong OBS.")
            else:
                print("   [!] Không thể kết nối luồng OBS này. Vui lòng kiểm tra lại số thứ tự camera hoặc địa chỉ stream.")

        # 2. Điền API Gemini 1 (Chính)
        api_key_1 = ""
        while not api_key_1:
            api_key_1 = input(">> [2/3] Nhập Gemini API Key 1 (Chính): ").strip()
            if not api_key_1:
                print("   [!] Bắt buộc nhập Gemini API Key 1.")

        # 3. Điền API Gemini 2 (Dự bị)
        api_key_2 = ""
        while not api_key_2:
            api_key_2 = input(">> [3/3] Nhập Gemini API Key 2 (Dự bị khi API 1 hết hạn mức): ").strip()
            if not api_key_2:
                print("   [!] Bắt buộc nhập Gemini API Key 2 dự bị.")
            elif api_key_2 == api_key_1:
                print("   [Cảnh báo] API 2 đang trùng với API 1 (vẫn chấp nhận, nhưng nên dùng 2 key khác Project để dự phòng quota).")

        # Xác nhận tổng hợp trước khi chạy
        print("\n=== XÁC NHẬN CẤU HÌNH ĐÃ ĐIỀN ĐỦ ===")
        print(f"  + Nguồn OBS : {obs_source}")
        print(f"  + API 1     : {api_key_1[:8]}...{api_key_1[-4:]}")
        print(f"  + API 2     : {api_key_2[:8]}...{api_key_2[-4:]}")
        confirm = input(">> Nhấn [Enter] để BẮT ĐẦU CHẠY (hoặc gõ 'n' để nhập lại): ").strip().lower()
        if confirm != "n":
            return obs_source, api_key_1, api_key_2

def console_input_player_names(engine: FIBABoxScoreEngine, team_code: str, detected_numbers: List[int]):
    team_name = engine.teams[team_code]["name"]
    print(f"\n--- THIẾT LẬP DANH SÁCH CẦU THỦ ĐỘI {team_name} ({team_code}) ---")
    print(f"Số áo AI quét được trên sân: {detected_numbers if detected_numbers else 'Chưa phát hiện số nào'}")

    extra = input(">> Nếu còn thiếu số áo, nhập thêm cách nhau bởi dấu phẩy (VD: 7, 15) hoặc Enter để bỏ qua: ").strip()
    if extra:
        for item in extra.split(","):
            if item.strip().isdigit():
                num = int(item.strip())
                if num not in detected_numbers:
                    detected_numbers.append(num)
    detected_numbers.sort()

    if not detected_numbers:
        print(f"[Thông báo] Chưa có số áo nào cho đội {team_name}. Hệ thống sẽ tự thêm khi phát hiện trong trận.")
        return

    print(f"Nhập tên cho từng số áo của đội {team_name} (thêm '(C)' cuối tên nếu là Đội trưởng):")
    for idx, num in enumerate(detected_numbers):
        raw_name = input(f"   + Số áo #{num} -> Tên cầu thủ: ").strip()
        if not raw_name:
            raw_name = f"CẦU THỦ #{num}"
        is_cap = "(C)" in raw_name.upper()
        clean_name = raw_name.replace("(C)", "").replace("(c)", "").strip()
        is_starter = (idx < engine.max_on_court)
        engine.register_player(team_code, num, clean_name, is_starter=is_starter, is_captain=is_cap)


if __name__ == "__main__":
    print("=" * 72)
    print(" HỆ THỐNG THỐNG KÊ BÓNG RỔ FIBA TỪ OBS STUDIO (DUAL GEMINI API FAILOVER)")
    print("=" * 72)

    # BƯỚC 1: Bắt buộc điền đủ Luồng OBS + API 1 + API 2 mới đi tiếp
    obs_source, api_1, api_2 = collect_required_config()

    # BƯỚC 2: Nhập thông tin 2 đội bóng & thể thức thi đấu
    print("\n--- BƯỚC 2: THÔNG TIN 2 ĐỘI THI ĐẤU & THỂ THỨC ---")
    fmt_choice = input(">> Thể thức thi đấu ([1] Bóng rổ 3x3 | [2] Bóng rổ 5x5) [Mặc định: 1 - 3x3]: ").strip()
    match_format = "5x5" if fmt_choice in ("2", "5x5", "5") else "3x3"
    print(f"    -> Đã chọn thể thức: Bóng rổ {match_format}")

    team_a_name = input(">> Tên Đội 1 (Enter để dùng 'HÀ NỘI'): ").strip() or "HÀ NỘI"
    team_a_code = input(">> Mã viết tắt Đội 1 (Enter để dùng 'HNB'): ").strip().upper() or "HNB"
    team_a_color = input(">> Màu áo Đội 1 (VD: Trắng): ").strip() or "Trắng"

    team_b_name = input("\n>> Tên Đội 2 (Enter để dùng 'BÌNH THUẬN'): ").strip() or "BÌNH THUẬN"
    team_b_code = input(">> Mã viết tắt Đội 2 (Enter để dùng 'BTB'): ").strip().upper() or "BTB"
    team_b_color = input(">> Màu áo Đội 2 (VD: Xanh đậm): ").strip() or "Xanh đậm"

    # Khởi tạo kết nối OBS, Engine FIBA và Gemini Failover
    obs_reader = AsyncOBSReader(obs_source)
    gemini_manager = GeminiFailoverManager(api_1, api_2)
    boxscore_engine = FIBABoxScoreEngine(
        team_a_code, team_a_name, team_a_color,
        team_b_code, team_b_name, team_b_color,
        match_format=match_format
    )
    vision_system = OBSVisionSystem(obs_reader, boxscore_engine, gemini_manager)

    # BƯỚC 3: Quét số áo từ luồng OBS & yêu cầu nhập tên cầu thủ
    input("\n>> Đảm bảo cầu thủ đang hiện trên OBS rồi nhấn [Enter] để AI quét số áo...")
    nums_a, nums_b = vision_system.scan_jerseys_pre_match()

    console_input_player_names(boxscore_engine, team_a_code, nums_a)
    console_input_player_names(boxscore_engine, team_b_code, nums_b)

    # BƯỚC 4: Chạy luồng Hitbox + Gemini & tự động mở Chrome
    print("\n[Hệ thống] Đã hoàn tất thiết lập! Đang khởi chạy theo dõi trực tiếp...")
    threading.Thread(target=vision_system.tracking_and_recording_loop, daemon=True).start()
    threading.Thread(target=vision_system.gemini_worker_loop, daemon=True).start()

    local_url = "http://127.0.0.1:5000"
    print(f"[Hệ thống] Đang tự động mở trình duyệt tại: {local_url}")
    threading.Timer(1.5, webbrowser.open, args=[local_url]).start()

    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)