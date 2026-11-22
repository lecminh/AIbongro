import os
import glob
import time
import queue
import threading
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import cv2
import json
import numpy as np
from typing import List, Optional, Literal, Union
from dataclasses import dataclass
from flask import Flask, Response, request, jsonify
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
    court_polygon: Optional[List[List[float]]] = Field(
        default=None,
        description="Tọa độ 4 góc ranh giới mặt sân thi đấu theo vạch biên [x, y] tỷ lệ từ 0.0 đến 1.0 (thứ tự: Top-Left, Top-Right, Bottom-Right, Bottom-Left)"
    )

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


class CourtBoundaryManager:
    """Quản lý ranh giới sân đấu (Đa giác N-điểm bám mép màn hình) + Vùng loại trừ (Exclusion Zones).
    Tự động lưu/tải từ court_config.json, hỗ trợ AI quét vạch và căn chỉnh tương tác trên web."""
    def __init__(self, match_format: str = "3x3", config_path: str = "court_config.json"):
        self.config_path = config_path
        self.match_format = match_format
        self.lock = threading.Lock()
        self.exclusion_zones: List[List[float]] = []  # Danh sách [x1, y1, x2, y2] tỷ lệ 0.0 -> 1.0

        # Tọa độ tỷ lệ mặc định (Ngũ giác 5 điểm cho 3x3 ôm trọn đáy màn hình khi vạch sân tràn khung hình)
        if "3" in match_format:
            self.polygon = [
                (0.00, 0.42),  # Mép trái nơi vạch biên cắt cạnh trái màn hình
                (0.62, 0.24),  # Góc trên-phải baseline
                (1.00, 0.36),  # Mép phải nơi vạch biên cắt cạnh phải màn hình
                (1.00, 1.00),  # Góc dưới-phải màn hình (ôm trọn góc tràn)
                (0.00, 1.00),  # Góc dưới-trái màn hình (ôm trọn góc tràn)
            ]
        else:
            self.polygon = [
                (0.00, 0.22),
                (1.00, 0.22),
                (1.00, 1.00),
                (0.00, 1.00),
            ]
        self.load_from_file()

    def load_from_file(self) -> bool:
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    loaded = False
                    if "polygon" in data and len(data["polygon"]) >= 3:
                        self.polygon = [(float(p[0]), float(p[1])) for p in data["polygon"]]
                        loaded = True
                    if "exclusion_zones" in data and isinstance(data["exclusion_zones"], list):
                        self.exclusion_zones = [
                            [float(z[0]), float(z[1]), float(z[2]), float(z[3])]
                            for z in data["exclusion_zones"] if len(z) == 4
                        ]
                        loaded = True
                    if loaded:
                        print(f"[Court Boundary] Đã tải {len(self.polygon)} điểm sân & {len(self.exclusion_zones)} vùng loại trừ từ {self.config_path}")
                        return True
            except Exception as e:
                print(f"[Court Boundary] Lỗi đọc {self.config_path}: {e}")
        return False

    def save_to_file(self):
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump({
                    "match_format": self.match_format,
                    "polygon": self.polygon,
                    "exclusion_zones": self.exclusion_zones,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=2)
            print(f"[Court Boundary] Đã lưu cấu hình sân ({len(self.polygon)} điểm, {len(self.exclusion_zones)} vùng loại trừ) vào {self.config_path}")
        except Exception as e:
            print(f"[Court Boundary] Lỗi lưu {self.config_path}: {e}")

    def set_polygon(self, pts: List[Union[List[float], tuple]]):
        if len(pts) < 3:
            return
        with self.lock:
            self.polygon = [
                (max(0.0, min(1.0, float(p[0]))), max(0.0, min(1.0, float(p[1]))))
                for p in pts
            ]
            self.save_to_file()

    def set_config(self, pts: Optional[List[Union[List[float], tuple]]] = None,
                   exclusion_zones: Optional[List[List[float]]] = None):
        with self.lock:
            if pts is not None and len(pts) >= 3:
                self.polygon = [
                    (max(0.0, min(1.0, float(p[0]))), max(0.0, min(1.0, float(p[1]))))
                    for p in pts
                ]
            if exclusion_zones is not None:
                clean_zones = []
                for z in exclusion_zones:
                    if len(z) == 4:
                        x1, y1, x2, y2 = [max(0.0, min(1.0, float(v))) for v in z]
                        clean_zones.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
                self.exclusion_zones = clean_zones
            self.save_to_file()

    def get_pixel_polygon(self, width: int, height: int) -> np.ndarray:
        with self.lock:
            return np.array([
                [int(px * width), int(py * height)] for px, py in self.polygon
            ], dtype=np.int32)

    def get_pixel_exclusions(self, width: int, height: int) -> List[tuple]:
        with self.lock:
            return [
                (int(z[0] * width), int(z[1] * height), int(z[2] * width), int(z[3] * height))
                for z in self.exclusion_zones
            ]

    def is_in_court(self, foot_x: float, foot_y: float, width: int, height: int) -> bool:
        fx_ratio = foot_x / max(1, width)
        fy_ratio = foot_y / max(1, height)
        with self.lock:
            # Ưu tiên 1: Nếu bàn chân nằm trong bất kỳ Vùng Loại Trừ nào -> loại ngay
            for x1, y1, x2, y2 in self.exclusion_zones:
                if x1 <= fx_ratio <= x2 and y1 <= fy_ratio <= y2:
                    return False
            poly_pts = np.array([
                [int(px * width), int(py * height)] for px, py in self.polygon
            ], dtype=np.int32)
        return cv2.pointPolygonTest(poly_pts, (float(foot_x), float(foot_y)), False) >= 0


class PersistentPlayerReID:
    """Bộ gắn cố định mã VĐV (VDV-01..VDV-06 cho 3x3, VDV-01..VDV-10 cho 5x5).
    Lưu hồ sơ ngoại hình đa vùng (Áo 45% + Quần 25% + Giày 30%) + vị trí cuối
    để tự động khôi phục đúng mã cũ khi YOLO mất dấu/đổi track_id."""

    SLOT_COLORS = [
        (0, 230, 118),   # VDV-01: Xanh lá neon
        (0, 212, 255),   # VDV-02: Vàng cyan
        (255, 196, 0),   # VDV-03: Xanh dương sáng
        (94, 211, 244),  # VDV-04: Vàng FIBA
        (75, 150, 238),  # VDV-05: Cam thể thao
        (235, 130, 255), # VDV-06: Tím sáng
        (160, 255, 120), # VDV-07
        (255, 170, 120), # VDV-08
        (120, 220, 255), # VDV-09
        (255, 220, 180), # VDV-10
    ]

    def __init__(self, max_slots: int = 6):
        self.max_slots = max_slots
        self.lock = threading.Lock()
        self.slots: dict = {}          # slot_id -> {feat, last_pos, last_seen, alias, color}
        self.track_to_slot: dict = {}  # yolo_track_id -> slot_id
        self.frame_idx: int = 0

    def reset(self):
        with self.lock:
            self.slots.clear()
            self.track_to_slot.clear()
            self.frame_idx = 0
            print("[Re-ID] Đã đặt lại toàn bộ mã VĐV cố định.")

    def rename_slot(self, slot_id: str, alias: str):
        with self.lock:
            if slot_id in self.slots:
                self.slots[slot_id]["alias"] = alias.strip()

    @staticmethod
    def _extract_part_hist(crop_hsv: np.ndarray) -> np.ndarray:
        if crop_hsv.size == 0:
            return np.zeros(20, dtype=np.float32)
        h_hist = cv2.calcHist([crop_hsv], [0], None, [12], [0, 180]).flatten()
        s_hist = cv2.calcHist([crop_hsv], [1], None, [4], [0, 256]).flatten()
        v_hist = cv2.calcHist([crop_hsv], [2], None, [4], [0, 256]).flatten()
        vec = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 1e-6 else vec

    def extract_appearance(self, frame: np.ndarray, bbox: tuple) -> np.ndarray:
        """Trích xuất vân tay ngoại hình 3 vùng: Áo (45%) + Quần (25%) + Giày (30%)."""
        h_img, w_img = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        bw, bh = x2 - x1, y2 - y1
        if bw < 6 or bh < 12:
            return np.zeros(60, dtype=np.float32)

        # Cắt 60% chiều rộng ở giữa để loại bỏ nền sân hai bên
        mx1 = x1 + int(bw * 0.20)
        mx2 = x2 - int(bw * 0.20)
        if mx2 <= mx1:
            mx1, mx2 = x1, x2

        roi = frame[y1:y2, mx1:mx2]
        if roi.size == 0:
            return np.zeros(60, dtype=np.float32)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        rh = hsv.shape[0]
        upper = hsv[int(rh * 0.16):int(rh * 0.50), :]  # Áo thi đấu
        lower = hsv[int(rh * 0.50):int(rh * 0.78), :]  # Quần / chân
        shoes = hsv[int(rh * 0.78):int(rh * 0.98), :]  # Giày / bàn chân

        f_up = self._extract_part_hist(upper) * 0.45
        f_lo = self._extract_part_hist(lower) * 0.25
        f_sh = self._extract_part_hist(shoes) * 0.30
        feat = np.concatenate([f_up, f_lo, f_sh]).astype(np.float32)
        norm = float(np.linalg.norm(feat))
        return feat / norm if norm > 1e-6 else feat

    def assign_ids(self, frame: np.ndarray, detections: List[dict]) -> List[dict]:
        """Gán mã cố định (VDV-01..VDV-06) cho danh sách detection trong sân.
        detections: list of {'bbox': (x1,y1,x2,y2), 'track_id': int, 'conf': float}"""
        with self.lock:
            self.frame_idx += 1
            h_img, w_img = frame.shape[:2]
            diag = max(1.0, float(np.hypot(w_img, h_img)))

            # Chuẩn bị đặc trưng & vị trí bàn chân cho từng detection
            enriched = []
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                feat = self.extract_appearance(frame, det["bbox"])
                foot = ((x1 + x2) * 0.5, float(y2))
                enriched.append({**det, "feat": feat, "foot": foot, "slot_id": None})

            used_slots = set()

            # Bước 1: Giữ nguyên mã cho các yolo_track_id đang được track liên tục
            for item in enriched:
                tid = item["track_id"]
                if tid > 0 and tid in self.track_to_slot:
                    sid = self.track_to_slot[tid]
                    if sid in self.slots and sid not in used_slots:
                        item["slot_id"] = sid
                        used_slots.add(sid)

            # Bước 2: Khớp 1-1 (Re-ID) cho các detection mới / vừa mất dấu với các slot đang vắng mặt
            unassigned = [it for it in enriched if it["slot_id"] is None]
            available_slots = [sid for sid in self.slots.keys() if sid not in used_slots]

            if unassigned and available_slots:
                pairs = []
                for idx_u, item in enumerate(unassigned):
                    for sid in available_slots:
                        sdata = self.slots[sid]
                        app_sim = float(np.dot(item["feat"], sdata["feat"]))
                        dist_norm = float(np.hypot(
                            item["foot"][0] - sdata["last_pos"][0],
                            item["foot"][1] - sdata["last_pos"][1]
                        )) / diag
                        # Ưu tiên gần vị trí vừa biến mất trong ~45 frame (3 giây)
                        age = self.frame_idx - sdata["last_seen"]
                        spatial_sim = max(0.0, 1.0 - dist_norm * 2.5) if age <= 60 else max(0.0, 1.0 - dist_norm * 1.2)
                        score = 0.72 * app_sim + 0.28 * spatial_sim
                        pairs.append((score, idx_u, sid))

                pairs.sort(key=lambda x: x[0], reverse=True)
                matched_u = set()
                for score, idx_u, sid in pairs:
                    if idx_u in matched_u or sid in used_slots:
                        continue
                    # Nếu đã đủ số lượng slot tối đa -> bắt buộc tái sử dụng slot giống nhất;
                    # Nếu chưa đủ -> chỉ tái sử dụng khi độ giống ngoại hình + vị trí >= 0.52
                    if len(self.slots) >= self.max_slots or score >= 0.52:
                        unassigned[idx_u]["slot_id"] = sid
                        used_slots.add(sid)
                        matched_u.add(idx_u)

            # Bước 3: Nếu vẫn còn người chưa có mã và chưa chạm trần max_slots -> cấp mã cố định mới
            for item in enriched:
                if item["slot_id"] is None and len(self.slots) < self.max_slots:
                    new_idx = len(self.slots)
                    sid = f"VDV-{new_idx + 1:02d}"
                    self.slots[sid] = {
                        "feat": item["feat"].copy(),
                        "last_pos": item["foot"],
                        "last_seen": self.frame_idx,
                        "alias": sid,
                        "color": self.SLOT_COLORS[new_idx % len(self.SLOT_COLORS)],
                    }
                    item["slot_id"] = sid
                    used_slots.add(sid)

            # Bước 4: Cập nhật ánh xạ track_to_slot và EMA ngoại hình cho các mã đã khớp
            results_out = []
            for item in enriched:
                sid = item["slot_id"]
                if sid is None:
                    continue
                sdata = self.slots[sid]
                if item["track_id"] > 0:
                    self.track_to_slot[item["track_id"]] = sid
                # Cập nhật hồ sơ ngoại hình mượt (EMA) khi detection rõ nét
                if item["conf"] >= 0.50 and float(np.linalg.norm(item["feat"])) > 0.1:
                    blended = 0.85 * sdata["feat"] + 0.15 * item["feat"]
                    bnorm = float(np.linalg.norm(blended))
                    if bnorm > 1e-6:
                        sdata["feat"] = blended / bnorm
                sdata["last_pos"] = item["foot"]
                sdata["last_seen"] = self.frame_idx

                results_out.append({
                    "bbox": item["bbox"],
                    "conf": item["conf"],
                    "track_id": item["track_id"],
                    "slot_id": sid,
                    "label": sdata["alias"],
                    "color": sdata["color"],
                })
            return results_out


class SmartBallTracker:
    """Bộ nhận diện & bám bóng rổ thông minh trong sân:
    1) Lọc detection YOLO trong vùng sân + hình dạng tròn hợp lệ.
    2) Dự phòng bắt bóng theo màu (Cam 5x5 / Vàng 3x3) trong cửa sổ cục bộ khi YOLO bỏ sót.
    3) Nội suy quỹ đạo ngắn (Motion Coasting) khi bóng bị che khuất tạm thời 1-6 frame."""

    def __init__(self, trail_len: int = 14):
        self.trail = deque(maxlen=trail_len)
        self.last_pos: Optional[tuple] = None   # (cx, cy, radius)
        self.vel: tuple = (0.0, 0.0)            # (vx, vy) px/frame
        self.missed_frames: int = 999
        self.max_coast_frames: int = 6

    def _is_valid_ball_region(self, cx: float, cy: float, w: int, h: int, court_mgr: Optional[CourtBoundaryManager]) -> bool:
        if court_mgr is None:
            return True
        fx, fy = cx / max(1, w), cy / max(1, h)
        # Loại bỏ bóng giả nằm trong vùng loại trừ đỏ (khán giả, quay phim, logo góc)
        for ex1, ey1, ex2, ey2 in court_mgr.exclusion_zones:
            if ex1 <= fx <= ex2 and ey1 <= fy <= ey2:
                return False
        # Bóng có thể bay cao lên tới bảng rổ nên cho phép nằm trong sân HOẶC phía trên sân (y >= 5%)
        if court_mgr.is_in_court(cx, cy, w, h):
            return True
        poly = court_mgr.get_pixel_polygon(w, h)
        if len(poly) >= 3:
            min_x, max_x = int(poly[:, 0].min()), int(poly[:, 0].max())
            max_y = int(poly[:, 1].max())
            return (min_x - 30 <= cx <= max_x + 30) and (int(h * 0.04) <= cy <= max_y + 15)
        return True

    def _color_fallback_detect(self, frame: np.ndarray, court_mgr: Optional[CourtBoundaryManager]) -> Optional[tuple]:
        """Tìm bóng dự phòng theo màu bóng rổ (Cam 5x5 / Vàng 3x3) trong cửa sổ cục bộ quanh vị trí dự đoán."""
        h, w = frame.shape[:2]
        if self.last_pos is not None and self.missed_frames <= 12:
            pred_x = int(self.last_pos[0] + self.vel[0])
            pred_y = int(self.last_pos[1] + self.vel[1])
            pad = 75 + self.missed_frames * 14
            rx1, ry1 = max(0, pred_x - pad), max(0, pred_y - pad)
            rx2, ry2 = min(w, pred_x + pad), min(h, pred_y + pad)
        else:
            # Quét toàn vùng sân khi chưa khóa được bóng
            rx1, ry1, rx2, ry2 = 0, int(h * 0.08), w, int(h * 0.96)

        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Màu cam bóng rổ 5x5 + vàng đậm bóng FIBA 3x3
        mask_orange = cv2.inRange(hsv, np.array([4, 115, 90]), np.array([22, 255, 255]))
        mask_yellow = cv2.inRange(hsv, np.array([18, 130, 125]), np.array([34, 255, 255]))
        mask = cv2.bitwise_or(mask_orange, mask_yellow)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = -1.0
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 55 or area > 2600:
                continue
            peri = cv2.arcLength(c, True)
            if peri <= 0:
                continue
            circularity = 4.0 * np.pi * area / (peri * peri)
            if circularity < 0.52:
                continue
            (cx_loc, cy_loc), rad = cv2.minEnclosingCircle(c)
            if rad < 5 or rad > 28:
                continue
            cx, cy = cx_loc + rx1, cy_loc + ry1
            if not self._is_valid_ball_region(cx, cy, w, h, court_mgr):
                continue

            score = circularity
            if self.last_pos is not None:
                dist = float(np.hypot(cx - (self.last_pos[0] + self.vel[0]), cy - (self.last_pos[1] + self.vel[1])))
                score += max(0.0, 1.0 - dist / 160.0)
            if score > best_score:
                best_score = score
                best = (int(cx), int(cy), int(max(7, rad)), 0.35)
        return best

    def update(self, frame: np.ndarray, yolo_ball_candidates: List[tuple],
               court_mgr: Optional[CourtBoundaryManager] = None) -> dict:
        """Trả về {'status': 'LOCKED'|'TRACKING'|'SEARCHING', 'pos': (cx,cy,r)|None, 'conf': float}"""
        h, w = frame.shape[:2]
        chosen = None

        # 1. Chọn candidate tốt nhất từ YOLO (đã lọc vùng sân & tỷ lệ hình tròn)
        valid_yolo = []
        for x1, y1, x2, y2, conf in yolo_ball_candidates:
            bw, bh = x2 - x1, y2 - y1
            if bw < 6 or bh < 6 or bw > 95 or bh > 95:
                continue
            aspect = bw / max(1.0, float(bh))
            if aspect < 0.55 or aspect > 1.80:
                continue
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if not self._is_valid_ball_region(cx, cy, w, h, court_mgr):
                continue
            rad = max(7, (bw + bh) // 4)
            # Ưu tiên bóng gần vị trí dự đoán nhất + conf cao
            score = conf
            if self.last_pos is not None and self.missed_frames <= 10:
                dist = float(np.hypot(cx - (self.last_pos[0] + self.vel[0]), cy - (self.last_pos[1] + self.vel[1])))
                score += max(0.0, 0.5 - dist / 240.0)
            valid_yolo.append((score, cx, cy, rad, conf))

        if valid_yolo:
            valid_yolo.sort(key=lambda t: t[0], reverse=True)
            _, cx, cy, rad, conf = valid_yolo[0]
            chosen = (cx, cy, rad, conf)
        else:
            # 2. Dự phòng bắt bóng bằng màu bóng rổ khi YOLO bỏ sót
            chosen = self._color_fallback_detect(frame, court_mgr)

        if chosen is not None:
            cx, cy, rad, conf = chosen
            if self.last_pos is not None and self.missed_frames <= 4:
                vx = 0.65 * (cx - self.last_pos[0]) + 0.35 * self.vel[0]
                vy = 0.65 * (cy - self.last_pos[1]) + 0.35 * self.vel[1]
                self.vel = (float(np.clip(vx, -55, 55)), float(np.clip(vy, -55, 55)))
            else:
                self.vel = (0.0, 0.0)
            self.last_pos = (cx, cy, rad)
            self.missed_frames = 0
            self.trail.append((cx, cy))
            return {"status": "LOCKED", "pos": (cx, cy, rad), "conf": conf}

        # 3. Quán tính dự đoán ngắn khi bóng bị che khuất 1..6 frame
        self.missed_frames += 1
        if self.last_pos is not None and self.missed_frames <= self.max_coast_frames:
            nx = int(np.clip(self.last_pos[0] + self.vel[0], 0, w - 1))
            ny = int(np.clip(self.last_pos[1] + self.vel[1], 0, h - 1))
            self.vel = (self.vel[0] * 0.90, self.vel[1] * 0.90 + 0.7)  # ma sát + trọng lực nhẹ
            self.last_pos = (nx, ny, self.last_pos[2])
            self.trail.append((nx, ny))
            return {"status": "TRACKING", "pos": (nx, ny, self.last_pos[2]), "conf": 0.0}

        if len(self.trail) > 0:
            self.trail.popleft()
        return {"status": "SEARCHING", "pos": None, "conf": 0.0}


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
        self.ball_tracker = SmartBallTracker(trail_len=14)
        self.ball_trail = self.ball_tracker.trail  # tương thích ngược

        # Bộ gắn cố định mã VĐV theo thể thức (3x3: tối đa 6 mã, 5x5: tối đa 10 mã)
        self.player_reid = PersistentPlayerReID(max_slots=self.engine.max_on_court * 2)

        # Trình quản lý ranh giới vạch sân thi đấu (Court Boundary Tracker)
        self.court_mgr = CourtBoundaryManager(match_format=self.engine.match_format)
        self.use_court_roi = True

        print("[YOLO] Đang khởi tạo mô hình nhận diện Hitbox cầu thủ (yolov8n.pt)...")
        self.yolo_model = YOLO("yolov8n.pt")

    @property
    def court_roi_polygon(self):
        return self.court_mgr.polygon

    @court_roi_polygon.setter
    def court_roi_polygon(self, val):
        self.court_mgr.set_polygon(val)

    def scan_jerseys_pre_match(self) -> tuple[List[int], List[int]]:
        """Chụp 3 khung hình HD từ luồng OBS gửi lên Gemini để đọc số áo & quét vạch biên sân"""
        print("\n[AI SCAN] Đang chụp 3 khung hình HD từ OBS (trong 1.5 giây) để đọc số áo & quét vạch sân...")
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
        Hãy quan sát kỹ áo thi đấu của các cầu thủ trên sân và các đường vạch biên sân thi đấu:
        1. Đọc SỐ ÁO của từng đội:
           - Đội A ({t1['name']} - {c1}): mặc áo màu {t1['color']}.
           - Đội B ({t2['name']} - {c2}): mặc áo màu {t2['color']}.
           QUY TẮC LOẠI TRỪ: BỎ QUA trọng tài (áo xám/sọc đen trắng, quần đen, đeo còi), khán giả, HLV, quay phim và cầu thủ dự bị ngồi ngoài biên.
           Mỗi đội tối đa {max_p + 1} số áo (cho 3x3) hoặc 12 số áo (cho 5x5).
        2. QUÉT RANH GIỚI VẠCH BIÊN SÂN THI ĐẤU (court_polygon):
           Xác định 4 điểm góc [x, y] (tỷ lệ 0.0 đến 1.0) bám sát các vạch biên trắng/vàng trên mặt sân thi đấu thực tế:
           - Điểm 1 (Top-Left): Góc trên bên trái baseline (loại bỏ khu vực ghế ngồi ngoài biên của VĐV dự bị).
           - Điểm 2 (Top-Right): Góc trên bên phải baseline.
           - Điểm 3 (Bottom-Right): Góc dưới bên phải (loại bỏ khu vực cameraman ngồi bệt).
           - Điểm 4 (Bottom-Left): Góc dưới bên trái.
        """
        try:
            print(f"[AI SCAN] Đang phân tích số áo & quét vạch sân bằng {self.gemini.active_label}...")
            res = self.gemini.analyze_images_roster(jpeg_list, prompt)
            if res:
                if res.court_polygon and len(res.court_polygon) >= 4:
                    self.court_mgr.set_polygon(res.court_polygon)
                    print(f"[AI SCAN] Đã tự động căn chỉnh vạch sân OBS: {self.court_mgr.polygon}")
                return sorted(list(set(res.team_a_numbers))), sorted(list(set(res.team_b_numbers)))
        except Exception as e:
            print(f"[Lỗi AI Scan] Không thể quét số áo (cả 2 API đều lỗi): {e}")

        return [], []

    def _draw_hitboxes(self, frame: np.ndarray) -> tuple[np.ndarray, int]:
        # Lưu bản sạch chưa vẽ viền/chữ để Re-ID ngoại hình & nhận diện màu bóng không bị nhiễu
        clean_frame = frame.copy()

        # Track cả người (cls 0) và quả bóng rổ (cls 32 - sports ball).
        # Conf 0.15 cho phép bắt bóng nhỏ đang bay nhanh; người vẫn lọc conf >= 0.35.
        results = self.yolo_model.track(clean_frame, persist=True, classes=[0, 32], conf=0.15, verbose=False)
        h, w = frame.shape[:2]
        court_poly_pts = self.court_mgr.get_pixel_polygon(w, h)

        # Vẽ viền ranh giới sân N-điểm (vàng FIBA) & các vùng loại trừ (đỏ mờ)
        if self.use_court_roi:
            if len(court_poly_pts) >= 3:
                cv2.polylines(frame, [court_poly_pts], isClosed=True, color=(94, 211, 244), thickness=2, lineType=cv2.LINE_AA)
                for idx, (px, py) in enumerate(court_poly_pts):
                    cv2.circle(frame, (int(px), int(py)), 4, (94, 211, 244), -1)
                    cv2.putText(frame, f"P{idx + 1}", (int(px) + 5, int(py) - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (94, 211, 244), 1)
            for ex1, ey1, ex2, ey2 in self.court_mgr.get_pixel_exclusions(w, h):
                cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), (70, 57, 230), 2)
                cv2.putText(frame, "NO-TRACK", (ex1 + 4, ey1 + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 255), 1)

        in_court_dets = []
        outside_dets = []
        yolo_ball_candidates = []

        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else 0

                if cls_id == 0 and conf >= 0.35:
                    foot_x = (x1 + x2) // 2
                    foot_y = y2
                    is_in_court = True
                    if self.use_court_roi:
                        is_in_court = self.court_mgr.is_in_court(foot_x, foot_y, w, h)

                    if is_in_court:
                        in_court_dets.append({"bbox": (x1, y1, x2, y2), "track_id": track_id, "conf": conf})
                    else:
                        outside_dets.append(((x1, y1, x2, y2), track_id))

                elif cls_id == 32 and conf >= 0.15:
                    yolo_ball_candidates.append((x1, y1, x2, y2, conf))

        # Gán mã VĐV cố định (VDV-01..VDV-06) bằng Re-ID đa vùng (Áo + Quần + Giày)
        assigned_players = self.player_reid.assign_ids(clean_frame, in_court_dets)
        player_count = len(assigned_players)

        # Vẽ người ngoài sân (mờ, không tính vào số lượng VĐV)
        for (x1, y1, x2, y2), tid in outside_dets:
            dim_color = (90, 95, 105)
            label = f"Outside #{tid}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), dim_color, 1)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 4, y1), dim_color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 205, 215), 1)

        # Cập nhật SmartBallTracker (YOLO + lọc sân + fallback màu bóng + nội suy quỹ đạo)
        ball_info = self.ball_tracker.update(
            clean_frame, yolo_ball_candidates, self.court_mgr if self.use_court_roi else None
        )
        ball_status = ball_info["status"]
        ball_pos = ball_info["pos"]
        ball_holder = None

        # Xác định VĐV gần bóng nhất (đang cầm / kiểm soát bóng)
        if ball_pos is not None and assigned_players:
            bcx, bcy, _ = ball_pos
            min_d = 95.0
            for pl in assigned_players:
                px1, py1, px2, py2 = pl["bbox"]
                pcx, pcy = (px1 + px2) * 0.5, py1 + (py2 - py1) * 0.45
                d = float(np.hypot(bcx - pcx, bcy - pcy))
                if d < min_d:
                    min_d = d
                    ball_holder = pl["slot_id"]

        # Vẽ VĐV trong sân với mã cố định
        for pl in assigned_players:
            x1, y1, x2, y2 = pl["bbox"]
            color = pl["color"]
            is_holder = (ball_holder == pl["slot_id"])
            tag_suffix = " [BALL]" if is_holder else ""
            label = f"{pl['label']}{tag_suffix} ({pl['conf']:.2f})"
            thick = 3 if is_holder else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, max(12, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (8, 35, 62), 1)

        # Vẽ vệt di chuyển & tâm ngắm quả bóng rổ
        trail = self.ball_tracker.trail
        trail_len = len(trail)
        for i in range(1, trail_len):
            thick = max(1, int((i / trail_len) * 4))
            t_color = (0, 165, 255) if ball_status == "LOCKED" else (0, 215, 255)
            cv2.line(frame, trail[i - 1], trail[i], t_color, thick)

        if ball_pos is not None:
            bcx, bcy, br = ball_pos
            ring_color = (0, 140, 255) if ball_status == "LOCKED" else (0, 220, 255)
            cv2.circle(frame, (bcx, bcy), br + 4, ring_color, 2)
            cv2.circle(frame, (bcx, bcy), 3, (0, 255, 255), -1)
            b_txt = f"BALL {ball_info['conf']:.2f}" if ball_status == "LOCKED" else "BALL (EST)"
            cv2.putText(frame, b_txt, (bcx - 28, max(14, bcy - br - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, ring_color, 1)

        # Thanh trạng thái HUD chuẩn không bị lỗi dấu ? trong OpenCV
        import unicodedata
        clean_status = "".join(c for c in unicodedata.normalize('NFKD', self.gemini_status) if not unicodedata.combining(c)).replace('đ', 'd').replace('Đ', 'D')
        holder_txt = f" | Holder: {ball_holder}" if ball_holder else ""

        cv2.rectangle(frame, (10, 10), (740, 82), (62, 35, 8), -1)
        cv2.rectangle(frame, (10, 10), (740, 82), (94, 211, 244), 1)
        cv2.putText(frame, f"OBS LIVE | On-Court: {player_count}/{self.player_reid.max_slots} ({self.engine.match_format}) | Ball: {ball_status}{holder_txt}", (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (94, 211, 244), 2)
        cv2.putText(frame, f"AI ({self.gemini.active_label}): {clean_status[:58]}", (20, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (235, 240, 248), 1)
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
                prefix = "<span class='star-badge' title='Đội hình xuất phát'>★</span> " if p.is_starter else "<span class='bench-dot' title='Dự bị'>•</span> "
                cap = " <span class='cap-badge'>ĐỘI TRƯỞNG</span>" if p.is_captain else ""
                if p.min_formatted == "DNP":
                    rows += f"<tr class='dnp-row'><td class='col-no'>{prefix}<b>#{p.no}</b></td><td class='left col-name'>{p.name}{cap}</td><td colspan='19' class='left dnp-label'>Không thi đấu (DNP)</td></tr>"
                    continue

                tot_sec += p.seconds_played
                tot_fg_m += p.fg_m; tot_fg_a += p.fg_a
                tot_2p_m += p.fg2_m; tot_2p_a += p.fg2_a
                tot_3p_m += p.fg3_m; tot_3p_a += p.fg3_a
                tot_ft_m += p.ft_m; tot_ft_a += p.ft_a
                tot_or += p.or_reb; tot_dr += p.dr_reb; tot_reb += p.tot_reb
                tot_as += p.assists; tot_to += p.turnovers; tot_st += p.steals; tot_bs += p.blocks
                tot_pf += p.pf; tot_fd += p.fd; tot_pts += p.pts

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
            <div class="btc-card team-section">
                <div class="team-header-bar">
                    <div class="team-title-wrap">
                        <span class="team-badge-pill">{code}</span>
                        <span class="team-header-name">{team['name']}</span>
                    </div>
                    <div class="team-color-indicator">Màu áo thi đấu: <b>{team['color']}</b></div>
                </div>
                <div class="table-responsive">
                    <table class="fiba-table">
                        <thead>
                            <tr class="th-group">
                                <th rowspan="2" class="col-no">Số áo</th>
                                <th rowspan="2" class="left col-name">Vận động viên</th>
                                <th rowspan="2" class="col-min">Phút</th>
                                <th colspan="2">Ném rổ (FG)</th>
                                <th colspan="2">2 Điểm</th>
                                <th colspan="2">3 Điểm</th>
                                <th colspan="2">Ném phạt</th>
                                <th colspan="3">Bắt bóng (REB)</th>
                                <th rowspan="2" title="Kiến tạo">KT</th>
                                <th rowspan="2" title="Mất bóng">MB</th>
                                <th rowspan="2" title="Cướp bóng">CB</th>
                                <th rowspan="2" title="Chắn bóng">CH</th>
                                <th colspan="2">Phạm lỗi</th>
                                <th rowspan="2">+/-</th>
                                <th rowspan="2" class="col-pts">Điểm</th>
                            </tr>
                            <tr class="th-sub">
                                <th title="Trúng / Tổng lần ném">T/N</th><th>%</th>
                                <th title="Trúng / Tổng lần ném">T/N</th><th>%</th>
                                <th title="Trúng / Tổng lần ném">T/N</th><th>%</th>
                                <th title="Trúng / Tổng lần ném">T/N</th><th>%</th>
                                <th title="Bắt bóng tấn công">TC</th><th title="Bắt bóng phòng thủ">PT</th><th>Tổng</th>
                                <th title="Lỗi cá nhân">CN</th><th title="Bị phạm lỗi">BL</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows}
                            <tr class="totals-row">
                                <td colspan="2" class="left"><b>TỔNG CỘNG ĐỘI {code}</b></td>
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
            return f"<b>{adv['pts_in_paint_pts']}</b> điểm <span class='score-sub'>({adv['pts_in_paint_m']}/{adv['pts_in_paint_a']} · {pct}%)</span>"

        def fmt_time(sec):
            return f"{int(sec // 60):02d}:{int(sec % 60):02d}"

        return f"""
        <!-- BẢNG TỶ SỐ TRỰC TIẾP (HERO BANNER PHONG CÁCH BTCGIAIDAU.ID.VN) -->
        <section id="section-scoreboard" class="hero-score-card">
            <div class="hero-top-row">
                <span class="hero-tag-pill">BTC GIẢI ĐẤU · THỐNG KÊ BÓNG RỔ {engine.match_format.upper()} TRỰC TIẾP</span>
                <span class="hero-live-badge" role="status" aria-atomic="true">
                    <span class="live-dot"></span> ĐANG GHI NHẬN TỪ OBS AI
                </span>
            </div>
            <div class="scoreboard-grid">
                <div class="team-side team-side-a">
                    <div class="team-meta-box">
                        <span class="team-code-tag">{c1}</span>
                        <div class="team-full-name">{t1['name']}</div>
                        <div class="team-jersey-tag">Trang phục: {t1['color']}</div>
                    </div>
                    <div class="score-digit">{t1['score']}</div>
                </div>
                <div class="score-center-divider">
                    <div class="vs-box">VS</div>
                    <div class="lead-change-badge">Đổi dẫn: {engine.lead_changes} · Hòa: {engine.times_tied}</div>
                </div>
                <div class="team-side team-side-b">
                    <div class="score-digit">{t2['score']}</div>
                    <div class="team-meta-box meta-right">
                        <span class="team-code-tag">{c2}</span>
                        <div class="team-full-name">{t2['name']}</div>
                        <div class="team-jersey-tag">Trang phục: {t2['color']}</div>
                    </div>
                </div>
            </div>
        </section>

        <!-- BIÊN BẢN THỐNG KÊ CHI TIẾT 2 ĐỘI -->
        <section id="section-boxscore" class="boxscore-stack">
            {render_team(c1)}
            {render_team(c2)}
        </section>

        <!-- CHỈ SỐ CHUYÊN SÂU & DIỄN BIẾN TRẬN ĐẤU -->
        <section id="section-summary" class="summary-grid">
            <div class="btc-card summary-card">
                <div class="card-header-bar">
                    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" aria-hidden="true"><path d="M12 20V10M18 20V4M6 20v-4"/></svg>
                    <span>CHỈ SỐ CHUYÊN SÂU ĐỘI BÓNG</span>
                </div>
                <table class="summary-table">
                    <thead><tr><th class="left">Hạng mục thống kê</th><th>{c1}</th><th>{c2}</th></tr></thead>
                    <tbody>
                        <tr><td class="left">Điểm từ mất bóng đối phương</td><td><b>{a1['pts_from_to']}</b></td><td><b>{a2['pts_from_to']}</b></td></tr>
                        <tr><td class="left">Điểm trong khu vực hình thang</td><td>{fmt_paint(a1)}</td><td>{fmt_paint(a2)}</td></tr>
                        <tr><td class="left">Điểm cơ hội hai (sau OREB)</td><td><b>{a1['second_chance_pts']}</b></td><td><b>{a2['second_chance_pts']}</b></td></tr>
                        <tr><td class="left">Điểm phản công nhanh</td><td><b>{a1['fast_break_pts']}</b></td><td><b>{a2['fast_break_pts']}</b></td></tr>
                        <tr><td class="left">Điểm từ vận động viên dự bị</td><td><b>{a1['bench_pts']}</b></td><td><b>{a2['bench_pts']}</b></td></tr>
                    </tbody>
                </table>
            </div>

            <div class="btc-card summary-card">
                <div class="card-header-bar">
                    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" aria-hidden="true"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                    <span>DIỄN BIẾN & THẾ TRẬN</span>
                </div>
                <table class="summary-table">
                    <thead><tr><th class="left">Chỉ số diễn biến</th><th>{c1}</th><th>{c2}</th></tr></thead>
                    <tbody>
                        <tr><td class="left">Cách biệt điểm lớn nhất</td><td><b>{a1['biggest_lead']}</b> <span class="score-sub">({a1['biggest_lead_score']})</span></td><td><b>{a2['biggest_lead']}</b> <span class="score-sub">({a2['biggest_lead_score']})</span></td></tr>
                        <tr><td class="left">Chuỗi ghi điểm liên tiếp cao nhất</td><td><b>{a1['biggest_run']}</b> điểm <span class="score-sub">({a1['biggest_run_score']})</span></td><td><b>{a2['biggest_run']}</b> điểm <span class="score-sub">({a2['biggest_run_score']})</span></td></tr>
                        <tr><td class="left">Số lần đổi ngôi dẫn điểm</td><td colspan="2" class="center-highlight">{engine.lead_changes} lần</td></tr>
                        <tr><td class="left">Số lần tỷ số hòa nhau</td><td colspan="2" class="center-highlight">{engine.times_tied} lần</td></tr>
                        <tr><td class="left">Tổng thời gian dẫn trước</td><td><b>{fmt_time(a1['time_with_lead_sec'])}</b></td><td><b>{fmt_time(a2['time_with_lead_sec'])}</b></td></tr>
                    </tbody>
                </table>
            </div>
        </section>
        """

@app.route("/")
def index():
    html = """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>BTC Giải Đấu · Hệ Thống Thống Kê Bóng Rổ Trực Tiếp AI</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&family=Chakra+Petch:wght@500;600;700;800&family=Orbitron:wght@700;800;900&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary: #0d3b66;
                --primary-deep: #08233e;
                --primary-hover: #154e85;
                --primary-light: #e8f1f8;
                --primary-pixel: #0a2a4a;
                --accent: #faf0ca;
                --accent-yellow: #f4d35e;
                --accent-soft: #fdfbf2;
                --gold: #f4d35e;
                --coral: #ee964b;
                --cyan: #38bdf8;
                --ink: #08233e;
                --ink-secondary: #1e293b;
                --muted: #64748b;
                --line: #cbd5e1;
                --line-strong: #08233e;
                --bg: #f4f6fa;
                --card-bg: #ffffff;
                --success: #10b981;
                --success-light: #ecfdf5;
                --danger: #e63946;
                --danger-light: #fff5f5;
                --px-border: 2px solid #08233e;
                --px-border-thick: 3px solid #08233e;
                --px-shadow-sm: 2px 2px 0px #08233e;
                --px-shadow: 3px 3px 0px #08233e;
                --px-shadow-lg: 5px 5px 0px #08233e;
            }

            * { box-sizing: border-box; }
            html { scroll-behavior: smooth; }

            body {
                margin: 0;
                padding: 0;
                font-family: "Be Vietnam Pro", system-ui, -apple-system, sans-serif;
                color: var(--ink);
                background-color: var(--bg);
                background-image:
                    linear-gradient(to right, rgba(13, 59, 102, 0.055) 1px, transparent 1px),
                    linear-gradient(to bottom, rgba(13, 59, 102, 0.055) 1px, transparent 1px);
                background-size: 24px 24px;
                line-height: 1.5;
                -webkit-font-smoothing: antialiased;
            }

            /* LAYOUT SHELL: SIDEBAR + WORKSPACE (BTCGIAIDAU.ID.VN STYLE) */
            .app-shell {
                display: flex;
                min-height: 100vh;
            }

            /* SIDEBAR TRÁI */
            .sidebar {
                width: 236px;
                flex-shrink: 0;
                background: var(--card-bg);
                border-right: var(--px-border);
                display: flex;
                flex-direction: column;
                position: sticky;
                top: 0;
                height: 100vh;
                z-index: 20;
            }

            .brand-box {
                padding: 18px 18px 16px;
                border-bottom: 1px solid var(--line);
                display: flex;
                align-items: center;
                gap: 10px;
            }

            .brand-logo {
                width: 34px;
                height: 34px;
                background: var(--primary);
                border: var(--px-border);
                box-shadow: var(--px-shadow-sm);
                display: flex;
                align-items: center;
                justify-content: center;
                color: var(--gold);
                flex-shrink: 0;
            }

            .brand-title {
                font-family: "Chakra Petch", sans-serif;
                font-weight: 800;
                font-size: 15px;
                letter-spacing: 0.4px;
                color: var(--primary);
                line-height: 1.15;
            }

            .brand-sub {
                font-size: 10.5px;
                font-weight: 600;
                color: var(--muted);
                letter-spacing: 0.4px;
                text-transform: uppercase;
            }

            .nav-group {
                padding: 18px 14px;
                flex: 1;
            }

            .nav-section-label {
                font-family: "Chakra Petch", sans-serif;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 1px;
                color: var(--muted);
                text-transform: uppercase;
                padding: 0 8px 10px;
            }

            .nav-link {
                display: flex;
                align-items: center;
                gap: 10px;
                padding: 10px 10px;
                margin-bottom: 6px;
                color: var(--ink-secondary);
                text-decoration: none;
                font-size: 13px;
                font-weight: 600;
                border: 2px solid transparent;
                transition: background-color 0.12s ease, border-color 0.12s ease, transform 0.12s ease;
            }

            .nav-link:hover, .nav-link:focus-visible {
                background: var(--primary-light);
                border-color: var(--primary-deep);
                color: var(--primary-deep);
                box-shadow: var(--px-shadow-sm);
                outline: none;
            }

            .nav-link.active {
                background: var(--primary);
                color: #ffffff;
                border-color: var(--primary-deep);
                box-shadow: var(--px-shadow-sm);
            }

            .sidebar-footer {
                padding: 14px 16px;
                border-top: 1px solid var(--line);
                font-size: 12px;
                color: var(--muted);
                display: flex;
                align-items: center;
                gap: 8px;
            }

            .status-dot {
                width: 8px;
                height: 8px;
                background: var(--success);
                display: inline-block;
                flex-shrink: 0;
            }

            /* MAIN CONTENT AREA */
            .main-content {
                flex: 1;
                min-width: 0;
                padding: 22px 28px 36px;
                max-width: 1380px;
                margin: 0 auto;
                width: 100%;
            }

            /* TOP PAGE HEADER */
            .page-topbar {
                display: flex;
                justify-content: space-between;
                align-items: flex-end;
                gap: 16px;
                padding-bottom: 14px;
                margin-bottom: 22px;
                border-bottom: 2px solid var(--primary-deep);
                flex-wrap: wrap;
            }

            .topbar-kicker {
                font-family: "Chakra Petch", sans-serif;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 1.2px;
                text-transform: uppercase;
                color: var(--muted);
                margin-bottom: 2px;
            }

            .topbar-heading {
                font-family: "Chakra Petch", sans-serif;
                font-size: 24px;
                font-weight: 800;
                color: var(--primary);
                margin: 0;
                line-height: 1.2;
            }

            .topbar-actions {
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
            }

            /* PIXEL-BRUTALIST BUTTONS */
            .btn-btc {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 9px 16px;
                font-family: "Be Vietnam Pro", sans-serif;
                font-size: 13px;
                font-weight: 700;
                border: var(--px-border);
                border-radius: 0;
                box-shadow: var(--px-shadow);
                cursor: pointer;
                transition: transform 0.1s ease, box-shadow 0.1s ease, background-color 0.1s ease;
                min-height: 40px;
            }

            .btn-btc:hover {
                transform: translate(-1px, -1px);
                box-shadow: 4px 4px 0px var(--primary-deep);
            }

            .btn-btc:active {
                transform: translate(2px, 2px);
                box-shadow: 1px 1px 0px var(--primary-deep);
            }

            .btn-btc:focus-visible {
                outline: 3px solid var(--coral);
                outline-offset: 2px;
            }

            .btn-primary {
                background: var(--primary);
                color: #ffffff;
            }
            .btn-primary:hover { background: var(--primary-hover); }

            .btn-white {
                background: var(--card-bg);
                color: var(--primary-deep);
            }
            .btn-white:hover { background: var(--primary-light); }

            .btn-gold {
                background: var(--gold);
                color: var(--primary-deep);
            }
            .btn-gold:hover { background: #f8de7a; }

            .btn-danger {
                background: var(--danger);
                color: #ffffff;
            }
            .btn-danger:hover { background: #c92a37; }

            .btn-danger-outline {
                background: var(--danger-light);
                color: var(--danger);
                border-color: var(--danger);
                box-shadow: 2px 2px 0px var(--danger);
            }

            .btn-sm {
                padding: 6px 12px;
                font-size: 12px;
                min-height: 34px;
                box-shadow: var(--px-shadow-sm);
            }

            /* CARD CHUẨN BTCGIAIDAU */
            .btc-card {
                background: var(--card-bg);
                border: var(--px-border);
                box-shadow: var(--px-shadow-lg);
                margin-bottom: 22px;
            }

            /* CAMERA & TRACKING CARD */
            .cam-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 12px;
                padding: 12px 16px;
                background: var(--primary-light);
                border-bottom: var(--px-border);
                flex-wrap: wrap;
            }

            .cam-title {
                display: flex;
                align-items: center;
                gap: 9px;
                font-family: "Chakra Petch", sans-serif;
                font-size: 14px;
                font-weight: 800;
                color: var(--primary-deep);
                letter-spacing: 0.4px;
                text-transform: uppercase;
            }

            .cam-badges {
                display: flex;
                gap: 8px;
                flex-wrap: wrap;
            }

            .tag-pixel {
                font-family: "Chakra Petch", sans-serif;
                font-size: 11px;
                font-weight: 700;
                padding: 3px 9px;
                border: 1.5px solid var(--primary-deep);
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            .tag-navy { background: var(--primary); color: #ffffff; }
            .tag-gold { background: var(--accent); color: var(--primary-deep); }

            .obs-feed-container {
                background: #08233e;
                user-select: none;
                border-bottom: var(--px-border);
                display: flex;
                justify-content: center;
                align-items: center;
            }

            .feed-stage {
                position: relative;
                display: inline-block;
                max-width: 100%;
                line-height: 0;
            }

            .feed-stage img {
                max-width: 100%;
                max-height: 540px;
                width: auto;
                height: auto;
                display: block;
            }

            .cam-footer {
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 12px;
                padding: 12px 16px;
                background: var(--card-bg);
                flex-wrap: wrap;
            }

            .sync-indicator {
                display: flex;
                align-items: center;
                gap: 8px;
                font-size: 12.5px;
                font-weight: 600;
                color: var(--muted);
            }

            .calib-panel {
                display: none;
                background: var(--accent-soft);
                border-top: var(--px-border);
                padding: 14px 16px;
            }

            .calib-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
                gap: 8px;
                margin-bottom: 10px;
            }

            .calib-hint {
                font-size: 12.5px;
                color: var(--ink-secondary);
                background: #ffffff;
                border: 1.5px solid var(--line);
                padding: 8px 12px;
                margin-bottom: 12px;
            }

            /* HERO SCOREBOARD CARD (NAVY BANNER GIỐNG BTCGIAIDAU.ID.VN) */
            .hero-score-card {
                background: linear-gradient(135deg, #0d3b66 0%, #0a2a4a 100%);
                color: #ffffff;
                border: var(--px-border-thick);
                box-shadow: var(--px-shadow-lg);
                padding: 20px 24px 24px;
                margin-bottom: 22px;
            }

            .hero-top-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 12px;
                margin-bottom: 18px;
                flex-wrap: wrap;
            }

            .hero-tag-pill {
                font-family: "Chakra Petch", sans-serif;
                font-size: 11.5px;
                font-weight: 700;
                letter-spacing: 0.9px;
                color: var(--gold);
                border: 1.5px solid var(--gold);
                padding: 4px 12px;
                background: rgba(8, 35, 62, 0.55);
                text-transform: uppercase;
            }

            .hero-live-badge {
                display: inline-flex;
                align-items: center;
                gap: 7px;
                font-family: "Chakra Petch", sans-serif;
                font-size: 11.5px;
                font-weight: 700;
                color: #ecfdf5;
                letter-spacing: 0.6px;
            }

            .live-dot {
                width: 8px;
                height: 8px;
                background: #34d399;
                display: inline-block;
                animation: blinkDot 1.6s infinite;
            }

            @keyframes blinkDot {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.35; }
            }

            @media (prefers-reduced-motion: reduce) {
                .live-dot { animation: none; }
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
                justify-content: space-between;
                gap: 16px;
                background: rgba(8, 35, 62, 0.45);
                border: 2px solid rgba(250, 240, 202, 0.28);
                padding: 14px 18px;
            }

            .team-meta-box {
                display: flex;
                flex-direction: column;
                gap: 4px;
            }
            .team-meta-box.meta-right {
                align-items: flex-end;
                text-align: right;
            }

            .team-code-tag {
                display: inline-block;
                width: fit-content;
                font-family: "Chakra Petch", sans-serif;
                font-size: 12px;
                font-weight: 800;
                padding: 2px 8px;
                background: var(--gold);
                color: var(--primary-deep);
                letter-spacing: 0.8px;
            }

            .team-full-name {
                font-family: "Chakra Petch", sans-serif;
                font-size: 22px;
                font-weight: 800;
                color: #ffffff;
                line-height: 1.15;
                text-transform: uppercase;
            }

            .team-jersey-tag {
                font-size: 12px;
                color: #cbd5e1;
                font-weight: 500;
            }

            .score-digit {
                font-family: "Orbitron", "Chakra Petch", monospace;
                font-size: 52px;
                font-weight: 900;
                color: var(--gold);
                line-height: 1;
                font-variant-numeric: tabular-nums;
                text-shadow: 3px 3px 0px #08233e;
            }

            .score-center-divider {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 8px;
            }

            .vs-box {
                font-family: "Orbitron", sans-serif;
                font-size: 16px;
                font-weight: 900;
                color: var(--primary-deep);
                background: var(--accent);
                border: 2px solid var(--primary-deep);
                padding: 6px 14px;
                box-shadow: 3px 3px 0px #08233e;
            }

            .lead-change-badge {
                font-size: 11.5px;
                font-weight: 600;
                color: var(--accent);
                white-space: nowrap;
            }

            /* TEAM BOX SCORE TABLES */
            .team-header-bar {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 12px 16px;
                background: var(--primary);
                color: #ffffff;
                border-bottom: var(--px-border);
                flex-wrap: wrap;
                gap: 10px;
            }

            .team-title-wrap {
                display: flex;
                align-items: center;
                gap: 10px;
            }

            .team-badge-pill {
                font-family: "Chakra Petch", sans-serif;
                font-weight: 800;
                font-size: 13px;
                padding: 3px 9px;
                background: var(--gold);
                color: var(--primary-deep);
                border: 1.5px solid var(--primary-deep);
            }

            .team-header-name {
                font-family: "Chakra Petch", sans-serif;
                font-size: 17px;
                font-weight: 800;
                letter-spacing: 0.4px;
                text-transform: uppercase;
            }

            .team-color-indicator {
                font-size: 12.5px;
                color: var(--accent);
            }

            .table-responsive {
                width: 100%;
                overflow-x: auto;
            }

            .fiba-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 12px;
                font-variant-numeric: tabular-nums;
            }

            .fiba-table th, .fiba-table td {
                padding: 7px 4px;
                text-align: center;
                border-bottom: 1px solid var(--line);
                border-right: 1px solid #e2e8f0;
                white-space: nowrap;
            }

            .fiba-table thead tr.th-group th {
                background: var(--primary-pixel);
                color: #ffffff;
                font-family: "Chakra Petch", sans-serif;
                font-weight: 700;
                font-size: 11.5px;
                text-transform: uppercase;
                letter-spacing: 0.3px;
                border-color: rgba(255, 255, 255, 0.16);
            }

            .fiba-table thead tr.th-sub th {
                background: var(--primary-light);
                color: var(--primary-deep);
                font-weight: 700;
                font-size: 11px;
                border-bottom: var(--px-border);
            }

            .fiba-table tbody tr:nth-child(even) {
                background: #f8fafc;
            }

            .fiba-table tbody tr:hover {
                background: var(--accent-soft);
            }

            .left { text-align: left !important; padding-left: 12px !important; }
            .col-no { min-width: 68px; font-weight: 700; color: var(--primary); }
            .col-name { min-width: 155px; font-weight: 700; color: var(--ink); }
            .col-min { font-weight: 600; color: var(--ink-secondary); }
            .col-pct { color: var(--muted); font-size: 11.5px; }
            .col-totreb { background: #f1f5f9; font-weight: 700; }
            .col-pts {
                background: var(--accent-soft);
                color: var(--primary-deep);
                font-family: "Chakra Petch", sans-serif;
                font-size: 14px;
                font-weight: 800;
            }

            .star-badge { color: #d97706; font-size: 13px; }
            .bench-dot { color: var(--muted); }
            .cap-badge {
                display: inline-block;
                font-family: "Chakra Petch", sans-serif;
                font-size: 10px;
                font-weight: 800;
                padding: 1px 5px;
                background: var(--gold);
                color: var(--primary-deep);
                border: 1px solid var(--primary-deep);
                margin-left: 4px;
            }

            .dnp-row td { color: var(--muted); background: #f8fafc; }
            .dnp-label { font-style: italic; color: var(--muted); }

            .pm-pill {
                display: inline-block;
                padding: 2px 7px;
                font-weight: 700;
                font-size: 11.5px;
                border: 1px solid transparent;
            }
            .pm-pos { background: var(--success-light); color: #047857; border-color: #10b981; }
            .pm-neg { background: var(--danger-light); color: #b91c1c; border-color: #ef4444; }
            .pm-zero { background: #f1f5f9; color: var(--muted); border-color: var(--line); }

            .totals-row td {
                background: var(--accent) !important;
                border-top: var(--px-border) !important;
                border-bottom: none !important;
                font-weight: 700;
                color: var(--primary-deep);
            }

            /* SUMMARY CARDS (2 CỘT) */
            .summary-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
            }

            .card-header-bar {
                display: flex;
                align-items: center;
                gap: 9px;
                padding: 11px 16px;
                background: var(--primary);
                color: #ffffff;
                font-family: "Chakra Petch", sans-serif;
                font-size: 13.5px;
                font-weight: 800;
                letter-spacing: 0.4px;
                border-bottom: var(--px-border);
            }

            .summary-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 13px;
                font-variant-numeric: tabular-nums;
            }

            .summary-table th, .summary-table td {
                padding: 9px 12px;
                border-bottom: 1px solid var(--line);
                text-align: center;
            }

            .summary-table thead th {
                background: var(--primary-light);
                color: var(--primary-deep);
                font-family: "Chakra Petch", sans-serif;
                font-weight: 700;
                font-size: 12px;
                border-bottom: var(--px-border);
            }

            .summary-table tbody tr:last-child td { border-bottom: none; }
            .score-sub { color: var(--muted); font-size: 11.5px; font-weight: 500; }
            .center-highlight {
                font-family: "Chakra Petch", sans-serif;
                font-weight: 800;
                color: var(--primary);
                background: var(--accent-soft);
            }

            /* FOOTER */
            .page-footer {
                margin-top: 28px;
                padding-top: 14px;
                border-top: 2px solid var(--primary-deep);
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
                gap: 10px;
                font-size: 12px;
                color: var(--muted);
            }

            /* RESPONSIVE */
            @media (max-width: 1024px) {
                .app-shell { flex-direction: column; }
                .sidebar {
                    width: 100%;
                    height: auto;
                    position: static;
                    border-right: none;
                    border-bottom: var(--px-border);
                }
                .nav-group { display: flex; flex-wrap: wrap; gap: 6px; padding: 10px 14px; }
                .nav-section-label, .sidebar-footer { display: none; }
                .nav-link { margin-bottom: 0; padding: 7px 10px; }
                .main-content { padding: 16px; }
                .scoreboard-grid { grid-template-columns: 1fr; }
                .summary-grid { grid-template-columns: 1fr; }
            }

            /* IN BIÊN BẢN PDF CHUẨN A4 */
            @media print {
                body { background: #ffffff !important; color: #000000 !important; font-size: 11px; }
                .sidebar, .cam-card, .topbar-actions, .page-footer { display: none !important; }
                .main-content { padding: 0 !important; max-width: 100% !important; }
                .page-topbar { border-bottom: 2px solid #000 !important; margin-bottom: 12px !important; }
                .btc-card, .hero-score-card {
                    box-shadow: none !important;
                    border: 1.5px solid #000 !important;
                    background: #ffffff !important;
                    color: #000000 !important;
                    page-break-inside: avoid;
                }
                .team-side, .hero-tag-pill, .vs-box {
                    background: #ffffff !important;
                    color: #000000 !important;
                    border-color: #000 !important;
                    box-shadow: none !important;
                }
                .team-full-name, .score-digit, .lead-change-badge, .team-jersey-tag {
                    color: #000000 !important;
                    text-shadow: none !important;
                }
                .team-header-bar, .card-header-bar, .fiba-table thead tr.th-group th {
                    background: #e5e7eb !important;
                    color: #000000 !important;
                }
                .fiba-table th, .fiba-table td, .summary-table th, .summary-table td {
                    border: 1px solid #333 !important;
                    color: #000000 !important;
                    padding: 4px 5px !important;
                }
                .totals-row td { background: #f3f4f6 !important; }
            }
        </style>
    </head>
    <body>
        <div class="app-shell">
            <!-- SIDEBAR ĐIỀU HÀNH (PHONG CÁCH BTCGIAIDAU.ID.VN) -->
            <aside class="sidebar" aria-label="Menu điều hành">
                <div class="brand-box">
                    <div class="brand-logo" aria-hidden="true">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
                            <circle cx="12" cy="12" r="9"/>
                            <path d="M12 3v18M3 12h18M5.6 5.6c3.8 3.2 3.8 9.6 0 12.8M18.4 5.6c-3.8 3.2-3.8 9.6 0 12.8"/>
                        </svg>
                    </div>
                    <div>
                        <div class="brand-title">BTC◈GIẢI ĐẤU</div>
                        <div class="brand-sub">AI Bóng Rổ · OBS</div>
                    </div>
                </div>

                <nav class="nav-group">
                    <div class="nav-section-label">MENU ĐIỀU HÀNH</div>
                    <a href="#section-camera" class="nav-link active">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="1"/></svg>
                        <span>Camera & Vạch sân</span>
                    </a>
                    <a href="#section-scoreboard" class="nav-link">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><rect x="2" y="4" width="20" height="16"/><path d="M12 4v16"/></svg>
                        <span>Bảng tỷ số trực tiếp</span>
                    </a>
                    <a href="#section-boxscore" class="nav-link">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2"/><rect x="9" y="3" width="6" height="4"/></svg>
                        <span>Biên bản thống kê</span>
                    </a>
                    <a href="#section-summary" class="nav-link">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M18 20V10M12 20V4M6 20v-6"/></svg>
                        <span>Chỉ số chuyên sâu</span>
                    </a>
                </nav>

                <div class="sidebar-footer" role="status" aria-atomic="true">
                    <span class="status-dot"></span>
                    <span>Đồng bộ trực tiếp mỗi 2 giây</span>
                </div>
            </aside>

            <!-- VÙNG LÀM VIỆC CHÍNH -->
            <main class="main-content">
                <header class="page-topbar">
                    <div>
                        <div class="topbar-kicker">ĐIỀU HÀNH GIẢI ĐẤU · NỀN TẢNG SỐ HÓA BÓNG RỔ AI</div>
                        <h1 class="topbar-heading">Bảng Điểm & Biên Bản Điện Tử Trực Tiếp</h1>
                    </div>
                    <div class="topbar-actions">
                        <button class="btn-btc btn-white" onclick="toggleCourtCalib()" id="btnTopCalib">
                            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" aria-hidden="true"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>
                            <span>Căn chỉnh vạch sân</span>
                        </button>
                        <button class="btn-btc btn-primary" onclick="window.print()">
                            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" aria-hidden="true"><path d="M6 9V2h12v7M6 18H4a2 2 0 01-2-2v-5a2 2 0 012-2h16a2 2 0 012 2v5a2 2 0 01-2 2h-2"/><rect x="6" y="14" width="12" height="8"/></svg>
                            <span>In / Xuất PDF Biên Bản</span>
                        </button>
                    </div>
                </header>

                <!-- KHUNG LUỒNG CAMERA OBS & TRACKING -->
                <section id="section-camera" class="btc-card cam-card">
                    <div class="cam-header">
                        <div class="cam-title">
                            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" aria-hidden="true"><circle cx="12" cy="12" r="3"/><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z"/></svg>
                            <span>LUỒNG OBS STUDIO · NHẬN DIỆN VĐV & QUẢ BÓNG</span>
                        </div>
                        <div class="cam-badges">
                            <span class="tag-pixel tag-navy">YOLOv8 TRACKING</span>
                            <span class="tag-pixel tag-gold">COURT ROI N-ĐIỂM</span>
                        </div>
                    </div>

                    <div class="obs-feed-container" id="feedContainer">
                        <div class="feed-stage">
                            <img id="obsFeedImg" src="/video_feed" alt="Luồng video trực tiếp từ OBS Studio" />
                            <canvas id="courtCanvas" style="position: absolute; top:0; left:0; width:100%; height:100%; display: none; cursor: crosshair;"></canvas>
                        </div>
                    </div>

                    <div class="cam-footer">
                        <div class="sync-indicator">
                            <span class="status-dot"></span>
                            <span>Tự động phân tích pha bóng & cập nhật biên bản mỗi 2 giây</span>
                        </div>
                        <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                            <button class="btn-btc btn-white btn-sm" onclick="resetPlayerIDs()" title="Đặt lại toàn bộ mã VDV-01..06 khi sang trận mới">
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" aria-hidden="true"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>
                                <span>Đặt lại mã VĐV</span>
                            </button>
                            <button id="btnCalibToggle" class="btn-btc btn-white btn-sm" onclick="toggleCourtCalib()">
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" aria-hidden="true"><path d="M12 20h9M16.5 3.5a2.1 2.1 0 013 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>
                                <span>Căn chỉnh vạch sân</span>
                            </button>
                        </div>
                    </div>

                    <!-- BẢNG CĂN CHỈNH VẠCH SÂN & VÙNG LOẠI TRỪ -->
                    <div id="calibPanel" class="calib-panel">
                        <div class="calib-row">
                            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                                <button id="modePolyBtn" class="btn-btc btn-primary btn-sm" onclick="setCalibMode('polygon')">
                                    Ranh giới sân (N-điểm)
                                </button>
                                <button id="modeExclBtn" class="btn-btc btn-danger-outline btn-sm" onclick="setCalibMode('exclusion')">
                                    Vẽ vùng loại trừ (Ghế dự bị / Quay phim)
                                </button>
                            </div>
                            <span id="calibStatus" style="font-size: 12.5px; font-weight: 700; color: #047857;"></span>
                        </div>

                        <div id="calibHint" class="calib-hint">
                            • Kéo chốt góc để bám sát vạch biên (tự động hút dính mép khung hình khi vạch sân tràn) • <b>Nhấp đúp (Double-click)</b> lên cạnh để thêm góc mới • <b>Chuột phải</b> vào chốt để xóa góc.
                        </div>

                        <div style="display: flex; gap: 8px; flex-wrap: wrap; align-items: center;">
                            <button class="btn-btc btn-gold btn-sm" onclick="saveCourtConfig()">Lưu cấu hình sân</button>
                            <button class="btn-btc btn-white btn-sm" onclick="setCourtPreset('3x3')">Mẫu 3x3 (5 góc ôm đáy)</button>
                            <button class="btn-btc btn-white btn-sm" onclick="setCourtPreset('full')">Toàn màn hình (4 góc)</button>
                            <button class="btn-btc btn-danger-outline btn-sm" onclick="clearExclusionZones()">Xóa hết vùng đỏ</button>
                            <button class="btn-btc btn-white btn-sm" onclick="toggleCourtCalib()" style="margin-left: auto;">Đóng</button>
                        </div>
                    </div>
                </section>

                <!-- BẢNG TỶ SỐ & BIÊN BẢN THỐNG KÊ TỰ ĐỘNG CẬP NHẬT -->
                <div id="boxscore-container">
                    <div class="btc-card" style="padding: 28px; text-align: center; font-weight: 600; color: var(--muted);">
                        Đang đồng bộ dữ liệu bảng điểm trực tiếp từ hệ thống OBS AI...
                    </div>
                </div>

                <footer class="page-footer">
                    <span><b>BTC◈GIẢI ĐẤU</b> · Hệ thống số hóa & thống kê bóng rổ tự động chuẩn FIBA</span>
                    <span>Phong cách giao diện đồng bộ theo btcgiaidau.id.vn</span>
                </footer>
            </main>
        </div>

        <script>
            function refreshBoxScore() {
                fetch('/boxscore_data').then(r => r.text()).then(html => {
                    if (html) document.getElementById('boxscore-container').innerHTML = html;
                }).catch(err => console.error("Lỗi cập nhật bảng điểm:", err));
            }
            setInterval(refreshBoxScore, 2000);
            refreshBoxScore();

            // Active state cho menu sidebar
            document.querySelectorAll('.nav-link').forEach(link => {
                link.addEventListener('click', () => {
                    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
                    link.classList.add('active');
                });
            });

            // QUẢN LÝ CĂN CHỈNH VẠCH SÂN (N-ĐIỂM + EDGE SNAP) & VÙNG LOẠI TRỪ (EXCLUSION ZONES)
            let calibActive = false;
            let calibMode = 'polygon'; // 'polygon' | 'exclusion'
            let courtPoints = [
                [0.00, 0.42],
                [0.62, 0.24],
                [1.00, 0.36],
                [1.00, 1.00],
                [0.00, 1.00]
            ];
            let exclusionZones = []; // [[x1, y1, x2, y2], ...]
            let activeDragIdx = -1;
            let drawingExcl = null;

            function snapToEdge(val) {
                const snapThreshold = 0.025;
                if (val < snapThreshold) return 0.0;
                if (val > 1.0 - snapThreshold) return 1.0;
                return val;
            }

            function setCalibMode(mode) {
                calibMode = mode;
                const pBtn = document.getElementById('modePolyBtn');
                const eBtn = document.getElementById('modeExclBtn');
                const hint = document.getElementById('calibHint');
                if (mode === 'polygon') {
                    pBtn.className = 'btn-btc btn-primary btn-sm';
                    eBtn.className = 'btn-btc btn-danger-outline btn-sm';
                    hint.innerHTML = '• Kéo chốt góc để bám sát vạch biên (tự động hút dính mép khung hình khi vạch sân tràn) • <b>Nhấp đúp (Double-click)</b> lên cạnh để thêm góc mới • <b>Chuột phải</b> vào chốt để xóa góc.';
                } else {
                    eBtn.className = 'btn-btc btn-danger btn-sm';
                    pBtn.className = 'btn-btc btn-white btn-sm';
                    hint.innerHTML = '• <b>Nhấn giữ và kéo chuột</b> trên video để khoanh vùng đỏ loại trừ (ghế dự bị, quay phim, khán giả) • <b>Chuột phải</b> vào ô đỏ để xóa.';
                }
                drawCalibCanvas();
            }

            function fetchCourtConfig() {
                fetch('/api/get_court_polygon').then(r => r.json()).then(data => {
                    if (data.polygon && data.polygon.length >= 3) {
                        courtPoints = data.polygon;
                    }
                    if (Array.isArray(data.exclusion_zones)) {
                        exclusionZones = data.exclusion_zones;
                    }
                    drawCalibCanvas();
                }).catch(e => console.error(e));
            }

            function toggleCourtCalib() {
                calibActive = !calibActive;
                const canvas = document.getElementById('courtCanvas');
                const panel = document.getElementById('calibPanel');
                const btn = document.getElementById('btnCalibToggle');
                if (calibActive) {
                    canvas.style.display = 'block';
                    panel.style.display = 'block';
                    btn.className = 'btn-btc btn-gold btn-sm';
                    fetchCourtConfig();
                    initCanvasEvents();
                    document.getElementById('section-camera').scrollIntoView({ behavior: 'smooth', block: 'start' });
                } else {
                    canvas.style.display = 'none';
                    panel.style.display = 'none';
                    btn.className = 'btn-btc btn-white btn-sm';
                }
            }

            function drawCalibCanvas() {
                const canvas = document.getElementById('courtCanvas');
                if (!canvas || !calibActive) return;
                const rect = canvas.getBoundingClientRect();
                canvas.width = rect.width;
                canvas.height = rect.height;
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                if (courtPoints.length >= 3) {
                    ctx.beginPath();
                    courtPoints.forEach((p, i) => {
                        const x = p[0] * canvas.width;
                        const y = p[1] * canvas.height;
                        if (i === 0) ctx.moveTo(x, y);
                        else ctx.lineTo(x, y);
                    });
                    ctx.closePath();
                    ctx.fillStyle = 'rgba(244, 211, 94, 0.16)';
                    ctx.fill();
                    ctx.strokeStyle = '#f4d35e';
                    ctx.lineWidth = 2.5;
                    ctx.stroke();

                    courtPoints.forEach((p, i) => {
                        const x = p[0] * canvas.width;
                        const y = p[1] * canvas.height;
                        ctx.beginPath();
                        ctx.arc(x, y, calibMode === 'polygon' ? 8 : 5, 0, Math.PI * 2);
                        ctx.fillStyle = (p[0] === 0 || p[0] === 1 || p[1] === 0 || p[1] === 1) ? '#10b981' : '#0d3b66';
                        ctx.fill();
                        ctx.strokeStyle = '#f4d35e';
                        ctx.lineWidth = 2;
                        ctx.stroke();

                        ctx.fillStyle = '#ffffff';
                        ctx.font = 'bold 11px "Chakra Petch", sans-serif';
                        const lx = x > canvas.width - 35 ? x - 26 : x + 10;
                        const ly = y < 18 ? y + 16 : y - 6;
                        ctx.fillText('P' + (i + 1), lx, ly);
                    });
                }

                const allExcl = [...exclusionZones];
                if (drawingExcl) {
                    allExcl.push([
                        Math.min(drawingExcl.x1, drawingExcl.x2),
                        Math.min(drawingExcl.y1, drawingExcl.y2),
                        Math.max(drawingExcl.x1, drawingExcl.x2),
                        Math.max(drawingExcl.y1, drawingExcl.y2)
                    ]);
                }
                allExcl.forEach((z, idx) => {
                    const x = z[0] * canvas.width;
                    const y = z[1] * canvas.height;
                    const w = (z[2] - z[0]) * canvas.width;
                    const h = (z[3] - z[1]) * canvas.height;
                    ctx.fillStyle = 'rgba(230, 57, 70, 0.30)';
                    ctx.fillRect(x, y, w, h);
                    ctx.strokeStyle = '#e63946';
                    ctx.lineWidth = 2;
                    ctx.strokeRect(x, y, w, h);
                    ctx.fillStyle = '#ffffff';
                    ctx.font = 'bold 11px "Chakra Petch", sans-serif';
                    ctx.fillText('LOẠI TRỪ #' + (idx + 1), x + 5, y + 14);
                });
            }

            function insertPointOnNearestEdge(mx, my) {
                let bestIdx = courtPoints.length;
                let minDist = Infinity;
                const n = courtPoints.length;
                for (let i = 0; i < n; i++) {
                    const a = courtPoints[i];
                    const b = courtPoints[(i + 1) % n];
                    const dx = b[0] - a[0], dy = b[1] - a[1];
                    const lenSq = dx * dx + dy * dy;
                    let t = lenSq > 0 ? ((mx - a[0]) * dx + (my - a[1]) * dy) / lenSq : 0;
                    t = Math.max(0, Math.min(1, t));
                    const px = a[0] + t * dx, py = a[1] + t * dy;
                    const d = Math.hypot(mx - px, my - py);
                    if (d < minDist) {
                        minDist = d;
                        bestIdx = i + 1;
                    }
                }
                courtPoints.splice(bestIdx, 0, [snapToEdge(mx), snapToEdge(my)]);
                drawCalibCanvas();
                saveCourtConfig();
            }

            function initCanvasEvents() {
                const canvas = document.getElementById('courtCanvas');
                if (!canvas || canvas.dataset.hasEvents) return;
                canvas.dataset.hasEvents = 'true';

                canvas.addEventListener('contextmenu', e => {
                    e.preventDefault();
                    const rect = canvas.getBoundingClientRect();
                    const mx = (e.clientX - rect.left) / rect.width;
                    const my = (e.clientY - rect.top) / rect.height;

                    if (calibMode === 'polygon') {
                        if (courtPoints.length <= 3) return;
                        let closestIdx = -1, minDist = 0.05;
                        courtPoints.forEach((p, i) => {
                            const d = Math.hypot(p[0] - mx, p[1] - my);
                            if (d < minDist) { minDist = d; closestIdx = i; }
                        });
                        if (closestIdx !== -1) {
                            courtPoints.splice(closestIdx, 1);
                            drawCalibCanvas();
                            saveCourtConfig();
                        }
                    } else {
                        for (let i = exclusionZones.length - 1; i >= 0; i--) {
                            const z = exclusionZones[i];
                            if (mx >= z[0] && mx <= z[2] && my >= z[1] && my <= z[3]) {
                                exclusionZones.splice(i, 1);
                                drawCalibCanvas();
                                saveCourtConfig();
                                break;
                            }
                        }
                    }
                });

                canvas.addEventListener('dblclick', e => {
                    if (calibMode !== 'polygon') return;
                    const rect = canvas.getBoundingClientRect();
                    const mx = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
                    const my = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
                    insertPointOnNearestEdge(mx, my);
                });

                canvas.addEventListener('mousedown', e => {
                    if (e.button !== 0) return;
                    const rect = canvas.getBoundingClientRect();
                    const mx = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
                    const my = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));

                    if (calibMode === 'polygon') {
                        let closestIdx = -1, minDist = 0.06;
                        courtPoints.forEach((p, i) => {
                            const d = Math.hypot(p[0] - mx, p[1] - my);
                            if (d < minDist) { minDist = d; closestIdx = i; }
                        });
                        activeDragIdx = closestIdx;
                    } else {
                        drawingExcl = {x1: mx, y1: my, x2: mx, y2: my};
                    }
                });

                window.addEventListener('mousemove', e => {
                    if (!calibActive) return;
                    const canvas = document.getElementById('courtCanvas');
                    const rect = canvas.getBoundingClientRect();
                    const mx = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
                    const my = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));

                    if (calibMode === 'polygon' && activeDragIdx !== -1) {
                        courtPoints[activeDragIdx] = [snapToEdge(mx), snapToEdge(my)];
                        drawCalibCanvas();
                    } else if (calibMode === 'exclusion' && drawingExcl) {
                        drawingExcl.x2 = mx;
                        drawingExcl.y2 = my;
                        drawCalibCanvas();
                    }
                });

                window.addEventListener('mouseup', () => {
                    if (activeDragIdx !== -1) {
                        activeDragIdx = -1;
                        saveCourtConfig();
                    } else if (drawingExcl) {
                        const x1 = Math.min(drawingExcl.x1, drawingExcl.x2);
                        const y1 = Math.min(drawingExcl.y1, drawingExcl.y2);
                        const x2 = Math.max(drawingExcl.x1, drawingExcl.x2);
                        const y2 = Math.max(drawingExcl.y1, drawingExcl.y2);
                        drawingExcl = null;
                        if ((x2 - x1) > 0.02 && (y2 - y1) > 0.02) {
                            exclusionZones.push([x1, y1, x2, y2]);
                            saveCourtConfig();
                        }
                        drawCalibCanvas();
                    }
                });

                window.addEventListener('resize', () => {
                    if (calibActive) drawCalibCanvas();
                });
            }

            function saveCourtConfig() {
                const status = document.getElementById('calibStatus');
                status.innerText = 'Đang lưu...';
                fetch('/api/set_court_polygon', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({polygon: courtPoints, exclusion_zones: exclusionZones})
                }).then(r => r.json()).then(data => {
                    status.innerText = 'Đã lưu (' + courtPoints.length + ' góc sân, ' + exclusionZones.length + ' vùng loại trừ)';
                    setTimeout(() => { if (status) status.innerText = ''; }, 3000);
                }).catch(e => {
                    status.innerText = 'Lỗi lưu: ' + e;
                });
            }

            function clearExclusionZones() {
                exclusionZones = [];
                drawCalibCanvas();
                saveCourtConfig();
            }

            function setCourtPreset(preset) {
                if (preset === '3x3') {
                    courtPoints = [[0.00, 0.42], [0.62, 0.24], [1.00, 0.36], [1.00, 1.00], [0.00, 1.00]];
                } else {
                    courtPoints = [[0.00, 0.00], [1.00, 0.00], [1.00, 1.00], [0.00, 1.00]];
                }
                drawCalibCanvas();
                saveCourtConfig();
            }

            function resetPlayerIDs() {
                fetch('/api/reset_player_ids', { method: 'POST' })
                    .then(r => r.json())
                    .then(() => {
                        const status = document.getElementById('calibStatus');
                        if (status) {
                            status.innerText = 'Đã đặt lại toàn bộ mã VĐV cố định!';
                            setTimeout(() => { status.innerText = ''; }, 3000);
                        }
                    }).catch(e => console.error(e));
            }
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

@app.route("/api/get_court_polygon")
def get_court_polygon():
    if vision_system and vision_system.court_mgr:
        return jsonify({
            "polygon": vision_system.court_mgr.polygon,
            "exclusion_zones": vision_system.court_mgr.exclusion_zones
        })
    return jsonify({"polygon": [], "exclusion_zones": []})

@app.route("/api/set_court_polygon", methods=["POST"])
def set_court_polygon():
    data = request.get_json(silent=True) or {}
    poly = data.get("polygon")
    excl = data.get("exclusion_zones")
    if vision_system and vision_system.court_mgr and (poly is not None or excl is not None):
        if poly is not None and len(poly) < 3:
            return jsonify({"status": "error", "message": "Polygon needs at least 3 points"}), 400
        vision_system.court_mgr.set_config(pts=poly, exclusion_zones=excl)
        return jsonify({
            "status": "ok",
            "polygon": vision_system.court_mgr.polygon,
            "exclusion_zones": vision_system.court_mgr.exclusion_zones
        })
    return jsonify({"status": "error", "message": "Invalid config"}), 400

@app.route("/api/reset_player_ids", methods=["POST"])
def reset_player_ids():
    if vision_system and hasattr(vision_system, "player_reid"):
        vision_system.player_reid.reset()
        return jsonify({"status": "ok"})
    return jsonify({"status": "error"}), 400


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
