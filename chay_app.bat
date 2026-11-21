@echo off
title AI Bong Ro FIBA - OBS Live Tracking
chcp 65001 >nul
echo [Auto-Update] Đang kiem tra cap nhat tu GitHub...
git pull origin main 2>nul
python AIbongro.py
if errorlevel 1 (
    echo.
    echo [Thong bao] Nhan phim bat ky de dong cua so...
    pause >nul
)
