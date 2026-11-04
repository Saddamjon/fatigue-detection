"""
Система определения усталости человека по видеоизображению.

Запуск:
    python main.py                      # веб-камера
    python main.py --source video.mp4   # видеофайл
    python main.py --source video.mp4 --no-display   # быстрый анализ без окна

Клавиши в окне:  Q / Esc — выход,  C — повторная калибровка,  S — снимок экрана,
                 U — режим «плохо себя чувствую» (вкл/выкл)
(работают и в русской раскладке; окно можно закрыть крестиком)
"""
import argparse
import concurrent.futures
import csv
import json
import math
import os
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config
from decision import FatigueDecision
from features import (FaceAnalyzer, ModelUnavailable, LEFT_EYE, RIGHT_EYE,
                      MOUTH_VERTICAL, MOUTH_HORIZONTAL)

LEVEL_COLORS = {0: (60, 180, 75), 1: (0, 165, 255), 2: (40, 40, 230)}   # BGR
WINDOW = "Fatigue detection"

# Клавиши с учётом русской раскладки (та же физическая клавиша)
KEYS_QUIT = {27, ord("q"), ord("Q"), ord("й"), ord("Й")}
KEYS_CALIB = {ord("c"), ord("C"), ord("с"), ord("С")}
KEYS_SHOT = {ord("s"), ord("S"), ord("ы"), ord("Ы")}
KEYS_UNWELL = {ord("u"), ord("U"), ord("г"), ord("Г")}

EYE_STATE_NAMES = {"open": "открыты", "closed": "закрыты", "unknown": "не видны"}


class EventLog:
    """Короткие сообщения о событиях на экране (кивок, смена позы, артефакт)."""

    def __init__(self, keep_sec=2.5):
        self.keep = keep_sec
        self.items = []

    def add(self, text, color):
        self.items.append((time.time(), text, color))

    def visible(self):
        now = time.time()
        self.items = [x for x in self.items if now - x[0] <= self.keep]
        return self.items[-3:]

FONT_CANDIDATES = [
    "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial Unicode.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
]


class TextRenderer:
    """Вывод кириллицы на кадр (cv2.putText её не поддерживает)."""

    def __init__(self):
        self.path = next((p for p in FONT_CANDIDATES if os.path.exists(p)), None)
        self.cache = {}
        self.items = []

    def font(self, size):
        if size not in self.cache:
            assert self.path is not None    # вызывается, только если шрифт найден
            self.cache[size] = ImageFont.truetype(self.path, size)
        return self.cache[size]

    def add(self, text, xy, color=(255, 255, 255), size=18):
        self.items.append((text, xy, color, size))

    def render(self, frame):
        if not self.items:
            return frame
        if self.path is None:   # шрифт не найден — латиница средствами OpenCV
            for text, (x, y), color, size in self.items:
                cv2.putText(frame, text.encode("ascii", "replace").decode(), (x, y + size),
                            cv2.FONT_HERSHEY_SIMPLEX, size / 32, color, 1, cv2.LINE_AA)
        else:
            img = Image.fromarray(frame[:, :, ::-1])
            draw = ImageDraw.Draw(img)
            for text, xy, color, size in self.items:
                draw.text(xy, text, font=self.font(size), fill=tuple(reversed(color)))
            frame = np.array(img)[:, :, ::-1].copy()
        self.items = []
        return frame


class Beeper:
    """Звуковой сигнал тревоги, не блокирующий обработку видео."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.busy = False

    def beep(self):
        if not self.enabled or self.busy:
            return
        self.busy = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            if sys.platform == "win32":
                import winsound
                winsound.Beep(1500, 400)
            else:
                print("\a", end="", flush=True)
                time.sleep(0.4)
        finally:
            self.busy = False


def draw_overlay(frame, f, st, text, fps, events=None, decision=None):
    h, w = frame.shape[:2]

    # Контуры глаз и рта
    if f.face_found and f.points is not None:
        eye_color = (0, 0, 255) if st.eyes_closed else (0, 255, 0)
        for idx in (LEFT_EYE, RIGHT_EYE):
            pts = f.points[idx].astype(np.int32)
            cv2.polylines(frame, [pts], True, eye_color, 1, cv2.LINE_AA)
        mouth = [p for pair in MOUTH_VERTICAL for p in pair] + list(MOUTH_HORIZONTAL)
        for i in mouth:
            x, y = f.points[i].astype(int)
            cv2.circle(frame, (x, y), 2, (255, 200, 0), -1, cv2.LINE_AA)
    # Рука у лица: точки руки (оранжевые — если закрывает глаза или рот)
    if f.hand_points:
        hcol = (0, 140, 255) if (f.hand_on_eyes or f.hand_on_mouth) else (160, 160, 160)
        for hp in f.hand_points:
            for x, y in hp.astype(int):
                cv2.circle(frame, (int(x), int(y)), 2, hcol, -1, cv2.LINE_AA)

    # Полупрозрачная панель с показателями
    panel_w, panel_h = 330, 300
    ev_items = events.visible() if events is not None else []
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (20, 20, 20), -1)
    if ev_items:
        cv2.rectangle(overlay, (w - 345, 0), (w, 14 + 26 * len(ev_items)), (20, 20, 20), -1)
    cv2.rectangle(overlay, (0, h - 70), (w, h), (20, 20, 20), -1)
    frame[:] = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    y = 8
    def line(s, line_color=(230, 230, 230), size=17):
        nonlocal y
        text.add(s, (10, y), line_color, size)
        y += size + 6

    if not f.face_found:
        line("Лицо не найдено", (80, 80, 255))
    else:
        line(f"EAR: {f.ear:.3f}   нейросеть: {f.blink:.2f}   MAR: {f.mar:.2f}")
        if st.phase == "monitoring":
            if st.eye_state == "unknown":
                why = "рука на глазах" if f.hand_on_eyes else "голова повёрнута"
                line(f"Глаза: не видны ({why})", (0, 165, 255))
            else:
                notes = []
                if st.squinting:
                    notes.append("прищур")
                if st.head_turned:
                    notes.append("поворот")
                extra = f" ({', '.join(notes)})" if notes else ""
                line(f"Открытость глаз: {st.openness * 100:.0f}%{extra}",
                     (80, 80, 255) if st.eyes_closed else (230, 230, 230))
        line(f"Голова: наклон {f.pitch:+.0f}°, поворот {f.yaw:+.0f}°")
        if st.phase == "monitoring" and decision is not None and decision.adapt:
            line(f"Норма позы: наклон {st.pose_pitch:+.0f}°, поворот {st.pose_yaw:+.0f}°",
                 (180, 180, 180), 15)
    if st.phase == "monitoring":
        line(f"PERCLOS: {st.perclos * 100:.1f}%")
        line(f"Моргание: {st.blink_dur:.2f} с,  {st.blink_rate:.0f}/мин")
        line(f"Зевки (5 мин): {st.yawns}   Кивки: {st.nods}")
        line(f"Индекс усталости: {st.index:.0f}")
        if decision is not None and decision.context:
            line(f"За рулём: {st.hours:.1f} ч   риск по времени суток: {st.circadian:.1f}",
                 (180, 180, 180), 15)
        line(f"Пороговый метод: {'ТРЕВОГА' if st.baseline_alarm else 'норма'}",
             (80, 80, 255) if st.baseline_alarm else (180, 180, 180), 15)
    if st.unwell:
        line("Режим: плохое самочувствие (U — выключить)", (0, 165, 255), 15)
    line(f"{fps:.0f} кадр/с", (150, 150, 150), 14)

    # События справа вверху
    if ev_items:
        ey = 10
        for _, msg, ev_color in ev_items:
            text.add(msg, (w - 330, ey), ev_color, 18)
            ey += 26

    # Нижняя строка: состояние
    if st.phase == "calibration":
        color = (255, 200, 0)
        left = int(math.ceil(st.calib_seconds_left))
        mode = decision.calib_mode if decision is not None else "guided"
        if st.calib_step == "open" and mode == "auto":
            msg = (f"Автокалибровка {st.calibration_progress * 100:.0f}% — "
                   f"система запоминает вашу обычную норму")
        elif st.calib_step == "open":
            msg = (f"Калибровка {st.calibration_progress * 100:.0f}% — сядьте так, как обычно "
                   f"сидите за рулём, моргайте как обычно")
        elif st.calib_step == "search":
            msg = f"Калибровка по записи: ищу момент, когда глаза закрыты ({left} с)"
        elif st.calib_step == "prepare":
            msg = f"Сейчас прозвучит сигнал — закройте глаза до второго сигнала ({left})"
        else:
            msg = f"Глаза закрыты... ещё {left} с, откройте после сигнала"
        bar = int(w * st.calibration_progress)
        cv2.rectangle(frame, (0, h - 6), (bar, h), color, -1)
        text.add(msg, (12, h - 52), color, 20)
    else:
        color = LEVEL_COLORS[st.level]
        status = "ТРЕВОГА! " if st.alarm else ""
        text.add(status + config.LEVEL_NAMES[st.level], (12, h - 62), color, 26)
        if st.reason:
            text.add(st.reason, (12, h - 28), (220, 220, 220), 16)
        advice = st.advice or (decision.baseline.state_warning if decision is not None else "")
        if advice:
            text.add(advice, (12, h - 96), (0, 200, 255), 18)
        # Шкала индекса (отметки — границы уровней с учётом режима самочувствия)
        bx, bw = w - 220, 200
        cv2.rectangle(frame, (bx, h - 45), (bx + bw, h - 25), (80, 80, 80), 1)
        cv2.rectangle(frame, (bx, h - 45), (bx + int(bw * st.index / 100), h - 25), color, -1)
        thresholds = (decision.level_thresholds if decision is not None
                      else (config.LEVEL_TIRED, config.LEVEL_DANGER))
        for thr in thresholds:
            x = bx + int(bw * thr / 100)
            cv2.line(frame, (x, h - 49), (x, h - 21), (200, 200, 200), 1)
        if st.alarm and int(time.time() * 4) % 2 == 0:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 12)
    return frame


CSV_FIELDS = ["t", "phase", "face", "ear", "ear_left", "ear_right", "mar", "pitch", "yaw", "roll",
              "blink", "squint", "openness", "ear_thr", "eye_state", "eye_mode", "squinting",
              "hand_eyes", "hand_mouth", "yawning", "pose_pitch", "pose_yaw", "nod", "pose_change",
              "artifact", "eyes_closed", "closure_now", "perclos", "blink_dur", "blink_rate",
              "yawns", "nods", "hours", "circadian", "unwell",
              "raw_index", "index", "level", "alarm", "baseline_alarm"]


def csv_row(f, st):
    return {
        "t": f"{st.t:.3f}", "phase": st.phase, "face": int(f.face_found),
        "ear": f"{f.ear:.4f}", "ear_left": f"{f.ear_left:.4f}", "ear_right": f"{f.ear_right:.4f}",
        "mar": f"{f.mar:.4f}", "pitch": f"{f.pitch:.2f}", "yaw": f"{f.yaw:.2f}", "roll": f"{f.roll:.2f}",
        "blink": f"{f.blink:.4f}", "squint": f"{f.squint:.4f}", "openness": f"{st.openness:.3f}",
        "ear_thr": f"{st.ear_threshold:.4f}" if st.phase == "monitoring" else "",
        "eye_state": st.eye_state if st.phase == "monitoring" else "",
        "eye_mode": st.eye_mode_used, "squinting": int(st.squinting),
        "hand_eyes": int(f.hand_on_eyes), "hand_mouth": int(f.hand_on_mouth),
        "yawning": int(st.yawning), "pose_pitch": f"{st.pose_pitch:.2f}", "pose_yaw": f"{st.pose_yaw:.2f}",
        "nod": int(st.nod), "pose_change": int(st.pose_adapted), "artifact": int(st.artifact),
        "hours": f"{st.hours:.3f}", "circadian": f"{st.circadian:.2f}", "unwell": int(st.unwell),
        "eyes_closed": int(st.eyes_closed), "closure_now": f"{st.closure_now:.3f}",
        "perclos": f"{st.perclos:.4f}", "blink_dur": f"{st.blink_dur:.3f}",
        "blink_rate": f"{st.blink_rate:.1f}", "yawns": st.yawns, "nods": st.nods,
        "raw_index": f"{st.raw_index:.2f}", "index": f"{st.index:.2f}", "level": st.level,
        "alarm": int(st.alarm), "baseline_alarm": int(st.baseline_alarm),
    }


def open_source(src, width, height):
    if src.isdigit():
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(int(src), backend)
        if config.CAMERA_MJPG:
            # Без MJPG многие веб-камеры при 1280x720 дают только 10 кадров/с
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        return cap, True
    return cv2.VideoCapture(src), False


def main():
    ap = argparse.ArgumentParser(description="Определение усталости по видео")
    ap.add_argument("--source", default="0", help="номер камеры или путь к видеофайлу")
    ap.add_argument("--no-calib", action="store_true", help="без персональной калибровки")
    ap.add_argument("--no-closed-calib", action="store_true",
                    help="без этапа калибровки с закрытыми глазами")
    ap.add_argument("--eye-mode", choices=["fused", "ear", "blendshape"], default=config.EYE_MODE,
                    help="признак закрытия глаз: fused (оба), ear (геометрия), blendshape (нейросеть)")
    ap.add_argument("--calib-sec", type=float, default=config.CALIBRATION_SECONDS)
    ap.add_argument("--log", default=None, help="CSV-файл журнала (по умолчанию logs/...)")
    ap.add_argument("--no-display", action="store_true", help="без окна (быстрый анализ видео)")
    ap.add_argument("--no-sound", action="store_true")
    ap.add_argument("--no-mirror", action="store_true", help="не отражать изображение камеры")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--model", default=config.MODEL_PATH)
    ap.add_argument("--no-adapt", action="store_true",
                    help="без адаптации к позе, прищуру и артефактам (прежний алгоритм, для сравнения)")
    ap.add_argument("--no-hand-check", action="store_true", help="не проверять руку у лица")
    ap.add_argument("--no-context", action="store_true",
                    help="не учитывать время за рулём и время суток")
    ap.add_argument("--unwell", action="store_true",
                    help="режим «плохо себя чувствую / принимаю лекарства» (выше чувствительность)")
    ap.add_argument("--start-time", default=None,
                    help="время начала записи ЧЧ:ММ (для видеофайлов; по умолчанию — текущее)")
    ap.add_argument("--hours-driven", type=float, default=0.0,
                    help="сколько часов водитель уже за рулём к началу записи")
    ap.add_argument("--calib-mode", choices=["guided", "video", "auto"], default=None,
                    help="калибровка: guided — с сигналами (по умолчанию для камеры), "
                         "video — по записи, закрытие глаз ищется само (по умолчанию для файла), "
                         "auto — без закрытия глаз, по первой минуте")
    ap.add_argument("--calib-start", type=float, default=0.0,
                    help="с какой секунды видеофайла начинать (там начинается калибровка)")
    ap.add_argument("--profile", default=None,
                    help="имя профиля водителя: если профиля нет — он создаётся после "
                         "полной калибровки, если есть — закрывать глаза не нужно")
    ap.add_argument("--profile-update", action="store_true",
                    help="провести полную калибровку и перезаписать профиль")
    ap.add_argument("--max-width", type=int, default=1280,
                    help="уменьшать кадры шире этого (видео с телефона 1080p обрабатывается быстрее)")
    args = ap.parse_args()
    if args.calib_mode is None:
        args.calib_mode = "guided" if args.source.isdigit() else "video"

    cap, is_camera = open_source(args.source, args.width, args.height)
    if not cap.isOpened():
        sys.exit(f"Не удалось открыть источник видео: {args.source}")
    fps_file = cap.get(cv2.CAP_PROP_FPS) or 30.0

    os.makedirs("logs", exist_ok=True)
    log_path = args.log or os.path.join("logs", datetime.now().strftime("session_%Y%m%d_%H%M%S.csv"))
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(log_file, fieldnames=CSV_FIELDS)
    writer.writeheader()

    start_clock = datetime.now()
    if args.start_time:
        hh, mm = (int(x) for x in args.start_time.split(":"))
        start_clock = start_clock.replace(hour=hh, minute=mm, second=0, microsecond=0)

    profile, profile_path = None, None
    if args.profile:
        os.makedirs(config.PROFILES_DIR, exist_ok=True)
        profile_path = os.path.join(config.PROFILES_DIR, f"{args.profile}.json")
        if os.path.exists(profile_path) and not args.profile_update:
            with open(profile_path, encoding="utf-8") as fh:
                profile = json.load(fh)
            print(f"Профиль «{args.profile}» загружен (создан {profile.get('created', '?')})")
        else:
            print(f"Профиль «{args.profile}» будет создан после полной калибровки")

    try:
        analyzer = FaceAnalyzer(args.model, hand_check=not args.no_hand_check)
    except ModelUnavailable as e:
        sys.exit(str(e))
    calib_sec = args.calib_sec
    if args.calib_mode == "auto" and calib_sec == config.CALIBRATION_SECONDS:
        calib_sec = config.AUTO_CALIB_SEC
    decision = FatigueDecision(calibrate=not args.no_calib, calibration_sec=calib_sec,
                               eye_mode=args.eye_mode,
                               closed_calibration=not args.no_closed_calib,
                               adapt=not args.no_adapt, context=not args.no_context,
                               unwell=args.unwell, start_clock=start_clock,
                               hours_driven=args.hours_driven,
                               calib_mode=args.calib_mode, profile=profile)
    profile_done = profile is not None or profile_path is None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if not is_camera else 0
    face_frames, all_frames, face_hint_shown = 0, 0, False
    events = EventLog()
    real_fps = cap.get(cv2.CAP_PROP_FPS)
    if is_camera:
        print(f"Камера: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, "
              f"заявлено {real_fps:.0f} кадр/с")
    text = TextRenderer()
    beeper = Beeper(enabled=not args.no_sound and not args.no_display)

    if not args.no_display:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    start = time.perf_counter()
    frame_idx, fps, last = 0, 0.0, time.perf_counter()
    level_time = {0: 0.0, 1: 0.0, 2: 0.0}
    prev_t = None
    baseline_alarms, prev_base = 0, False
    last_flush = time.perf_counter()

    # Анализ кадра выполняется в отдельном потоке с ограничением по времени:
    # если библиотека зависнет, окно не «замёрзнет» и программу можно закрыть
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    # job — задание анализа кадра; busy — ждём ли мы его результат
    job: concurrent.futures.Future = concurrent.futures.Future()
    busy, pending_t, stall_start = False, 0.0, None
    stalls = []                      # (время сеанса, длительность) зависаний
    stuck = False                    # модель так и не ответила к моменту выхода
    stall_timeout = None if args.no_display else config.STALL_TIMEOUT_SEC

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if is_camera and not args.no_mirror:
                frame = cv2.flip(frame, 1)
            frame = np.asarray(frame)
            if frame.shape[1] > args.max_width:
                scale = args.max_width / frame.shape[1]
                frame = np.asarray(cv2.resize(frame, (args.max_width, int(frame.shape[0] * scale))))
            if not is_camera:
                # Время кадра берётся из самого файла: телефоны часто пишут видео
                # с переменной частотой кадров, и номер кадра / fps даёт ошибку
                pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                video_t = pos if pos > 0 or frame_idx == 0 else frame_idx / fps_file
                if prev_t is not None and video_t <= prev_t:
                    video_t = prev_t + 1.0 / fps_file
                if video_t < args.calib_start:
                    frame_idx += 1
                    continue

            if not busy:
                t = (time.perf_counter() - start) if is_camera else video_t
                job = pool.submit(analyzer.process, frame, int(t * 1000))
                busy, pending_t = True, t
            try:
                wait = stall_timeout if stall_start is None else 0.01
                f = job.result(timeout=wait)
            except concurrent.futures.TimeoutError:
                # Модель не отвечает: показываем живое изображение с предупреждением
                if stall_start is None:
                    stall_start = time.perf_counter()
                    print(f"[!] Модель не отвечает (время сеанса {pending_t:.0f} с)")
                waited = time.perf_counter() - stall_start + config.STALL_TIMEOUT_SEC
                text.add(f"Модель не отвечает {waited:.0f} с — нажмите Q, чтобы выйти",
                         (12, 12), (80, 80, 255), 22)
                cv2.imshow(WINDOW, text.render(frame))
                key = cv2.waitKeyEx(30)
                if (key in KEYS_QUIT or
                        cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1):
                    stuck = True
                    break
                continue
            if stall_start is not None:
                dur = time.perf_counter() - stall_start + config.STALL_TIMEOUT_SEC
                stalls.append((pending_t, dur))
                print(f"[!] Модель снова отвечает, зависание длилось {dur:.0f} с")
                stall_start = None
            busy = False
            t = pending_t

            st = decision.update(f, t)
            writer.writerow(csv_row(f, st))
            all_frames += 1
            face_frames += int(f.face_found)
            if not face_hint_shown and all_frames >= 100 and face_frames < 0.3 * all_frames:
                face_hint_shown = True
                hint = "Лицо почти не находится — поставьте камеру ближе, лучше горизонтально"
                print("[!]", hint)
                events.add(hint, (80, 80, 255))
            if not profile_done and decision.phase == "monitoring":
                profile_done = True
                if decision.baseline.closed_calibrated:
                    with open(profile_path, "w", encoding="utf-8") as fh:
                        json.dump(decision.profile_dict(args.profile), fh, ensure_ascii=False, indent=2)
                    print(f"Профиль «{args.profile}» сохранён: {profile_path}")
                    events.add(f"Профиль «{args.profile}» сохранён", (200, 200, 200))
                else:
                    print(f"[!] Профиль «{args.profile}» не сохранён: не удалось откалибровать "
                          f"закрытые глаза ({decision.baseline.note})")
            if time.perf_counter() - last_flush > 2.0:   # журнал не теряется при аварийной остановке
                log_file.flush()
                last_flush = time.perf_counter()

            if st.phase == "monitoring" and prev_t is not None:
                level_time[st.level] += t - prev_t
            prev_t = t
            if st.baseline_alarm and not prev_base:
                baseline_alarms += 1
            prev_base = st.baseline_alarm
            if st.alarm or st.beep:
                beeper.beep()
            if st.nod:
                events.add("Сонный кивок", (80, 80, 255))
            if st.pose_adapted:
                events.add("Новая поза — норма обновлена", (200, 200, 200))
            if st.artifact:
                events.add("Чихание/кашель — не учтено", (0, 200, 255))

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(1e-6, now - last)
            last = now
            frame_idx += 1

            if args.no_display:
                if frame_idx % 300 == 0:
                    done = f"{100 * frame_idx / total_frames:3.0f}%  " if total_frames else ""
                    print(f"  {done}{t:7.1f} с  уровень: {config.LEVEL_NAMES[st.level]:<22} "
                          f"индекс {st.index:5.1f}")
                continue

            frame = draw_overlay(frame, f, st, text, fps, events, decision)
            frame = text.render(frame)
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKeyEx(1)
            # Окно закрыли крестиком — завершаем программу
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in KEYS_QUIT:
                break
            if key in KEYS_CALIB:
                decision.recalibrate()
            if key in KEYS_UNWELL:
                on = decision.toggle_unwell()
                events.add("Режим «плохо себя чувствую»: " + ("включён" if on else "выключен"),
                           (0, 165, 255))
            if key in KEYS_SHOT:
                os.makedirs("screenshots", exist_ok=True)
                p = os.path.join("screenshots", datetime.now().strftime("shot_%H%M%S.png"))
                cv2.imwrite(p, frame)
                print("Снимок сохранён:", p)
    finally:
        cap.release()
        log_file.close()
        cv2.destroyAllWindows()
        if stall_start is not None:
            stalls.append((pending_t, time.perf_counter() - stall_start + config.STALL_TIMEOUT_SEC))
        if not stuck:
            analyzer.close()
            pool.shutdown(wait=True)

    if not stuck:
        decision.finish(prev_t or 0.0)
        if not profile_done and decision.phase == "monitoring":
            if decision.baseline.closed_calibrated:
                with open(profile_path, "w", encoding="utf-8") as fh:
                    json.dump(decision.profile_dict(args.profile), fh, ensure_ascii=False, indent=2)
                print(f"Профиль «{args.profile}» сохранён: {profile_path}")
            else:
                print(f"[!] Профиль «{args.profile}» не сохранён: не удалось откалибровать "
                      f"закрытые глаза ({decision.baseline.note})")
    total = sum(level_time.values())
    print("\n=== Итоги сеанса ===")
    b = decision.baseline
    if decision.calibration_incomplete:
        print(f"[!] Калибровка не завершилась: запись или сеанс короче нужного "
              f"(нужно около {decision.calibration_sec:.0f} с"
              f"{' + до ' + str(int(config.CLOSED_SEARCH_SEC)) + ' с на закрытие глаз' if decision.calib_mode == 'video' else ''})")
    elif b.personal:
        print(f"Норма человека: EAR открытых глаз {b.ear_open:.3f}, закрытых {b.ear_closed:.3f} "
              f"(порог {b.ear_threshold:.3f}); нейросеть {b.blink_open:.2f} / {b.blink_closed:.2f}; "
              f"MAR {b.mar:.3f}, наклон {b.pitch:+.1f}°")
    else:
        print(f"Калибровка не проводилась: фиксированный порог EAR {b.ear_threshold:.2f}")
    if b.note:
        print("Калибровка:", b.note)
    if b.state_warning:
        print("[!]", b.state_warning)
    if all_frames:
        share = 100 * face_frames / all_frames
        print(f"Лицо найдено на {share:.0f}% кадров" +
              ("" if share >= 80 else " — мало: поставьте камеру ближе и лучше горизонтально"))
    print(f"Признак закрытия глаз: {decision.eye_mode}; адаптация: "
          f"{'включена' if decision.adapt else 'выключена (прежний алгоритм)'}")
    if decision.unwell:
        print("Режим «плохо себя чувствую» был включён")
    print(f"Время наблюдения: {total:.0f} с")
    for lvl, name in config.LEVEL_NAMES.items():
        share = 100 * level_time[lvl] / total if total else 0
        print(f"  {name:<22} {level_time[lvl]:7.1f} с  ({share:4.1f}%)")
    tt = decision.totals
    print(f"Моргания: {tt['blinks']}, длительные закрытия: {tt['long_closures']}, "
          f"зевки: {tt['yawns']}, кивки: {tt['nods']}, тревоги: {tt['alarms']}")
    print(f"Смены позы: {tt['pose_changes']}, отброшено закрытий (чихание/кашель/рука): "
          f"{tt['artifacts']}, перерывы: {tt['breaks']}")
    print(f"Срабатывания порогового метода: {baseline_alarms}")
    if stalls:
        print("Зависания модели: " + ", ".join(f"на {s:.0f}-й с — {d:.0f} с" for s, d in stalls))
    print("Журнал:", log_path)
    if stuck:
        # Зависший поток библиотеки не даёт Python завершиться обычным способом
        print("Модель так и не ответила — принудительное завершение.")
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
