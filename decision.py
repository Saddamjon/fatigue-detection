"""
Модуль принятия решения об уровне усталости.

Этапы:
  1. Персональная двухточечная калибровка — признаки конкретного человека
     при открытых и закрытых глазах в его ПРИВЫЧНОЙ позе за рулём.
  2. Адаптация к позе: норма положения головы обновляется, когда водитель
     устойчиво меняет позу (откинулся, облокотился); норма открытых глаз
     запоминается отдельно для каждого наклона головы.
  3. Определение состояния глаз: открыт / закрыт / не виден. Прищур и
     сильный поворот головы не путаются с закрытием глаз (используется
     оценка нейросети); рука на глазах — кадр не учитывается.
  4. События: моргания, длительные закрытия, зевки, кивки (медленное падение
     головы и резкий возврат). Короткие закрытия при чихании и кашле
     отбрасываются как артефакты.
  5. Временные окна — PERCLOS, длительность морганий, число зевков и кивков.
  6. Нечёткий вывод — индекс усталости 0..100; контекст (время за рулём,
     время суток) усиливает признаки с лица, но сам тревогу не вызывает.
  7. Сглаживание и гистерезис — защита от ложных переключений.
  8. Правило безопасности — закрытие глаз дольше порога вызывает тревогу
     немедленно. Фильтры артефактов это правило не отключают.
"""
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

import config
import fuzzy


def circadian_risk(hour):
    """Риск сонливости 0..1 по времени суток (час может быть дробным)."""
    xs = [a for a, _ in config.CIRCADIAN_TABLE]
    ys = [b for _, b in config.CIRCADIAN_TABLE]
    return float(np.interp(hour % 24, xs, ys))


@dataclass
class Baseline:
    """Личная норма человека (из калибровки)."""
    ear_open: float = config.DEFAULT_EAR
    ear_closed: float = config.DEFAULT_EAR * config.FALLBACK_CLOSED_EAR_RATIO
    blink_open: float = 0.0
    blink_closed: float = config.FALLBACK_CLOSED_BLINK
    squint: float = 0.0
    mar: float = config.DEFAULT_MAR
    pitch: float = config.DEFAULT_PITCH
    yaw: float = config.DEFAULT_YAW
    personal: bool = False          # была ли калибровка открытых глаз
    closed_calibrated: bool = False  # удался ли этап закрытых глаз
    note: str = ""
    state_warning: str = ""         # отличие от профиля (возможна усталость уже на старте)

    def ear_threshold_for(self, ear_open):
        """Порог EAR (ниже — глаз закрыт) при заданной норме открытых глаз."""
        if not self.personal:
            return config.BASELINE_EAR_THRESHOLD
        ear_closed = self.ear_closed * ear_open / self.ear_open
        return ear_closed + config.CLOSED_LEVEL * (ear_open - ear_closed)

    @property
    def ear_threshold(self):
        return self.ear_threshold_for(self.ear_open)

    @property
    def yawn_threshold(self):
        return max(config.YAWN_MAR_MIN, self.mar + config.YAWN_MAR_DELTA)

    def openness(self, ear, blink, mode=config.EYE_MODE, ear_open=None):
        """
        Открытость глаза 0..1 внутри личного диапазона:
        0 — как при закрытых глазах на калибровке, 1 — как при открытых.
        ear_open — норма открытых глаз для текущего наклона головы
        (уровень закрытых глаз масштабируется пропорционально).
        """
        e_open = self.ear_open if ear_open is None else ear_open
        e_closed = self.ear_closed * e_open / self.ear_open
        o_ear = (ear - e_closed) / max(1e-6, e_open - e_closed)
        o_bs = (self.blink_closed - blink) / max(1e-6, self.blink_closed - self.blink_open)
        o_ear, o_bs = float(np.clip(o_ear, 0, 1)), float(np.clip(o_bs, 0, 1))
        if mode == "ear":
            return o_ear
        if mode == "blendshape":
            return o_bs
        w = config.FUSION_WEIGHT_EAR
        return w * o_ear + (1 - w) * o_bs


class PoseEyeNorm:
    """
    Норма открытых глаз (EAR) для разных наклонов головы.
    Учится только на кадрах, где нейросеть уверенно видит открытый глаз,
    и не может уйти далеко от калиброванной нормы — поэтому не «привыкает»
    к постепенно закрывающимся глазам уставшего водителя.
    """

    def __init__(self, baseline):
        self.b = baseline
        self.bins = {}

    def _key(self, pitch):
        return int(round((pitch - self.b.pitch) / config.POSE_BIN_DEG))

    def learn(self, pitch, ear):
        key = self._key(pitch)
        if key == 0:
            return              # положение калибровки: норма уже измерена
        d = self.bins.setdefault(key, [])
        if len(d) < config.POSE_BIN_MAXLEN:
            d.append(ear)       # заполненная норма замораживается — не дрейфует со временем

    def ear_open(self, pitch):
        """Норма для текущего наклона и признак того, что она выучена."""
        d = self.bins.get(self._key(pitch))
        if d is None or len(d) < config.POSE_BIN_MIN_SAMPLES:
            return self.b.ear_open, False
        v = float(np.median(d))
        lo, hi = self.b.ear_open * config.POSE_NORM_MIN_RATIO, self.b.ear_open * config.POSE_NORM_MAX_RATIO
        return float(np.clip(v, lo, hi)), True


class PoseTracker:
    """
    Норма положения головы и обнаружение сонных кивков.

    Смена позы: голова неподвижна в новом положении POSE_ADAPT_SEC секунд
    при открытых глазах — это новая норма (откинулся, облокотился).
    Кивок: голова медленно уходит от нормы и резко возвращается. Ровное
    смещение без рывка и быстрый взгляд вниз кивком не считаются.
    """

    def __init__(self, pitch, yaw):
        self.calib_pitch, self.calib_yaw = pitch, yaw
        self.pitch, self.yaw = pitch, yaw
        self.hist = deque()
        self.adaptations = 0
        self.jerk = False
        self._short = deque()
        self._state = "rest"
        self._t_rest = None
        self._track = []
        self._sign = 0

    def update(self, t, pitch, yaw, eyes_open):
        """Возвращает (кивок, норма_обновлена). Признак рывка — в self.jerk."""
        self._short.append((t, pitch))
        while self._short and self._short[0][0] < t - config.ARTIFACT_JERK_WINDOW_SEC:
            self._short.popleft()
        ps = [p for _, p in self._short]
        self.jerk = max(ps) - min(ps) >= config.ARTIFACT_JERK_DEG
        if config.POSE_ADAPT and self._adapt(t, pitch, yaw, eyes_open):
            return False, True
        return self._nod(t, pitch), False

    def _adapt(self, t, pitch, yaw, eyes_open):
        self.hist.append((t, pitch, yaw, eyes_open))
        while self.hist and self.hist[0][0] < t - config.POSE_ADAPT_SEC:
            self.hist.popleft()
        if len(self.hist) < 5 or t - self.hist[0][0] < config.POSE_ADAPT_SEC * 0.9:
            return False
        arr = np.array([(p, y, o) for _, p, y, o in self.hist], dtype=float)
        if arr[:, 2].mean() < config.POSE_MIN_OPEN_SHARE:
            return False        # глаза закрыты — это может быть засыпание, а не смена позы
        if arr[:, 0].std() > config.POSE_STABLE_STD_PITCH or arr[:, 1].std() > config.POSE_STABLE_STD_YAW:
            return False        # голова ещё двигается
        new_p, new_y = float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))
        if abs(new_p - self.pitch) < config.POSE_MIN_CHANGE_DEG and abs(new_y - self.yaw) < config.POSE_MIN_CHANGE_DEG:
            return False
        if (abs(new_p - self.calib_pitch) > config.POSE_MAX_SHIFT_DEG or
                abs(new_y - self.calib_yaw) > config.POSE_MAX_SHIFT_DEG):
            return False
        self.pitch, self.yaw = new_p, new_y
        self.adaptations += 1
        self._state, self._t_rest = "rest", t
        return True

    def _nod(self, t, pitch):
        dev = pitch - self.pitch
        a = abs(dev)
        if self._state == "rest":
            if a <= config.NOD_REST_DEG:
                self._t_rest = t
            elif a >= config.NOD_DROP_DEG:
                self._state = "down"
                self._sign = 1 if dev > 0 else -1
                self._track = [(t, a)]
                if self._t_rest is None:
                    self._t_rest = t
            return False
        same_side = (dev > 0) == (self._sign > 0)
        if same_side:
            self._track.append((t, a))
        peak = max(x for _, x in self._track)
        if a <= peak * config.NOD_RETURN_SHARE or not same_side:
            # Падение — до ПЕРВОГО достижения нижней точки, возврат — от ПОСЛЕДНЕГО:
            # взгляд на приборы (быстро вниз, задержка, вверх) так не похож на кивок
            near = [tt for tt, x in self._track if x >= peak * config.NOD_PEAK_SHARE]
            fall = near[0] - self._t_rest
            recovery = t - near[-1]
            self._state = "rest"
            self._t_rest = t if a <= config.NOD_REST_DEG else None
            return (fall >= config.NOD_MIN_FALL_SEC and recovery <= config.NOD_MAX_RECOVERY_SEC
                    and recovery < fall)
        return False


@dataclass
class State:
    """Результат работы алгоритма на текущем кадре."""
    t: float
    phase: str                    # "calibration" | "monitoring"
    calib_step: str = ""          # "open" | "prepare" | "closed"
    calib_seconds_left: float = 0.0
    calibration_progress: float = 0.0
    beep: bool = False            # подать звуковой сигнал на этом кадре
    face_found: bool = False
    no_face_warning: bool = False
    eye_state: str = "open"       # "open" | "closed" | "unknown"
    eyes_closed: bool = False
    eyes_hidden: bool = False     # рука на глазах или голова сильно повёрнута
    squinting: bool = False
    head_turned: bool = False
    eye_mode_used: str = ""       # какой признак глаз использован на кадре
    openness: float = 1.0         # открытость глаза 0..1 (личный диапазон)
    ear_open_norm: float = 0.0    # норма открытых глаз для текущего наклона
    ear_threshold: float = 0.0
    ear_smooth: float = 0.0
    blink_smooth: float = 0.0
    yawning: bool = False
    pose_pitch: float = 0.0       # текущая норма положения головы
    pose_yaw: float = 0.0
    nod: bool = False             # на этом кадре обнаружен кивок
    pose_adapted: bool = False    # на этом кадре обновлена норма позы
    artifact: bool = False        # на этом кадре отброшено закрытие (чихание/кашель)
    level: int = 0                # 0 норма, 1 начальная усталость, 2 опасная сонливость
    index: float = 0.0            # сглаженный индекс усталости
    raw_index: float = 0.0
    alarm: bool = False           # немедленная тревога (длительное закрытие глаз)
    reason: str = ""
    advice: str = ""              # совет (перерыв), не влияет на уровень
    perclos: float = 0.0
    blink_dur: float = 0.0
    blink_rate: float = 0.0       # морганий в минуту
    yawns: int = 0
    nods: int = 0
    closure_now: float = 0.0      # длительность текущего закрытия глаз, с
    hours: float = 0.0            # время за рулём без перерыва
    circadian: float = 0.0        # риск по времени суток
    unwell: bool = False
    baseline_alarm: bool = False  # решение классического порогового метода
    totals: dict = field(default_factory=dict)


class _Episode:
    """Отслеживает непрерывный интервал, когда условие истинно."""

    def __init__(self):
        self.start = None

    def update(self, active, t):
        """Возвращает (начало, длительность) завершившегося эпизода или None."""
        if active and self.start is None:
            self.start = t
        elif not active and self.start is not None:
            s, self.start = self.start, None
            return s, t - s
        return None

    def cancel(self):
        self.start = None

    def duration(self, t):
        return 0.0 if self.start is None else t - self.start


class FatigueDecision:
    def __init__(self, calibrate=True, calibration_sec=None,
                 eye_mode=config.EYE_MODE, closed_calibration=config.CALIB_CLOSED_EYES,
                 adapt=True, context=config.CONTEXT, unwell=False,
                 start_clock=None, hours_driven=0.0, calib_mode="guided", profile=None):
        """
        calib_mode: "guided" — калибровка с сигналами (веб-камера);
                    "video"  — по записи: закрытие глаз ищется в видео автоматически;
                    "auto"   — без закрытия глаз, по первой минуте.
        profile: сохранённый профиль водителя (словарь) — этап закрытых глаз не нужен.
        """
        self.calibrate = calibrate
        self.calib_mode = calib_mode
        self.profile = profile
        if calibration_sec is None:
            calibration_sec = config.AUTO_CALIB_SEC if calib_mode == "auto" else config.CALIBRATION_SECONDS
        self.calibration_sec = calibration_sec
        self.eye_mode = eye_mode
        self.closed_calibration = closed_calibration
        self.adapt = adapt              # адаптация к позе, прищуру, артефактам
        self.context = context
        self.unwell = unwell
        self.start_clock = start_clock or datetime.now()
        self.hours_driven = hours_driven
        self.baseline = Baseline()
        self.pose = PoseTracker(self.baseline.pitch, self.baseline.yaw)
        self.eye_norm = PoseEyeNorm(self.baseline)
        self._calib = {"ear": [], "blink": [], "squint": [], "mar": [], "pitch": [], "yaw": []}
        self._calib_closed = {"ear": [], "blink": []}
        self._search = []           # (t, ear, blink, closed_like) при поиске закрытия в записи
        self._calib_start = None
        self._step = "open"
        self._step_start = None
        self.phase = "calibration" if calibrate else "monitoring"

        self._eye = _Episode()          # закрытие глаз для статистики (моргания, PERCLOS)
        self._alarm_eye = _Episode()    # закрытие глаз для правила безопасности
        self._yawn = _Episode()
        self._yawn_counted = False
        self._static_nod = _Episode()   # кивок по старому правилу (без адаптации)
        self._static_nod_counted = False
        self._baseline_eye = _Episode()

        self._frames = deque()   # (t, closed) — для PERCLOS
        self._blinks = deque()   # (t_end, duration)
        self._pending = deque()  # закрытия, ожидающие проверки на артефакт
        self._yawns = deque()    # t
        self._nods = deque()     # t
        self._jerks = deque()    # моменты резких рывков головы
        self._hand_mouth = deque()  # моменты, когда рука у рта (кратковременно)
        self._hand_mouth_ep = _Episode()
        self._last_face_t = None
        self._monitor_start = None
        self._drive_start = None
        self._unknown_since = None

        self._recent = deque(maxlen=config.MEDIAN_FRAMES)   # (ear, blink) для медианы
        self._frame_dt = None    # средний интервал между кадрами

        self._index = 0.0
        self._last_t = None
        self._level = 0
        self._candidate = 0
        self._candidate_since = None

        self.totals = {"blinks": 0, "long_closures": 0, "yawns": 0, "nods": 0, "alarms": 0,
                       "artifacts": 0, "pose_changes": 0, "breaks": 0}
        self._alarm_active = False

    # ------------------------------------------------------------------
    def recalibrate(self):
        drive_start = self._drive_start   # время за рулём при перекалибровке не сбрасывается
        self.__init__(calibrate=True, calibration_sec=self.calibration_sec,
                      eye_mode=self.eye_mode, closed_calibration=self.closed_calibration,
                      adapt=self.adapt, context=self.context, unwell=self.unwell,
                      start_clock=self.start_clock, hours_driven=self.hours_driven,
                      calib_mode=self.calib_mode, profile=self.profile)
        self._drive_start = drive_start

    def toggle_unwell(self):
        self.unwell = not self.unwell
        return self.unwell

    @property
    def level_thresholds(self):
        if self.unwell:
            return config.UNWELL_LEVEL_TIRED, config.UNWELL_LEVEL_DANGER
        return config.LEVEL_TIRED, config.LEVEL_DANGER

    @property
    def alarm_closure_sec(self):
        return config.UNWELL_ALARM_CLOSURE_SEC if self.unwell else config.ALARM_CLOSURE_SEC

    # ------------------------------------------------------------------
    def profile_dict(self, name):
        """Данные для сохранения профиля водителя (только после полной калибровки)."""
        b = self.baseline
        return {
            "name": name,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "ear_closed_ratio": round(b.ear_closed / b.ear_open, 4),
            "blink_open": round(b.blink_open, 4),
            "blink_closed": round(b.blink_closed, 4),
            "closed_calibrated": b.closed_calibrated,
            "ear_open_reference": round(b.ear_open, 4),   # для справки: зависит от камеры
        }

    def _apply_profile(self):
        """Профиль вместо этапа закрытых глаз: уровень закрытых глаз = норма × коэффициент."""
        b, p = self.baseline, self.profile
        b.ear_closed = b.ear_open * float(p["ear_closed_ratio"])
        if float(p["blink_closed"]) > b.blink_open + 0.2:
            b.blink_closed = float(p["blink_closed"])
        b.closed_calibrated = bool(p.get("closed_calibrated", False))
        b.note = f"использован профиль «{p.get('name', '')}» от {p.get('created', '?')}"
        if b.blink_open - float(p["blink_open"]) > config.PROFILE_ALERT_DELTA:
            b.state_warning = ("Глаза прикрыты сильнее, чем обычно для вас — "
                               "возможно, вы уже устали")

    def _finish_open_step(self):
        """Итог первого этапа: «нормальные» значения при открытых глазах."""
        c = self._calib
        # Норма открытых глаз — по кадрам, где нейросеть видит открытый глаз
        # (моргания и прищуривания в норму не попадают)
        blinks = np.array(c["blink"])
        open_idx = blinks <= max(config.OPEN_FRAME_BLINK_MAX, float(np.median(blinks)))
        ears = np.array(c["ear"])
        if open_idx.sum() >= config.MIN_CALIBRATION_FRAMES // 2:
            ears, blinks = ears[open_idx], blinks[open_idx]
        ear = float(np.median(ears))
        if not (config.CALIB_EAR_MIN < ear < config.CALIB_EAR_MAX):
            self.baseline = Baseline(note=f"калибровка отклонена: EAR {ear:.3f} вне диапазона")
            self._reset_pose_models()
            return False
        blink_open = float(np.median(blinks))
        self.baseline = Baseline(
            ear_open=ear,
            ear_closed=ear * config.FALLBACK_CLOSED_EAR_RATIO,
            blink_open=blink_open,
            blink_closed=max(config.FALLBACK_CLOSED_BLINK, blink_open + 0.2),
            squint=float(np.median(c["squint"])),
            mar=float(np.median(c["mar"])),
            pitch=float(np.median(c["pitch"])), yaw=float(np.median(c["yaw"])),
            personal=True,
            note="этап закрытых глаз не проводился")
        self._reset_pose_models()
        return True

    def _reset_pose_models(self):
        self.pose = PoseTracker(self.baseline.pitch, self.baseline.yaw)
        self.eye_norm = PoseEyeNorm(self.baseline)

    def _finish_closed_step(self):
        """Итог второго этапа: уровень признаков при закрытых глазах."""
        b = self.baseline
        cc = self._calib_closed
        if len(cc["ear"]) < 3:
            b.note = "закрытые глаза: лицо не найдено, использован стандартный коэффициент"
            return
        ear_closed = float(np.median(cc["ear"]))
        if ear_closed > b.ear_open * (1 - config.CLOSED_MIN_DROP):
            b.note = (f"закрытие глаз не обнаружено (EAR {ear_closed:.3f} при открытых "
                      f"{b.ear_open:.3f}), использован стандартный коэффициент")
            return
        b.ear_closed = ear_closed
        blink_closed = float(np.median(cc["blink"]))
        if blink_closed > b.blink_open + 0.2:
            b.blink_closed = blink_closed
        b.closed_calibrated = True
        b.note = "двухточечная калибровка выполнена"

    def _calibration_update(self, f, t, st):
        if self._calib_start is None:
            self._calib_start = self._step_start = t
        elapsed = t - self._step_start
        st.calib_step = self._step

        if self._step == "open":
            if f.face_found:
                for k in self._calib:
                    self._calib[k].append(getattr(f, k))
            enough = len(self._calib["ear"]) >= config.MIN_CALIBRATION_FRAMES
            st.calib_seconds_left = max(0.0, self.calibration_sec - elapsed)
            # Если лицо было видно слишком мало кадров, этап продлевается
            st.calibration_progress = min(1.0 if enough else 0.99, elapsed / self.calibration_sec)
            if elapsed >= self.calibration_sec and enough:
                ok = self._finish_open_step()
                if ok and self.profile:
                    self._apply_profile()          # профиль: закрывать глаза не нужно
                    self.phase = "monitoring"
                elif ok and self.closed_calibration and self.calib_mode == "guided":
                    self._step, self._step_start = "prepare", t
                elif ok and self.closed_calibration and self.calib_mode == "video":
                    self._step, self._step_start = "search", t
                else:
                    if ok and self.calib_mode == "auto":
                        self.baseline.note = "автокалибровка (без этапа закрытых глаз)"
                    self.phase = "monitoring"
            return

        if self._step == "search":
            self._search_update(f, t, st, elapsed)
            return

        if self._step == "prepare":
            st.calib_seconds_left = max(0.0, config.CLOSED_PREPARE_SEC - elapsed)
            st.calibration_progress = 1.0
            if elapsed >= config.CLOSED_PREPARE_SEC:
                self._step, self._step_start = "closed", t
                st.beep = True            # сигнал «закройте глаза»
            return

        if self._step == "closed":
            st.calib_seconds_left = max(0.0, config.CLOSED_HOLD_SEC - elapsed)
            st.calibration_progress = 1.0
            if f.face_found and elapsed >= config.CLOSED_SKIP_SEC:
                self._calib_closed["ear"].append(f.ear)
                self._calib_closed["blink"].append(f.blink)
            if elapsed >= config.CLOSED_HOLD_SEC:
                self._finish_closed_step()
                self.phase = "monitoring"
                st.beep = True            # сигнал «откройте глаза»

    def _search_update(self, f, t, st, elapsed):
        """Калибровка по записи: найти в видео отрезок, где человек закрыл глаза."""
        b = self.baseline
        st.calib_seconds_left = max(0.0, config.CLOSED_SEARCH_SEC - elapsed)
        st.calibration_progress = 1.0
        if f.face_found:
            closed_like = (f.ear <= b.ear_open * (1 - config.CLOSED_MIN_DROP) and
                           (f.blink >= config.CLOSED_SEARCH_BLINK or f.ear <= b.ear_open * 0.5))
            self._search.append((t, f.ear, f.blink, closed_like))
        run = self._longest_closed_run()
        # Закрытие уже закончилось (глаза снова открыты) — можно не ждать конца поиска
        reopened = (run is not None and self._search and not self._search[-1][3]
                    and t - run[1] > config.CLOSED_SEARCH_GAP_SEC)
        if reopened or elapsed >= config.CLOSED_SEARCH_SEC:
            self._finish_search()

    def _finish_search(self):
        """Итог поиска закрытых глаз в записи (вызывается и при конце записи)."""
        b = self.baseline
        run = self._longest_closed_run()
        if run is not None:
            s, e = run[0] + config.CLOSED_SEARCH_TRIM_SEC, run[1] - config.CLOSED_SEARCH_TRIM_SEC
            for tt, ear, blink, _ in self._search:
                if s <= tt <= e:
                    self._calib_closed["ear"].append(ear)
                    self._calib_closed["blink"].append(blink)
            self._finish_closed_step()
            if b.closed_calibrated:
                b.note = f"двухточечная калибровка по записи (глаза закрыты {run[1] - run[0]:.1f} с)"
        else:
            b.note = "в записи не найдено закрытие глаз — использован стандартный коэффициент"
        self.phase = "monitoring"

    def _longest_closed_run(self):
        """Самый длинный отрезок «глаза закрыты» (с допуском на 1–2 ошибочных кадра)."""
        best, start, last = None, None, None
        for tt, _, _, closed in self._search:
            if closed:
                if start is None or tt - last > config.CLOSED_SEARCH_GAP_SEC:
                    start = tt
                last = tt
                if last - start >= config.CLOSED_SEARCH_MIN_SEC and (best is None or last - start > best[1] - best[0]):
                    best = (start, last)
        return best

    # ------------------------------------------------------------------
    def _smoothed(self, f, t):
        """Медиана по последним кадрам — только при достаточной частоте кадров."""
        if self._last_t is not None:
            dt = t - self._last_t
            if dt > 0:
                self._frame_dt = dt if self._frame_dt is None else 0.9 * self._frame_dt + 0.1 * dt
        self._recent.append((f.ear, f.blink))
        fps = 1.0 / self._frame_dt if self._frame_dt else 0.0
        if fps >= config.MEDIAN_MIN_FPS and len(self._recent) == self._recent.maxlen:
            arr = np.array(self._recent)
            return float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))
        return f.ear, f.blink

    def _remove_frames(self, start, end):
        """Убрать из окна PERCLOS кадры отброшенного закрытия."""
        self._frames = deque((ft, c) for ft, c in self._frames if not (start <= ft <= end))

    def _cancel_episode(self, t):
        """Закрытие глаз прервано тем, что глаза перестали быть видны: не учитываем его."""
        if self._eye.start is not None:
            self._remove_frames(self._eye.start, t)
            self._eye.cancel()

    @staticmethod
    def _any_in(times, a, b):
        return any(a <= x <= b for x in times)

    def _is_artifact(self, s, e):
        """Короткое закрытие при чихании или кашле."""
        jerk = self._any_in(self._jerks, s - config.ARTIFACT_WINDOW_BEFORE, e + config.ARTIFACT_WINDOW_AFTER)
        nod = self._any_in(self._nods, s - config.NOD_EXCLUDE_WINDOW_SEC, e + config.NOD_EXCLUDE_WINDOW_SEC)
        hand = self._any_in(self._hand_mouth, s - config.HAND_MOUTH_WINDOW_SEC, e + config.HAND_MOUTH_WINDOW_SEC)
        # Рывок при сонном кивке — признак усталости, а не помеха
        return (jerk and not nod) or hand

    def _resolve_episodes(self, t, st, force=False):
        """Закрытие глаз засчитывается через ARTIFACT_WINDOW_AFTER секунд после окончания,
        когда уже известно, был ли вокруг него рывок головы или рука у рта."""
        while self._pending and (force or t >= self._pending[0][1] + config.ARTIFACT_WINDOW_AFTER):
            s, e, d = self._pending.popleft()
            if d < 0.05:
                continue
            if self.adapt and d < config.ARTIFACT_MAX_CLOSURE_SEC and self._is_artifact(s, e):
                self._remove_frames(s, e)
                self.totals["artifacts"] += 1
                st.artifact = True
                continue
            self._blinks.append((e, min(d, 2.0)))
            self.totals["blinks"] += 1
            if d >= config.LONG_CLOSURE_SEC:
                self.totals["long_closures"] += 1

    def finish(self, t):
        """Конец сеанса: засчитать закрытия, ожидающие проверки;
        если запись кончилась во время поиска закрытых глаз — подвести его итог."""
        if self.phase == "calibration" and self._step == "search":
            self._finish_search()
        st = State(t=t, phase=self.phase)
        self._resolve_episodes(t, st, force=True)

    @property
    def calibration_incomplete(self):
        return self.phase == "calibration"

    # ------------------------------------------------------------------
    def update(self, f, t):
        """f — FrameFeatures, t — время кадра в секундах."""
        st = State(t=t, phase=self.phase, face_found=f.face_found, unwell=self.unwell)

        # ---------- 1. Калибровка ----------
        if self.phase == "calibration":
            # st.phase остаётся "calibration": этот кадр относится к калибровке,
            # даже если на нём калибровка завершилась
            self._calibration_update(f, t, st)
            st.totals = dict(self.totals)
            return st

        if self._monitor_start is None:
            self._monitor_start = t
        if self._drive_start is None:
            self._drive_start = t
        b = self.baseline
        st.pose_pitch, st.pose_yaw = self.pose.pitch, self.pose.yaw
        self._prune(t)
        self._resolve_episodes(t, st)
        self._context(st, t)

        # ---------- Нет лица ----------
        if not f.face_found:
            st.eye_state = "unknown"
            st.no_face_warning = (self._last_face_t is not None and
                                  t - self._last_face_t >= config.NO_FACE_WARN_SEC)
            self._cancel_episode(t)
            # Если глаза были закрыты и лицо пропало (голова упала из кадра),
            # отсчёт закрытия для правила безопасности продолжается
            st.closure_now = self._alarm_eye.duration(t)
            self._apply_alarm(st)
            self._fill_from_windows(st, t)
            st.index = self._index
            st.level = 2 if st.alarm else self._level
            if st.alarm:
                st.reason = f"глаза закрылись и лицо пропало ({st.closure_now:.1f} с)"
            elif st.no_face_warning:
                st.reason = "лицо не найдено"
            st.totals = dict(self.totals)
            self._last_t = t
            return st
        if self._last_face_t is not None and t - self._last_face_t >= config.BREAK_MIN_SEC:
            self._drive_start = t                 # долго не было лица — это был перерыв
            self.totals["breaks"] += 1
            self._context(st, t)
        self._last_face_t = t

        # ---------- 2. Рот и зевки ----------
        ear, blink = self._smoothed(f, t)
        st.ear_smooth, st.blink_smooth = ear, blink
        self._hand_mouth_ep.update(f.hand_on_mouth, t)
        if f.hand_on_mouth:
            if self._hand_mouth_ep.duration(t) <= config.HAND_MOUTH_MAX_SEC:
                self._hand_mouth.append(t)
            else:
                # Рука у рта слишком долго — это поза, а не кашель: забываем этот эпизод
                start = self._hand_mouth_ep.start
                self._hand_mouth = deque(x for x in self._hand_mouth if x < start)
        mouth_open = f.mar > b.yawn_threshold and not f.hand_on_mouth
        if self._yawn.update(mouth_open, t) is not None:
            self._yawn_counted = False
        yawn_dur = self._yawn.duration(t)
        st.yawning = mouth_open and yawn_dur <= config.YAWN_MAX_SEC
        if mouth_open and not self._yawn_counted and yawn_dur >= config.YAWN_MIN_SEC:
            self._yawns.append(t)
            self._yawn_counted = True
            self.totals["yawns"] += 1

        # ---------- 3. Состояние глаз ----------
        self._eye_state(f, ear, blink, st)

        # ---------- 4. Поза и кивки ----------
        if b.personal:
            if self.adapt:
                nod, adapted = self.pose.update(t, f.pitch, f.yaw, st.eye_state == "open")
                if self.pose.jerk:
                    self._jerks.append(t)
                if adapted:
                    st.pose_adapted = True
                    self.totals["pose_changes"] += 1
                st.pose_pitch, st.pose_yaw = self.pose.pitch, self.pose.yaw
            else:
                nod = self._static_nod_update(f, t)
            if nod:
                st.nod = True
                self._nods.append(t)
                self.totals["nods"] += 1

        # ---------- 5. События глаз ----------
        # Правило безопасности: все кадры, где глаза видны (включая зевок)
        if st.eye_state == "unknown":
            self._alarm_eye.cancel()
        else:
            self._alarm_eye.update(st.eye_state == "closed", t)
        # Статистика: зевок — глаза зажмурены, такие кадры не учитываем
        stats_state = "unknown" if st.yawning else st.eye_state
        if stats_state == "unknown":
            self._cancel_episode(t)
        else:
            closed = stats_state == "closed"
            ended = self._eye.update(closed, t)
            if ended is not None:
                self._pending.append((ended[0], t, ended[1]))
            self._frames.append((t, closed))
        if st.eye_state == "unknown":
            self._unknown_since = self._unknown_since if self._unknown_since is not None else t
        else:
            self._unknown_since = None
        st.closure_now = self._alarm_eye.duration(t)

        # Классический метод для сравнения: фиксированный порог EAR (сырой, без фильтра)
        base_closed = f.ear < config.BASELINE_EAR_THRESHOLD
        self._baseline_eye.update(base_closed, t)
        st.baseline_alarm = self._baseline_eye.duration(t) >= config.BASELINE_CLOSURE_SEC

        # ---------- 6. Временные окна ----------
        self._fill_from_windows(st, t)

        # ---------- 7. Нечёткий вывод ----------
        # Первые секунды окно почти пустое: одно моргание дало бы PERCLOS 100%.
        # Поэтому решение принимается только после накопления данных.
        warmup = t - self._monitor_start < config.WARMUP_SEC
        if warmup:
            raw, details = self._index, None
        else:
            values = {"perclos": st.perclos, "blink_dur": st.blink_dur,
                      "yawns": st.yawns, "nods": st.nods}
            if self.context:
                values.update(hours=st.hours, circadian=st.circadian)
            raw, details = fuzzy.infer(values, return_details=True)
        st.raw_index = raw

        # ---------- 8. Сглаживание и гистерезис ----------
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        a = 1.0 - math.exp(-dt / config.SMOOTHING_TAU_SEC) if dt > 0 else 1.0
        self._index += a * (raw - self._index)
        st.index = self._index

        tired, danger = self.level_thresholds
        candidate = 2 if self._index >= danger else 1 if self._index >= tired else 0
        if candidate != self._candidate:
            self._candidate, self._candidate_since = candidate, t
        if (candidate != self._level and
                t - self._candidate_since >= config.LEVEL_HOLD_SEC):
            self._level = candidate

        # ---------- 9. Правило безопасности ----------
        self._apply_alarm(st)
        st.level = 2 if st.alarm else self._level
        st.reason = self._explain(st, details)
        st.totals = dict(self.totals)
        return st

    # ------------------------------------------------------------------
    def _eye_state(self, f, ear, blink, st):
        """Открыт / закрыт / не виден — с учётом прищура, поворота и наклона головы."""
        b = self.baseline
        if not b.personal:
            # Без калибровки — классический фиксированный порог EAR
            st.eye_mode_used = "fixed"
            st.ear_open_norm = config.DEFAULT_EAR
            st.ear_threshold = config.BASELINE_EAR_THRESHOLD
            st.openness = float(np.clip(ear / config.DEFAULT_EAR, 0, 1))
            if f.hand_on_eyes:
                st.eye_state, st.eyes_hidden = "unknown", True
            else:
                st.eye_state = "closed" if ear < config.BASELINE_EAR_THRESHOLD else "open"
            st.eyes_closed = st.eye_state == "closed"
            return

        yaw_dev = abs(f.yaw - b.yaw)
        st.head_turned = yaw_dev > config.MAX_YAW_DEVIATION_DEG
        mode = self.eye_mode
        if self.adapt:
            st.squinting = (f.squint - b.squint > config.SQUINT_DELTA and
                            blink < (b.blink_open + b.blink_closed) / 2)
        hidden = f.hand_on_eyes or yaw_dev > config.EYES_HIDDEN_YAW_DEG
        if st.head_turned and (mode != "fused" or not self.adapt):
            hidden = True               # без нейросетевого признака повёрнутую голову не оцениваем
        if self.adapt and mode == "fused" and (st.squinting or st.head_turned):
            mode = "blendshape"         # при прищуре и повороте EAR ненадёжен
        st.eye_mode_used = mode

        ear_open, _learned = (self.eye_norm.ear_open(f.pitch)
                              if self.adapt and config.POSE_EYE_NORM else (b.ear_open, False))
        st.ear_open_norm = ear_open
        st.ear_threshold = b.ear_threshold_for(ear_open)
        if hidden:
            st.eye_state, st.eyes_hidden = "unknown", True
            st.openness = float("nan")
            return
        st.openness = b.openness(ear, blink, mode, ear_open)
        st.eye_state = "closed" if st.openness < config.CLOSED_LEVEL else "open"
        st.eyes_closed = st.eye_state == "closed"

        # Обучение нормы глаз для текущего наклона: только когда нейросеть уверенно
        # видит открытый глаз и нет опасной сонливости. Норма диапазона учится один
        # раз (затем заморожена) и ограничена — так она не «привыкает» к усталости
        if (self.adapt and config.POSE_EYE_NORM and not st.squinting and not st.yawning
                and not st.head_turned and self._level < 2
                and blink <= b.blink_open + config.POSE_LEARN_BLINK_MARGIN):
            self.eye_norm.learn(f.pitch, ear)

    def _static_nod_update(self, f, t):
        """Прежнее правило кивка (для сравнения): отклонение от калибровки дольше 0,4 с."""
        b = self.baseline
        nodding = abs(f.pitch - b.pitch) > config.NOD_STATIC_DEG
        if self._static_nod.update(nodding, t) is not None:
            self._static_nod_counted = False
        if nodding and not self._static_nod_counted and self._static_nod.duration(t) >= config.NOD_STATIC_SEC:
            self._static_nod_counted = True
            return True
        return False

    def _apply_alarm(self, st):
        need = self.alarm_closure_sec + (config.YAWN_ALARM_EXTRA_SEC if st.yawning else 0.0)
        st.alarm = st.closure_now >= need
        if st.alarm and not self._alarm_active:
            self.totals["alarms"] += 1
        self._alarm_active = st.alarm

    def _context(self, st, t):
        st.hours = (t - (self._drive_start if self._drive_start is not None else t)) / 3600.0 + self.hours_driven
        clock = self.start_clock + timedelta(seconds=t)
        st.circadian = circadian_risk(clock.hour + clock.minute / 60.0)
        if not self.context:
            return
        if st.hours >= config.BREAK_ADVICE_HOURS:
            st.advice = f"За рулём {st.hours:.1f} ч без перерыва — рекомендуется остановка"
        elif st.circadian >= 0.8:
            st.advice = "Ночное время: риск засыпания повышен"

    def _prune(self, t):
        for d in (self._jerks, self._hand_mouth):
            while d and d[0] < t - 10:
                d.popleft()

    def _fill_from_windows(self, st, t):
        short_from = t - config.SHORT_WINDOW_SEC
        long_from = t - config.LONG_WINDOW_SEC
        while self._frames and self._frames[0][0] < short_from:
            self._frames.popleft()
        while self._blinks and self._blinks[0][0] < short_from:
            self._blinks.popleft()
        while self._yawns and self._yawns[0] < long_from:
            self._yawns.popleft()
        while self._nods and self._nods[0] < long_from:
            self._nods.popleft()

        if self._frames:
            st.perclos = sum(c for _, c in self._frames) / len(self._frames)
        if self._blinks:
            st.blink_dur = float(np.mean([d for _, d in self._blinks]))
        span = min(config.SHORT_WINDOW_SEC, max(1e-6, t - (self._monitor_start or t)))
        st.blink_rate = len(self._blinks) * 60.0 / span if span >= 10 else 0.0
        st.yawns = len(self._yawns)
        st.nods = len(self._nods)

    def _explain(self, st, details):
        """Короткое объяснение решения — какие признаки повлияли."""
        if st.alarm:
            return f"глаза закрыты {st.closure_now:.1f} с"
        if (self._unknown_since is not None and st.eyes_hidden and
                st.t - self._unknown_since >= config.EYES_HIDDEN_WARN_SEC):
            return "глаза не видны"
        if details is None:
            return "накопление данных"
        parts = []
        mu = details["memberships"]
        if mu["perclos"]["high"] > 0.5 or mu["perclos"]["medium"] > 0.5 or st.perclos >= 0.15:
            parts.append(f"PERCLOS {st.perclos * 100:.0f}%")
        if mu["blink_dur"]["slow"] > 0.5 or mu["blink_dur"]["long"] > 0.5:
            parts.append(f"медленные моргания {st.blink_dur:.2f} с")
        if st.yawns >= 2:
            parts.append(f"зевки: {st.yawns}")
        if st.nods >= 1:
            parts.append(f"кивки: {st.nods}")
        if parts and self.context and (mu["circadian"]["high"] > 0.5 or mu["hours"]["long"] > 0.5):
            parts.append("ночь" if mu["circadian"]["high"] > 0.5 else "долгая езда")
        return ", ".join(parts)
