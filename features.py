"""
Модуль извлечения признаков.

На каждом кадре нейросеть MediaPipe Face Landmarker находит 478 ключевых
точек лица, по которым вычисляются:
  * EAR  (Eye Aspect Ratio)   — степень открытия глаз;
  * MAR  (Mouth Aspect Ratio) — степень открытия рта (зевота);
  * углы поворота головы (pitch, yaw, roll) — наклоны и кивки;
  * оценки нейросети (blendshapes): моргание eyeBlink и прищур eyeSquint.

Дополнительно нейросеть MediaPipe Hand Landmarker проверяет, не закрывает ли
рука глаза или рот (протирание глаз, кашель, сморкание) — такие кадры
нельзя считать закрытием глаз или зевком.
"""
import math
import os
import urllib.request
from dataclasses import dataclass
from typing import Optional

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

import config

# Индексы точек сетки MediaPipe Face Mesh.
# Порядок для глаза: p1 (внешний угол), p2, p3 (верхнее веко),
# p4 (внутренний угол), p5, p6 (нижнее веко) — как в формуле Соукуповой и Чеха.
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
# Рот: три вертикальные пары и горизонталь (уголки внутреннего контура губ)
MOUTH_VERTICAL = [(82, 87), (13, 14), (312, 317)]
MOUTH_HORIZONTAL = (78, 308)
NOSE_TIP = 1
LOWER_LIP = 17


@dataclass
class FrameFeatures:
    """Признаки одного кадра."""
    face_found: bool
    ear: float = 0.0          # среднее по двум глазам
    ear_left: float = 0.0
    ear_right: float = 0.0
    mar: float = 0.0
    pitch: float = 0.0        # наклон вперёд/назад, градусы
    yaw: float = 0.0          # поворот влево/вправо
    roll: float = 0.0         # наклон к плечу
    # Оценка моргания от нейросети (blendshape eyeBlink): 0 — открыт, 1 — закрыт.
    # Модель обучена на множестве лиц и меньше зависит от формы глаза, чем EAR.
    blink: float = 0.0        # среднее по двум глазам
    blink_left: float = 0.0
    blink_right: float = 0.0
    # Прищур (blendshape eyeSquint): поднимается нижнее веко. При прищуре EAR
    # падает, хотя глаз открыт, — его нельзя путать с закрытием.
    squint: float = 0.0
    # Рука у лица (если включена проверка руки)
    hand_checked: bool = False
    hand_on_eyes: bool = False
    hand_on_mouth: bool = False
    points: Optional[np.ndarray] = None   # точки в пикселях (для отрисовки)
    hand_points: Optional[list] = None    # точки рук в пикселях (для отрисовки)


def _dist(a, b):
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(p):
    """EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)."""
    return (_dist(p[1], p[5]) + _dist(p[2], p[4])) / (2.0 * _dist(p[0], p[3]) + 1e-6)


def mouth_aspect_ratio(pts):
    """MAR = среднее вертикальное раскрытие / ширина рта."""
    v = np.mean([_dist(pts[a], pts[b]) for a, b in MOUTH_VERTICAL])
    h = _dist(pts[MOUTH_HORIZONTAL[0]], pts[MOUTH_HORIZONTAL[1]])
    return v / (h + 1e-6)


def rotation_to_euler(m):
    """Углы Эйлера (градусы) из матрицы преобразования лица 4x4."""
    r = np.asarray(m)[:3, :3]
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, -r[2, 0]))))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return pitch, yaw, roll


def eye_box(pts, eye_idx):
    """Область одного глаза: от брови до верха щеки, чуть шире самого глаза.
    Размер считается от ширины глаза — он не меняется при моргании."""
    c0, c1 = pts[eye_idx[0]], pts[eye_idx[3]]          # уголки глаза
    w = float(np.linalg.norm(c1 - c0))
    cx, cy = (c0 + c1) / 2
    return (cx - 0.65 * w, cy - 0.55 * w, cx + 0.65 * w, cy + 0.45 * w)


def mouth_nose_box(pts):
    """Область рта и носа (кашель, сморкание): от кончика носа до нижней губы."""
    c0, c1 = pts[MOUTH_HORIZONTAL[0]], pts[MOUTH_HORIZONTAL[1]]
    w = float(np.linalg.norm(c1 - c0))
    x0, x1 = min(c0[0], c1[0]) - 0.1 * w, max(c0[0], c1[0]) + 0.1 * w
    y0 = pts[NOSE_TIP][1] - 0.2 * w
    y1 = pts[LOWER_LIP][1] + 0.3 * w
    return (x0, y0, x1, y1)


def points_in_box(points, box):
    x0, y0, x1, y1 = box
    return int(np.sum((points[:, 0] >= x0) & (points[:, 0] <= x1) &
                      (points[:, 1] >= y0) & (points[:, 1] <= y1)))


def hand_occlusion(face_pts, hands):
    """
    Закрывает ли рука глаза или рот. Считаем точки руки внутри области:
    положение «щека на ладони» (водитель облокотился) не закрывает глаза
    и не должно считаться помехой.
    """
    if face_pts is None or not hands:
        return False, False
    boxes = [eye_box(face_pts, RIGHT_EYE), eye_box(face_pts, LEFT_EYE)]
    mbox = mouth_nose_box(face_pts)
    on_eyes = any(points_in_box(h, b) >= config.HAND_POINTS_MIN for h in hands for b in boxes)
    on_mouth = any(points_in_box(h, mbox) >= config.HAND_POINTS_MIN for h in hands)
    return on_eyes, on_mouth


class ModelUnavailable(RuntimeError):
    """Файла модели нет, и скачать его не удалось (нет интернета)."""


def ensure_model(path, url):
    """
    Возвращает путь к модели. Интернет нужен только один раз — если файла
    ещё нет. Сам анализ видео работает без интернета.
    """
    if os.path.exists(path):
        return path
    print(f"Скачиваю модель {os.path.basename(path)} ...")
    tmp = path + ".part"
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)       # недокачанный файл не остаётся под настоящим именем
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise ModelUnavailable(
            f"Нет файла модели {os.path.basename(path)}, и скачать его не удалось ({e}).\n"
            f"Скопируйте {os.path.basename(path)} в папку программы с компьютера, "
            f"где программа уже запускалась, или подключитесь к интернету один раз.") from None
    print("Модель сохранена:", path)
    return path


class FaceAnalyzer:
    """Обёртка над MediaPipe Face Landmarker (и Hand Landmarker) в режиме видеопотока."""

    def __init__(self, model_path=config.MODEL_PATH, hand_check=config.HAND_CHECK,
                 hand_model_path=config.HAND_MODEL_PATH):
        options = vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(
                model_asset_path=ensure_model(model_path, config.MODEL_URL)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts = -1

        self.hands = None
        if hand_check:
            try:
                hopt = vision.HandLandmarkerOptions(
                    base_options=mp_tasks.BaseOptions(
                        model_asset_path=ensure_model(hand_model_path, config.HAND_MODEL_URL)),
                    running_mode=vision.RunningMode.VIDEO,
                    num_hands=2,
                    min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self.hands = vision.HandLandmarker.create_from_options(hopt)
            except Exception as e:      # без модели руки программа всё равно работает
                print("Проверка руки у лица отключена:", e)
                self.hands = None
        self._frame_no = 0
        self._last_hands = []
        self._hand_ts = -1

    def process(self, frame_bgr, timestamp_ms: int) -> FrameFeatures:
        # Временные метки для MediaPipe должны строго возрастать
        timestamp_ms = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = timestamp_ms

        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(image, timestamp_ms)

        # Рука проверяется не на каждом кадре (через HAND_CHECK_EVERY), между
        # проверками используется последний результат
        if self.hands is not None and self._frame_no % config.HAND_CHECK_EVERY == 0:
            self._hand_ts = max(timestamp_ms, self._hand_ts + 1)
            hres = self.hands.detect_for_video(image, self._hand_ts)
            self._last_hands = [np.array([[p.x * w, p.y * h] for p in hand], dtype=np.float32)
                                for hand in hres.hand_landmarks]
        self._frame_no += 1

        if not result.face_landmarks:
            return FrameFeatures(face_found=False, hand_checked=self.hands is not None,
                                 hand_points=self._last_hands or None)

        lm = result.face_landmarks[0]
        # Переводим нормированные координаты в пиксели,
        # чтобы пропорции глаза не искажались из-за формата кадра
        pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)

        ear_r = eye_aspect_ratio(pts[RIGHT_EYE])
        ear_l = eye_aspect_ratio(pts[LEFT_EYE])
        mar = mouth_aspect_ratio(pts)

        pitch = yaw = roll = 0.0
        if result.facial_transformation_matrixes:
            pitch, yaw, roll = rotation_to_euler(result.facial_transformation_matrixes[0])

        blink_l = blink_r = squint = 0.0
        if result.face_blendshapes:
            scores = {c.category_name: c.score for c in result.face_blendshapes[0]}
            blink_l = float(scores.get("eyeBlinkLeft", 0.0))
            blink_r = float(scores.get("eyeBlinkRight", 0.0))
            squint = (float(scores.get("eyeSquintLeft", 0.0)) +
                      float(scores.get("eyeSquintRight", 0.0))) / 2.0

        on_eyes, on_mouth = hand_occlusion(pts, self._last_hands)

        return FrameFeatures(
            face_found=True,
            ear=(ear_l + ear_r) / 2.0, ear_left=ear_l, ear_right=ear_r,
            mar=mar, pitch=pitch, yaw=yaw, roll=roll,
            blink=(blink_l + blink_r) / 2.0, blink_left=blink_l, blink_right=blink_r,
            squint=squint,
            hand_checked=self.hands is not None, hand_on_eyes=on_eyes, hand_on_mouth=on_mouth,
            points=pts, hand_points=self._last_hands or None,
        )

    def close(self):
        self.landmarker.close()
        if self.hands is not None:
            self.hands.close()
