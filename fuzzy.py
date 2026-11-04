"""
Нечёткий логический вывод (схема Мамдани) для оценки уровня усталости.

Входы (лингвистические переменные):
  perclos    — доля времени с закрытыми глазами в окне, 0..1
  blink_dur  — средняя длительность моргания, секунды
  yawns      — число зевков за длинное окно (5 мин)
  nods       — число кивков головой за длинное окно (5 мин)
  hours      — время за рулём без перерыва, часы (контекст)
  circadian  — риск сонливости по времени суток, 0..1 (контекст)
Выход:
  fatigue    — индекс усталости 0..100

Реализовано без сторонних библиотек, чтобы каждый шаг
(фаззификация, правила, агрегация, дефаззификация) был виден и
его можно было описать в главе 2.
"""
import numpy as np


def trap(x, a, b, c, d):
    """Трапециевидная функция принадлежности (a <= b <= c <= d)."""
    x = np.asarray(x, dtype=float)
    left = np.ones_like(x) if b == a else np.clip((x - a) / (b - a), 0, 1)
    right = np.ones_like(x) if d == c else np.clip((d - x) / (d - c), 0, 1)
    return np.minimum(left, right) * ((x >= a) & (x <= d))


def tri(x, a, b, c):
    return trap(x, a, b, b, c)


BIG = 1e9  # «бесконечность» для открытых справа трапеций

# --- Функции принадлежности входов --------------------------------------
INPUTS = {
    "perclos": {
        "low":    lambda x: trap(x, -1, 0, 0.05, 0.10),
        "medium": lambda x: tri(x, 0.05, 0.12, 0.20),
        "high":   lambda x: trap(x, 0.15, 0.25, BIG, BIG),
    },
    "blink_dur": {
        "normal": lambda x: trap(x, -1, 0, 0.20, 0.30),
        "slow":   lambda x: tri(x, 0.20, 0.32, 0.45),
        "long":   lambda x: trap(x, 0.35, 0.50, BIG, BIG),
    },
    "yawns": {
        "none":     lambda x: trap(x, -1, 0, 0, 1),
        "some":     lambda x: tri(x, 0, 1.5, 3),
        "frequent": lambda x: trap(x, 2, 3, BIG, BIG),
    },
    "nods": {
        "none":     lambda x: trap(x, -1, 0, 0, 1),
        "some":     lambda x: tri(x, 0, 1, 2),
        "frequent": lambda x: trap(x, 1, 2, BIG, BIG),
    },
    # Контекст. У этих переменных нет правил, которые срабатывают сами по
    # себе: они только усиливают признаки, полученные с лица.
    "hours": {
        "short": lambda x: trap(x, -1, 0, 2, 4),
        "long":  lambda x: trap(x, 2, 4, BIG, BIG),
    },
    "circadian": {
        "low":  lambda x: trap(x, -1, 0, 0.6, 0.9),
        "high": lambda x: trap(x, 0.6, 0.9, BIG, BIG),
    },
}

# --- Функции принадлежности выхода --------------------------------------
UNIVERSE = np.linspace(0, 100, 201)
OUTPUT = {
    "normal": trap(UNIVERSE, 0, 0, 25, 45),
    "tired":  trap(UNIVERSE, 30, 45, 55, 70),
    "danger": trap(UNIVERSE, 55, 75, 100, 100),
}

# --- База правил --------------------------------------------------------
# Каждое правило: (условие, вывод). Условие — список пар (переменная, терм),
# объединённых через И (минимум). Несколько правил с одним выводом
# объединяются через ИЛИ (максимум).
RULES = [
    ([("perclos", "low"), ("blink_dur", "normal"), ("yawns", "none"), ("nods", "none")], "normal"),
    ([("perclos", "low"), ("blink_dur", "normal"), ("yawns", "some"), ("nods", "none")], "normal"),
    ([("perclos", "medium")], "tired"),
    ([("blink_dur", "slow")], "tired"),
    ([("blink_dur", "long")], "tired"),
    ([("yawns", "frequent")], "tired"),
    ([("yawns", "some"), ("blink_dur", "slow")], "tired"),
    ([("nods", "some")], "tired"),
    ([("perclos", "high")], "danger"),
    ([("perclos", "medium"), ("blink_dur", "long")], "danger"),
    ([("perclos", "medium"), ("yawns", "frequent")], "danger"),
    ([("nods", "frequent")], "danger"),
    # Контекстные правила: те же признаки усталости ночью или после долгой
    # езды без перерыва оцениваются как более опасные
    ([("perclos", "medium"), ("circadian", "high")], "danger"),
    ([("blink_dur", "slow"), ("circadian", "high")], "danger"),
    ([("perclos", "medium"), ("hours", "long")], "danger"),
    ([("blink_dur", "slow"), ("hours", "long")], "danger"),
    ([("nods", "some"), ("circadian", "high")], "danger"),
]


def fuzzify(values):
    """Степени принадлежности каждого входа каждому терму.
    Не переданные контекстные входы считаются нулевыми (контекст выключен)."""
    return {var: {term: float(fn(values.get(var, 0.0))) for term, fn in terms.items()}
            for var, terms in INPUTS.items()}


def infer(values, return_details=False):
    """
    values: dict с ключами perclos, blink_dur, yawns, nods
            (и необязательными hours, circadian).
    Возвращает индекс усталости 0..100 (центр тяжести).
    """
    mu = fuzzify(values)
    strength = {name: 0.0 for name in OUTPUT}
    fired = []
    for cond, out in RULES:
        s = min(mu[var][term] for var, term in cond)      # И = минимум
        strength[out] = max(strength[out], s)              # ИЛИ = максимум
        if s > 0:
            fired.append((cond, out, s))

    # Приоритет более опасного вывода: если одновременно сработали правила
    # «устал» и «опасно», доверие к менее тяжёлому выводу снижается.
    # Без этого центр тяжести «усредняет» состояния и при явной сонливости
    # индекс застревает около 70 (между двумя термами).
    strength["tired"] = min(strength["tired"], 1.0 - strength["danger"])
    strength["normal"] = min(strength["normal"],
                             1.0 - max(strength["tired"], strength["danger"]))

    # Импликация (усечение) и агрегация
    agg = np.zeros_like(UNIVERSE)
    for name, s in strength.items():
        agg = np.maximum(agg, np.minimum(OUTPUT[name], s))

    if agg.sum() < 1e-9:
        index = 0.0
    else:
        index = float((UNIVERSE * agg).sum() / agg.sum())  # центр тяжести

    if return_details:
        return index, {"memberships": mu, "rule_strengths": strength, "fired": fired}
    return index
