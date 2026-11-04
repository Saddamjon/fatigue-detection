"""
Графики по журналу сеанса — для главы 3 (экспериментальная часть).

    python plot_session.py logs/session_20261028_143000.csv
Сохраняет PNG рядом с CSV.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

SERIES = "#2a78d6"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GOOD, WARNING, CRITICAL = "#0ca30c", "#fab219", "#d03b3b"


def style(ax, ylabel):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)


def hline(ax, y, label, x_start):
    ax.axhline(y, color=INK2, linestyle="--", linewidth=1)
    ax.text(x_start, y, label, color=INK, fontsize=8, va="bottom", ha="left",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))


def main(path):
    df = pd.read_csv(path)
    mon = df[df.phase == "monitoring"].copy()
    if mon.empty:
        sys.exit("В журнале нет данных этапа наблюдения.")
    mon["m"] = mon.t / 60.0
    x_start = mon.m.min()
    face = mon[mon.face == 1]
    # Показатели окна выводим после периода накопления данных
    ready = mon[mon.t >= mon.t.min() + config.WARMUP_SEC]

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, facecolor="white",
                             gridspec_kw={"height_ratios": [1.2, 1, 1, 1.3]})

    # 1. EAR и пороги
    ax = axes[0]
    ax.plot(face.m, face.ear, color=SERIES, linewidth=1)
    if "ear_thr" in mon and mon.ear_thr.notna().any():
        # Личный порог меняется вместе с наклоном головы (норма глаз по позе)
        thr = mon[mon.ear_thr.notna()]
        ax.plot(thr.m, thr.ear_thr, color=INK2, linestyle="--", linewidth=1, drawstyle="steps-post")
        ax.text(x_start, float(thr.ear_thr.iloc[0]), "личный порог", color=INK, fontsize=8,
                va="bottom", ha="left", bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))
    hline(ax, config.BASELINE_EAR_THRESHOLD, f"фиксированный порог {config.BASELINE_EAR_THRESHOLD}", x_start)
    style(ax, "EAR")
    ax.set_title("Степень открытия глаз (EAR)", loc="left", color=INK, fontsize=11)

    # 2. PERCLOS
    ax = axes[1]
    ax.plot(ready.m, ready.perclos * 100, color=SERIES, linewidth=2)
    style(ax, "%")
    ax.set_title(f"PERCLOS — доля закрытых глаз за {config.SHORT_WINDOW_SEC} с",
                 loc="left", color=INK, fontsize=11)

    # 3. Длительность моргания
    ax = axes[2]
    ax.plot(ready.m, ready.blink_dur, color=SERIES, linewidth=2)
    style(ax, "с")
    ax.set_title("Средняя длительность моргания", loc="left", color=INK, fontsize=11)

    # 4. Индекс усталости и уровни
    ax = axes[3]
    ax.axhspan(0, config.LEVEL_TIRED, color=GOOD, alpha=0.08, linewidth=0)
    ax.axhspan(config.LEVEL_TIRED, config.LEVEL_DANGER, color=WARNING, alpha=0.12, linewidth=0)
    ax.axhspan(config.LEVEL_DANGER, 100, color=CRITICAL, alpha=0.08, linewidth=0)
    for y0, y1, name in [(0, config.LEVEL_TIRED, 0), (config.LEVEL_TIRED, config.LEVEL_DANGER, 1),
                         (config.LEVEL_DANGER, 100, 2)]:
        ax.text(x_start, y1 - 2, " " + config.LEVEL_NAMES[name],
                color=INK2, fontsize=8, va="top")
    ax.plot(mon.m, mon["index"], color=SERIES, linewidth=2, label="индекс (сглаженный)")
    alarms = mon[(mon.alarm == 1) & (mon.alarm.shift(fill_value=0) == 0)]
    if len(alarms):
        ax.scatter(alarms.m, np.full(len(alarms), 95), marker="v", s=60, color=CRITICAL,
                   edgecolors="white", linewidths=1.5, zorder=3, label="тревога (закрытие ≥ 2 с)")
    if "nod" in mon:
        nods = mon[mon.nod == 1]
        if len(nods):
            ax.scatter(nods.m, np.full(len(nods), 88), marker="o", s=40, color=WARNING,
                       edgecolors="white", linewidths=1.5, zorder=3, label="сонный кивок")
    base = mon[(mon.baseline_alarm == 1) & (mon.baseline_alarm.shift(fill_value=0) == 0)]
    if len(base):
        ax.scatter(base.m, np.full(len(base), 5), marker="^", s=50, color=INK2,
                   edgecolors="white", linewidths=1.5, zorder=3, label="пороговый метод")
    ax.set_ylim(0, 100)
    style(ax, "индекс")
    ax.set_title("Индекс усталости (нечёткий вывод)", loc="left", color=INK, fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.22), ncol=4, frameon=False, fontsize=9)
    ax.set_xlabel("время, мин", color=INK2)

    fig.tight_layout()
    out = path.rsplit(".", 1)[0] + ".png"
    fig.savefig(out, dpi=200, facecolor="white")
    print("График сохранён:", out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Использование: python plot_session.py logs/session_....csv")
    main(sys.argv[1])
