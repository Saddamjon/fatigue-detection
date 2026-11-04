"""
Пакетная обработка записей с телефона — материал для главы 3.

1. Положите видео в папку videos/
2. Заполните sessions.csv (пример — sessions_example.csv): одна строка на запись
3. Запустите:
       python batch_analyze.py                    # все записи
       python batch_analyze.py --profile Саддам   # с профилем (закрывать глаза не нужно)
       python batch_analyze.py --compare          # плюс прежний алгоритм (--no-adapt) для сравнения
       python batch_analyze.py --phone FatigueTester --profile Саддам --compare
                                                  # папка из приложения на телефоне (видео + sessions.csv)
Уже обработанные записи повторно не считаются (кроме --force).

Результаты в папке results/:
    summary.csv        — показатели каждого сеанса (открывается в Excel)
    by_condition.csv   — средние по условиям (утро, обед, бодрость, перед сном)
    summary.png        — графики
"""
import argparse
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

HERE = os.path.dirname(os.path.abspath(__file__))
CONDITION_ORDER = ["утро", "обед", "бодрость", "перед_сном"]
CONDITION_NAMES = {"утро": "Утро (после сна)", "обед": "После обеда",
                   "бодрость": "Обычная бодрость", "перед_сном": "Перед сном"}
SERIES, INK, INK2, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"


def read_sessions(path):
    if not os.path.exists(path):
        sys.exit(f"Нет файла {path}. Скопируйте sessions_example.csv в sessions.csv и заполните.")
    # «;» — так пишут приложение на телефоне и русский Excel; иначе определяем сами
    with open(path, encoding="utf-8-sig") as fh:
        header = fh.readline()
    sep = ";" if ";" in header else None
    df = pd.read_csv(path, sep=sep, engine="python", encoding="utf-8-sig", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(how="all")
    for col in ("file", "condition"):
        if col not in df.columns:
            sys.exit(f"В {path} нет обязательного столбца «{col}»")
    df = df[df["file"].notna() & (df["file"].str.strip() != "")]
    for col in ("kss_before", "kss_after", "calib_start", "sleep_hours"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].str.replace(",", "."), errors="coerce")
    return df


def run_main(video, log, extra, force):
    """Запуск main.py без окна. Вывод сохраняется рядом с журналом (.txt)."""
    if os.path.exists(log) and not force:
        return True
    cmd = [sys.executable, os.path.join(HERE, "main.py"), "--source", video,
           "--no-display", "--log", log] + extra
    print("   ", os.path.basename(log), "...", flush=True)
    res = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, encoding="utf-8",
                         errors="replace")
    with open(log[:-4] + ".txt", "w", encoding="utf-8") as fh:
        fh.write(res.stdout)
        fh.write(res.stderr)
    if res.returncode != 0 or not os.path.exists(log):
        print("    [!] ошибка обработки, подробности в", log[:-4] + ".txt")
        return False
    return True


def rising(series):
    s = series.fillna(0).astype(int)
    return int(((s == 1) & (s.shift(fill_value=0) == 0)).sum())


def metrics(log):
    """Показатели сеанса по журналу (после калибровки и накопления данных)."""
    d = pd.read_csv(log)
    out = {"длительность_мин": round((d.t.max() - d.t.min()) / 60, 1) if len(d) else 0,
           "лицо_найдено_%": round(100 * d.face.mean(), 0) if len(d) else 0}
    mon = d[d.phase == "monitoring"]
    if mon.empty:
        out["примечание"] = "калибровка не завершилась"
        return out
    mon = mon[mon.t >= mon.t.min() + config.WARMUP_SEC]
    known = mon[mon.eye_state.isin(["open", "closed"])]
    out.update({
        "наблюдение_мин": round((mon.t.max() - mon.t.min()) / 60, 1),
        "PERCLOS_%": round(100 * (known.eye_state == "closed").mean(), 1) if len(known) else np.nan,
        "моргание_с": round(mon.blink_dur[mon.blink_dur > 0].mean(), 3),
        "морганий_в_мин": round(mon.blink_rate[mon.blink_rate > 0].mean(), 1),
        "зевки": int(mon.yawns.diff().clip(lower=0).sum()),
        "кивки": int(mon.nod.sum()),
        "тревоги": rising(mon.alarm),
        "пороговый_метод_тревоги": rising(mon.baseline_alarm),
        "индекс_средний": round(mon["index"].mean(), 1),
        "индекс_макс": round(mon["index"].max(), 1),
        "норма_%": round(100 * (mon.level == 0).mean(), 1),
        "усталость_%": round(100 * (mon.level == 1).mean(), 1),
        "сонливость_%": round(100 * (mon.level == 2).mean(), 1),
        "смены_позы": int(mon.pose_change.sum()),
        "артефакты": int(mon.artifact.sum()),
    })
    txt = log[:-4] + ".txt"
    if os.path.exists(txt):
        for line in open(txt, encoding="utf-8", errors="replace"):
            if line.startswith("Калибровка:"):
                out["калибровка"] = line.split(":", 1)[1].strip()
            if "прикрыты сильнее" in line:
                out["предупреждение"] = line.strip("[!] \n")
    return out


def plot(summary, out_png):
    conds = [c for c in CONDITION_ORDER if c in set(summary.condition)] + \
            [c for c in summary.condition.unique() if c not in CONDITION_ORDER]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor="white")
    for ax, col, title in [(axes[0], "PERCLOS_%", "PERCLOS, % (среднее по условию)"),
                           (axes[1], "моргание_с", "Средняя длительность моргания, с")]:
        vals = [summary[summary.condition == c][col].mean() for c in conds]
        n = [int(summary[summary.condition == c][col].notna().sum()) for c in conds]
        bars = ax.bar(range(len(conds)), vals, color=SERIES, width=0.6)
        for b, v, k in zip(bars, vals, n):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}\nn={k}" if col == "моргание_с"
                        else f"{v:.1f}\nn={k}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([CONDITION_NAMES.get(c, c) for c in conds], fontsize=8, color=INK2)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        top = np.nanmax(vals) if len(vals) and not np.all(np.isnan(vals)) else 0
        ax.set_ylim(0, top * 1.25 if top > 0 else 1)
    ax = axes[2]
    if "kss" in summary and summary.kss.notna().any():
        s = summary[summary.kss.notna()]
        ax.scatter(s.kss, s["индекс_средний"], s=50, color=SERIES, edgecolors="white", linewidths=1.5)
        for _, r in s.iterrows():
            ax.annotate(CONDITION_NAMES.get(r.condition, r.condition).split()[0], (r.kss, r["индекс_средний"]),
                        fontsize=7, color=INK2, xytext=(4, 3), textcoords="offset points")
        ax.set_xlabel("самооценка сонливости KSS (1–9)", color=INK2, fontsize=9)
        ax.set_xlim(0.5, 9.5)
    ax.set_title("Индекс усталости и самооценка", loc="left", fontsize=10, color=INK)
    for a in axes:
        a.grid(axis="y", color=GRID, linewidth=0.8)
        a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a.spines[sp].set_color(GRID)
        a.tick_params(colors=INK2, labelsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, facecolor="white")


def main():
    ap = argparse.ArgumentParser(description="Пакетная обработка записей")
    ap.add_argument("--sessions", default="sessions.csv")
    ap.add_argument("--videos", default="videos", help="папка с видео")
    ap.add_argument("--profile", default=None, help="профиль водителя для всех записей")
    ap.add_argument("--compare", action="store_true", help="также прежний алгоритм (--no-adapt)")
    ap.add_argument("--force", action="store_true", help="пересчитать уже обработанные")
    ap.add_argument("--out", default="results")
    ap.add_argument("--phone", default=None,
                    help="папка, скопированная с телефона (в ней видео и sessions.csv)")
    args = ap.parse_args()
    os.chdir(HERE)
    if args.phone:
        if not os.path.isdir(args.phone):
            sys.exit(f"Нет папки {args.phone}. Скопируйте папку с телефона в {HERE}")
        args.videos = args.phone
        args.sessions = os.path.join(args.phone, "sessions.csv")

    sessions = read_sessions(args.sessions)
    logdir = os.path.join("logs", "batch")
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for _, r in sessions.iterrows():
        video = r["file"].strip()
        path = video if os.path.isabs(video) else os.path.join(args.videos, video)
        print(f"• {video} ({r['condition']})")
        if not os.path.exists(path):
            print("    [!] файл не найден:", path)
            continue
        stem = os.path.splitext(os.path.basename(video))[0]
        extra = []
        cs = r.get("calib_start")
        if cs is not None and not pd.isna(cs):
            extra += ["--calib-start", str(cs)]
        tm = r.get("time")
        if isinstance(tm, str) and ":" in tm:
            extra += ["--start-time", tm.strip()]
        if args.profile:
            extra += ["--profile", args.profile]
        row = {k: r.get(k) for k in sessions.columns}
        log = os.path.join(logdir, stem + ".csv")
        if run_main(path, log, extra, args.force):
            row.update(metrics(log))
        if args.compare:
            log2 = os.path.join(logdir, stem + "_noadapt.csv")
            if run_main(path, log2, extra + ["--no-adapt"], args.force):
                m2 = metrics(log2)
                for k in ("PERCLOS_%", "индекс_средний", "тревоги", "кивки", "сонливость_%"):
                    if k in m2:
                        row["прежн_" + k] = m2[k]
        rows.append(row)

    if not rows:
        sys.exit("Нет обработанных записей.")
    summary = pd.DataFrame(rows)
    kss_cols = [c for c in ("kss_before", "kss_after") if c in summary]
    if kss_cols:
        summary["kss"] = summary[kss_cols].mean(axis=1)

    # Excel с русскими настройками: разделитель «;» и десятичная запятая
    summary.to_csv(os.path.join(args.out, "summary.csv"), sep=";", decimal=",",
                   index=False, encoding="utf-8-sig")
    num = summary.select_dtypes("number").columns
    by_cond = summary.groupby("condition")[list(num)].mean().round(2)
    by_cond.insert(0, "сеансов", summary.groupby("condition").size())
    by_cond.to_csv(os.path.join(args.out, "by_condition.csv"), sep=";", decimal=",",
                   encoding="utf-8-sig")
    plot(summary, os.path.join(args.out, "summary.png"))

    print("\n=== Средние по условиям ===")
    cols = [c for c in ("сеансов", "kss", "PERCLOS_%", "моргание_с", "индекс_средний",
                        "сонливость_%", "тревоги", "прежн_PERCLOS_%", "прежн_тревоги") if c in by_cond]
    order = [c for c in CONDITION_ORDER if c in by_cond.index] + \
            [c for c in by_cond.index if c not in CONDITION_ORDER]
    print(by_cond.loc[order, cols].to_string())
    if "kss" in summary and summary.kss.notna().sum() >= 4:
        print("\nСвязь с самооценкой KSS (коэффициент Спирмена, от -1 до 1):")
        for c in ("индекс_средний", "PERCLOS_%", "моргание_с"):
            if c in summary:
                rho = summary[["kss", c]].dropna().corr(method="spearman").iloc[0, 1]
                print(f"  {c:<16} {rho:+.2f}")
    print(f"\nРезультаты: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
