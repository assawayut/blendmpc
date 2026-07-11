"""Render the quadruped learning-curve figure (light + dark) from results/."""

from __future__ import annotations

import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "docs", "assets")

THEMES = {
    "light": {
        "residual": "#2a78d6",
        "scratch": "#1baf7a",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "baseline": "#c3c2b7",
    },
    "dark": {
        "residual": "#3987e5",
        "scratch": "#199e70",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "baseline": "#383835",
    },
}


def load(mode):
    out = defaultdict(dict)
    with open(os.path.join(HERE, "results", f"{mode}.csv")) as f:
        for row in csv.reader(f):
            out[int(row[1])][int(row[2])] = float(np.mean([float(v) for v in row[3:]]))
    return out


def curve(data):
    steps = sorted(next(iter(data.values())).keys())
    mat = np.array([[data[s][t] for t in steps] for s in sorted(data)])
    return np.array(steps), mat.mean(0), mat.min(0), mat.max(0)


def render(theme, path):
    c = THEMES[theme]
    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=170)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    mpc = curve(load("mpc"))[1][0]
    oracle = curve(load("oracle"))[1][0]
    ax.axhline(mpc, color=c["ink2"], lw=1.6, ls=(0, (4, 3)), zorder=2)
    ax.axhline(oracle, color=c["muted"], lw=1.4, ls=(0, (1, 2)), zorder=2)

    for mode, label in [
        ("residual", "Residual SAC over trot MPC"),
    ]:
        steps, mean, lo, hi = curve(load(mode))
        ax.fill_between(steps, lo, hi, color=c[mode], alpha=0.18, lw=0, zorder=3)
        ax.plot(
            steps,
            mean,
            color=c[mode],
            lw=2,
            solid_capstyle="round",
            zorder=4,
            label=label,
        )

    ax.plot(
        [], [], color=c["ink2"], lw=1.6, ls=(0, (4, 3)), label="MPC (nominal model)"
    )
    ax.plot([], [], color=c["muted"], lw=1.4, ls=(0, (1, 2)), label="MPC (true model)")

    ax.set_xlim(0, None)
    ax.set_ylim(-28, -4)
    ax.grid(axis="y", color=c["grid"], lw=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(c["baseline"])
    ax.tick_params(colors=c["muted"], labelsize=9)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v else "0")
    )
    ax.set_xlabel("environment steps", color=c["ink2"], fontsize=10)
    ax.set_ylabel("episode return", color=c["ink2"], fontsize=10)

    ax.text(
        0,
        1.14,
        "Gait MPC trotting under an unmodeled overload",
        transform=ax.transAxes,
        color=c["ink"],
        fontsize=13,
        fontweight="bold",
        va="bottom",
    )
    ax.text(
        0,
        1.045,
        "Go2 trot in place, trunk mass ×3 vs the MPC's model · mean over 5 "
        "fixed eval episodes · band: min–max over 3 seeds",
        transform=ax.transAxes,
        color=c["ink2"],
        fontsize=9,
        va="bottom",
    )

    handles, labels = ax.get_legend_handles_labels()
    order = [
        labels.index(x)
        for x in (
            "Residual SAC over trot MPC",
            "MPC (nominal model)",
            "MPC (true model)",
        )
    ]
    leg = ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="lower right",
        bbox_to_anchor=(0.99, 0.30),
        frameon=False,
        fontsize=9.5,
    )
    for t in leg.get_texts():
        t.set_color(c["ink2"])

    os.makedirs(OUT, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    for theme in ("light", "dark"):
        render(theme, os.path.join(OUT, f"quadruped_trot_{theme}.png"))
