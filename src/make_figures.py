"""Draw the figures used by the 10-parameter algorithm report."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_yieldstar import load_overlay_points
from overlay_10par import PARAM_NAMES, fit_10par, predict

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "overlay_lot.csv"
OUT = ROOT / "report" / "figures"

font_manager.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

NAVY = "#1f3a5f"
TEAL = "#1f7a6b"
ORANGE = "#d06b2b"
RED = "#b23a48"
GOLD = "#c5922c"
INK = "#243040"
MUTED = "#5d6b7a"
PAPER = "#fbfaf7"


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", OUT / name)


def _fit(wafer: str, angle: str = "0"):
    points = load_overlay_points(CSV, wafer, angle)
    result = fit_10par(
        points.field_center_x,
        points.field_center_y,
        points.target_x,
        points.target_y,
        points.overlay_x,
        points.overlay_y,
    )
    return points, result


def figure_flow() -> None:
    fig, ax = plt.subplots(figsize=(8.2, 9.2), facecolor=PAPER)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")
    fig.patch.set_facecolor(PAPER)
    steps = [
        (11.0, "1  读取 OV_DATA", "每行是一个传感器角度下的一次量测"),
        (9.35, "2  只保留一片晶圆、角度 0°", "不把 0° 和 180° 做 (0+180)/2"),
        (7.70, "3  坐标分开取", "晶圆项用场中心 (Xc, Yc)\n场内项用标记相对场中心 (x, y)"),
        (5.85, "4  10 列设计矩阵，普通最小二乘", "3296 个点同时求解，不先按场平均"),
        (4.20, "5  得到 10 个系数", "T、WRot、WNonOrt、WExp、FRot、FMag、FAsRot、FAsMag"),
        (2.55, "6  残差", "每个点的实测减去模型\nRes = 样本标准差，分母 N−1"),
    ]
    for y, title, detail in steps:
        box = FancyBboxPatch(
            (1.15, y - 0.72),
            7.7,
            1.35,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            linewidth=1.2,
            edgecolor=NAVY,
            facecolor="white",
        )
        ax.add_patch(box)
        ax.text(5.0, y + 0.28, title, ha="center", va="center", fontsize=13, color=NAVY, fontweight="bold")
        ax.text(5.0, y - 0.28, detail, ha="center", va="center", fontsize=10, color=INK)
    for y0, y1 in ((10.28, 9.98), (8.63, 8.33), (6.98, 6.68), (5.13, 4.83), (3.48, 3.18)):
        ax.annotate(
            "",
            xy=(5.0, y1),
            xytext=(5.0, y0),
            arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.6),
        )
    ax.set_title("10 项计算流程", fontsize=16, color=INK, pad=8)
    _save(fig, "01_flow.png")


def figure_coordinates(points) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.6), facecolor=PAPER)
    fig.patch.set_facecolor(PAPER)
    ax = axes[0]
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    wafer = Circle((0, 0), 150, facecolor="#eef3f8", edgecolor=NAVY, lw=1.4)
    ax.add_patch(wafer)
    # notch down, orientation 0
    ax.plot([0, 0], [-150, -142], color=NAVY, lw=3, solid_capstyle="butt")
    ax.scatter(points.field_center_x, points.field_center_y, s=12, c=TEAL, zorder=3)
    # highlight one full field near center-left that we use in the right panel
    key = (points.field_col == -5) & (points.field_row == -2)
    xc = float(points.field_center_x[key][0])
    yc = float(points.field_center_y[key][0])
    ax.annotate(
        "",
        xy=(xc, yc),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.6),
    )
    ax.scatter([xc], [yc], s=40, c=ORANGE, zorder=4)
    ax.text(xc - 8, yc + 10, "示例场\n(Xc, Yc)", color=ORANGE, fontsize=9, ha="right")
    ax.text(6, -132, "notch", color=NAVY, fontsize=8)
    ax.set_xlim(-175, 175)
    ax.set_ylim(-175, 175)
    ax.set_xlabel("晶圆 X / mm")
    ax.set_ylabel("晶圆 Y / mm")
    ax.set_title("场中心：COL × 25.125，ROW × 15.98")
    ax.axhline(0, color="#d5dbe3", lw=0.6)
    ax.axvline(0, color="#d5dbe3", lw=0.6)

    ax = axes[1]
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.add_patch(Rectangle((-12.6025, -8.03), 25.205, 16.06, fill=False, ec=NAVY, lw=1.2, label="Image 1"))
    ax.add_patch(
        Rectangle(
            (-12.5625, -7.99),
            25.125,
            15.98,
            fill=False,
            ec=TEAL,
            lw=1.0,
            ls="--",
            label="网格场",
        )
    )
    tx = points.target_x[key]
    ty = points.target_y[key]
    ax.scatter(tx, ty, s=36, c=NAVY, zorder=3, label="20 个标记")
    ax.scatter([0], [0], marker="+", s=120, c=ORANGE, zorder=4, label="场中心")
    ax.scatter([tx.mean()], [ty.mean()], marker="x", s=80, c=RED, zorder=4, label="标记重心")
    # one intrafield vector
    j = int(np.argmax(tx + ty))
    ax.annotate(
        "",
        xy=(tx[j], ty[j]),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=1.4),
    )
    ax.text(tx[j] - 0.3, ty[j] + 0.55, "(x, y)", color=GOLD, fontsize=10)
    ax.set_xlim(-16, 18)
    ax.set_ylim(-11, 12)
    ax.set_xlabel("场内 x / mm")
    ax.set_ylabel("场内 y / mm")
    ax.set_title("标记坐标相对场中心，不是相对重心")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle("两套坐标不要加在一起再送进晶圆项", fontsize=15, color=INK)
    _save(fig, "02_coordinates.png")


def figure_tis() -> None:
    wafer = "NFP01255.22"
    results = {}
    counts = {}
    for angle in ("0", "180"):
        points, result = _fit(wafer, angle)
        results[angle] = result
        counts[angle] = len(points)
    p0 = load_overlay_points(CSV, wafer, "0")
    p1 = load_overlay_points(CSV, wafer, "180")
    avg = fit_10par(
        p0.field_center_x,
        p0.field_center_y,
        p0.target_x,
        p0.target_y,
        0.5 * (p0.overlay_x + p1.overlay_x),
        0.5 * (p0.overlay_y + p1.overlay_y),
    )
    labels = ["角度 0°\n（表中算法）", "角度 180°", "(0+180)/2", "CSV 第13337行"]
    tx = [
        results["0"].as_dict["T_X"],
        results["180"].as_dict["T_X"],
        avg.as_dict["T_X"],
        -0.113,
    ]
    colors = [TEAL, NAVY, ORANGE, RED]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), facecolor=PAPER, gridspec_kw={"width_ratios": [1.05, 1.2]})
    fig.patch.set_facecolor(PAPER)
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_facecolor(PAPER)
    ax.add_patch(FancyBboxPatch((0.6, 5.6), 8.6, 3.5, boxstyle="round,pad=0.04,rounding_size=0.1", fc="white", ec=NAVY, lw=1.2))
    ax.text(4.9, 8.45, "同一标记的相邻两行", ha="center", fontsize=12, color=NAVY, fontweight="bold")
    ax.text(
        4.9,
        7.15,
        "角度 0°    OV = (2.372, 1.153) nm\n角度 180°  OV = (2.318, 1.111) nm",
        ha="center",
        va="center",
        fontsize=11,
        color=INK,
        linespacing=1.6,
    )
    ax.text(4.9, 6.05, "TIS ≈ (0° − 180°) / 2，修正值才是二者平均", ha="center", fontsize=9, color=MUTED)
    ax.add_patch(FancyBboxPatch((0.6, 0.7), 8.6, 4.2, boxstyle="round,pad=0.04,rounding_size=0.1", fc="#f4fbf8", ec=TEAL, lw=1.2))
    ax.text(4.9, 4.15, "表中用的是角度 0° 的原始行", ha="center", fontsize=12, color=TEAL, fontweight="bold")
    ax.text(
        4.9,
        2.55,
        f"每片 {counts['0']} 点，等于总行数的一半\n"
        "RECIPE_SETTING_3 的角度列是 0\n"
        "RESULTS_USED = 3296",
        ha="center",
        va="center",
        fontsize=11,
        color=INK,
        linespacing=1.55,
    )

    ax = axes[1]
    ax.set_facecolor("white")
    ypos = np.arange(len(labels))[::-1]
    left = -0.125
    ax.barh(ypos, np.array(tx) - left, left=left, color=colors, height=0.62)
    ax.axvline(-0.113, color=RED, lw=0.8, ls=":")
    ax.set_yticks(ypos, labels, fontsize=9)
    ax.set_xlabel("T_X / nm")
    ax.set_xlim(-0.125, -0.078)
    ax.set_title("晶圆 22 的 T_X：只有角度 0° 对上表")
    for y, value in zip(ypos, tx):
        ax.text(value + 0.0008, y, f"{value:.3f}", va="center", ha="left", fontsize=9, color=INK)
    fig.suptitle("(0+180)/2 是另一组数，对不上第 13337 行", fontsize=14, color=INK)
    _save(fig, "03_tis_choice.png")


def _wafer_term_field(kind: str, xc, yc):
    dx = np.zeros_like(xc)
    dy = np.zeros_like(yc)
    if kind == "T_X":
        dx += 1
    elif kind == "T_Y":
        dy += 1
    elif kind == "WRot":
        dx += -yc
        dy += xc
    elif kind == "WNonOrt":
        dx += -yc
    elif kind == "WExp_X":
        dx += xc
    elif kind == "WExp_Y":
        dy += yc
    return dx, dy


def _field_term_field(kind: str, x, y):
    dx = np.zeros_like(x)
    dy = np.zeros_like(y)
    if kind == "FRot":
        dx += -y
        dy += x
    elif kind == "FMag":
        dx += x
        dy += y
    elif kind == "FAsRot":
        dx += -y
        dy += -x
    elif kind == "FAsMag":
        dx += x
        dy += -y
    return dx, dy


def figure_terms() -> None:
    fig, axes = plt.subplots(2, 5, figsize=(12.6, 6.4), facecolor=PAPER)
    fig.patch.set_facecolor(PAPER)
    radius = 150.0
    xs = np.linspace(-120, 120, 7)
    xx, yy = np.meshgrid(xs, xs)
    keep = xx**2 + yy**2 <= 125**2
    xc, yc = xx[keep], yy[keep]
    fx = np.linspace(-12, 12, 5)
    fy = np.linspace(-8, 8, 5)
    fxx, fyy = np.meshgrid(fx, fy)

    wafer_terms = [
        ("T_X", "平移 X", "dx += T_X"),
        ("T_Y", "平移 Y", "dy += T_Y"),
        ("WRot", "晶圆旋转", "dx −= R·Yc\ndy += R·Xc"),
        ("WNonOrt", "非正交", "dx −= N·Yc"),
        ("WExp_X", "膨胀 X", "dx += Ex·Xc"),
    ]
    # WExp_Y is the 6th wafer term; the bottom row is the four field terms plus WExp_Y
    # Layout: top 5 wafer-side, bottom starts with WExp_Y then 4 field terms.
    bottom = [
        ("WExp_Y", "膨胀 Y", "dy += Ey·Yc", "wafer"),
        ("FRot", "场旋转", "dx −= r·y\ndy += r·x", "field"),
        ("FMag", "场倍率", "dx += m·x\ndy += m·y", "field"),
        ("FAsRot", "非对称旋转", "dx −= ar·y\ndy −= ar·x", "field"),
        ("FAsMag", "非对称倍率", "dx += am·x\ndy −= am·y", "field"),
    ]
    for ax, (key, title, formula) in zip(axes[0], wafer_terms):
        dx, dy = _wafer_term_field(key, xc, yc)
        scale = np.max(np.hypot(dx, dy)) or 1.0
        ax.quiver(xc, yc, dx / scale, dy / scale, angles="xy", scale_units="xy", scale=0.055, color=NAVY, width=0.012)
        ax.add_patch(Circle((0, 0), 130, fill=False, ec="#c5d0dc", lw=0.8))
        ax.set_aspect("equal")
        ax.set_xlim(-160, 160)
        ax.set_ylim(-170, 170)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=11, color=NAVY, pad=4)
        ax.text(0.5, 0.06, formula, transform=ax.transAxes, ha="center", va="bottom", fontsize=8, color=MUTED)
        ax.set_facecolor("white")
    for ax, (key, title, formula, kind) in zip(axes[1], bottom):
        if kind == "wafer":
            dx, dy = _wafer_term_field(key, xc, yc)
            scale = np.max(np.hypot(dx, dy)) or 1.0
            ax.quiver(xc, yc, dx / scale, dy / scale, angles="xy", scale_units="xy", scale=0.055, color=NAVY, width=0.012)
            ax.add_patch(Circle((0, 0), 130, fill=False, ec="#c5d0dc", lw=0.8))
            ax.set_xlim(-160, 160)
            ax.set_ylim(-170, 170)
        else:
            dx, dy = _field_term_field(key, fxx, fyy)
            scale = np.max(np.hypot(dx, dy)) or 1.0
            ax.quiver(fxx, fyy, dx / scale, dy / scale, angles="xy", scale_units="xy", scale=0.18, color=TEAL, width=0.012)
            ax.add_patch(Rectangle((-13, -9), 26, 18, fill=False, ec="#c5d0dc", lw=0.8))
            ax.set_xlim(-20, 20)
            ax.set_ylim(-16, 16)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=11, color=TEAL if kind == "field" else NAVY, pad=4)
        ax.text(0.5, 0.06, formula, transform=ax.transAxes, ha="center", va="bottom", fontsize=8, color=MUTED)
        ax.set_facecolor("white")
    fig.suptitle("10 项各自把矢量场扭成的形状（幅度已归一化，只看方向）", fontsize=14, color=INK, y=0.98)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.88, bottom=0.04, hspace=0.42, wspace=0.16)
    _save(fig, "04_ten_terms.png")


def figure_why_joint(points, result) -> None:
    # absolute-coordinate field magnification, for the contrast bar
    xc = points.field_center_x
    yc = points.field_center_y
    x = points.target_x
    y = points.target_y
    X = xc + x
    Y = yc + y
    n = x.size
    matrix = np.zeros((2 * n, 10))
    matrix[0::2, 0] = 1
    matrix[0::2, 1] = X
    matrix[0::2, 2] = Y
    matrix[0::2, 3] = x
    matrix[0::2, 4] = y
    matrix[1::2, 5] = 1
    matrix[1::2, 6] = X
    matrix[1::2, 7] = Y
    matrix[1::2, 8] = x
    matrix[1::2, 9] = y
    obs = np.empty(2 * n)
    obs[0::2] = points.overlay_x
    obs[1::2] = points.overlay_y
    coef, *_ = np.linalg.lstsq(matrix, obs, rcond=None)
    # ax, by of absolute model: coef[3] is ax, coef[9] is by
    fmag_abs = 0.5 * (coef[3] + coef[9])

    # cascade: field-mean wafer fit, then field fit on residuals
    keys = {}
    for i in range(n):
        keys.setdefault((int(points.field_col[i]), int(points.field_row[i])), []).append(i)
    xc_f, yc_f, dx_f, dy_f = [], [], [], []
    for idxs in keys.values():
        idxs = np.array(idxs)
        xc_f.append(xc[idxs[0]])
        yc_f.append(yc[idxs[0]])
        dx_f.append(points.overlay_x[idxs].mean())
        dy_f.append(points.overlay_y[idxs].mean())
    xc_f, yc_f, dx_f, dy_f = map(np.array, (xc_f, yc_f, dx_f, dy_f))
    nf = len(xc_f)
    aw = np.zeros((2 * nf, 6))
    bw = np.zeros(2 * nf)
    aw[0::2, 0] = 1
    aw[0::2, 2] = -yc_f
    aw[0::2, 3] = -yc_f
    aw[0::2, 4] = xc_f
    aw[1::2, 1] = 1
    aw[1::2, 2] = xc_f
    aw[1::2, 5] = yc_f
    bw[0::2] = dx_f
    bw[1::2] = dy_f
    pw, *_ = np.linalg.lstsq(aw, bw, rcond=None)
    tx, ty, wrot, wnon, wex, wey = pw
    pred_x = tx + wex * xc - (wrot + wnon) * yc
    pred_y = ty + wrot * xc + wey * yc
    af = np.zeros((2 * n, 6))
    bf = np.zeros(2 * n)
    af[0::2, 0] = 1
    af[0::2, 2] = -y
    af[0::2, 3] = x
    af[0::2, 4] = -y
    af[0::2, 5] = x
    af[1::2, 1] = 1
    af[1::2, 2] = x
    af[1::2, 3] = y
    af[1::2, 4] = -x
    af[1::2, 5] = -y
    bf[0::2] = points.overlay_x - pred_x
    bf[1::2] = points.overlay_y - pred_y
    pf, *_ = np.linalg.lstsq(af, bf, rcond=None)
    fmag_cascade = pf[3]
    wexp_cascade = wex
    fmag_joint = result.as_dict["FMag"]
    wexp_joint = result.as_dict["WExp_X"]

    fig = plt.figure(figsize=(11.4, 5.4), facecolor=PAPER)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.15], height_ratios=[1, 1], hspace=0.45, wspace=0.28)
    ax = fig.add_subplot(grid[:, 0])
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.add_patch(Rectangle((-12.6, -8.0), 25.2, 16.0, fc="#f4f7fb", ec=NAVY, lw=1.2))
    ax.scatter([0], [0], marker="+", s=140, c=ORANGE, zorder=4)
    ax.scatter([-0.92], [1.39], marker="x", s=90, c=RED, zorder=4)
    ax.annotate("", xy=(-0.92, 1.39), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.3))
    ax.text(0.3, -1.15, "场中心\n晶圆项用这里", color=ORANGE, fontsize=9)
    ax.text(-0.7, 2.3, "20 个标记的重心\n约 (−0.92, +1.39) mm", color=RED, fontsize=9)
    ax.set_xlim(-16, 16)
    ax.set_ylim(-11, 11)
    ax.set_xlabel("场内 x / mm")
    ax.set_ylabel("场内 y / mm")
    ax.set_title("完整场的标记不关于场中心对称")

    sub = fig.add_subplot(grid[0, 1])
    sub2 = fig.add_subplot(grid[1, 1])
    groups = [
        (sub, "WExp_X / ppm", [wexp_joint, wexp_cascade], ["一次最小二乘", "先按场平均"], -0.001, (-0.0036, 0.0006)),
        (sub2, "FMag / ppm", [fmag_joint, fmag_cascade, fmag_abs], ["一次最小二乘", "先按场平均", "晶圆项用绝对坐标"], 0.027, (0.0252, 0.0302)),
    ]
    for axis, ylabel, values, names, csv_value, ylim in groups:
        axis.set_facecolor("white")
        colors = [TEAL, ORANGE, NAVY][: len(values)]
        xpos = np.arange(len(values))
        axis.bar(xpos, values, color=colors, width=0.68)
        axis.axhline(csv_value, color=RED, lw=1.0, ls="--")
        axis.set_xticks(xpos, names, fontsize=8)
        axis.set_ylabel(ylabel)
        axis.set_ylim(*ylim)
        for x_pos, value in zip(xpos, values):
            if value >= 0:
                axis.text(x_pos, value + 0.00015, f"{value:.4f}", ha="center", va="bottom", fontsize=7.5, color=INK)
            else:
                axis.text(x_pos, value - 0.00012, f"{value:.4f}", ha="center", va="top", fontsize=7.5, color=INK)
    sub.set_title("晶圆 22：红虚线是 CSV 圆整后的目标", fontsize=11)
    sub2.legend(
        handles=[
            plt.Line2D([0], [0], color=RED, ls="--", label="CSV 圆整值"),
        ],
        fontsize=8,
        frameon=False,
        loc="upper right",
    )
    fig.suptitle("场中心坐标 + 全部测点一次求解", fontsize=14, color=INK)
    _save(fig, "05_why_joint.png")
    return {
        "fmag_abs": fmag_abs,
        "fmag_cascade": fmag_cascade,
        "wexp_cascade": wexp_cascade,
        "fmag_joint": fmag_joint,
        "wexp_joint": wexp_joint,
    }


def figure_verify(fit22, fit23) -> None:
    csv22 = np.array([-0.113, -0.097, 0.001, 0.0, -0.001, -0.001, -0.005, 0.027, -0.032, 0.01, 0.844, 0.962])
    csv23 = np.array([-0.09, 0.051, 0.001, -0.003, -0.002, 0.0, -0.006, 0.01, -0.028, 0.018, 0.831, 0.882])
    raw22 = np.r_[fit22.coefficients, fit22.res_x, fit22.res_y]
    raw23 = np.r_[fit23.coefficients, fit23.res_x, fit23.res_y]
    labels = list(PARAM_NAMES) + ["Res_X", "Res_Y"]
    units = ["nm", "nm", "µrad", "µrad", "ppm", "ppm", "µrad", "ppm", "µrad", "ppm", "nm", "nm"]

    fig, ax = plt.subplots(figsize=(12.2, 6.2), facecolor=PAPER)
    fig.patch.set_facecolor(PAPER)
    ax.axis("off")
    cell = []
    colors = []
    for i, name in enumerate(labels):
        r22 = round(float(raw22[i]), 3)
        r23 = round(float(raw23[i]), 3)
        ok22 = abs(r22 - csv22[i]) < 5e-4
        ok23 = abs(r23 - csv23[i]) < 5e-4
        cell.append(
            [
                name,
                units[i],
                f"{raw22[i]:.6f}",
                f"{r22:.3f}",
                f"{csv22[i]:.3f}",
                f"{raw23[i]:.6f}",
                f"{r23:.3f}",
                f"{csv23[i]:.3f}",
                "一致" if ok22 and ok23 else "不一致",
            ]
        )
        colors.append(["#f4fbf8" if ok22 and ok23 else "#fdecea"] * 9)
    table = ax.table(
        cellText=cell,
        colLabels=["项", "单位", "晶圆22计算", "圆整", "CSV", "晶圆23计算", "圆整", "CSV", ""],
        cellColours=colors,
        colColours=["#1f3a5f"] * 9,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.45)
    for (row, col), cell_obj in table.get_celld().items():
        if row == 0:
            cell_obj.set_text_props(color="white", fontweight="bold")
            cell_obj.set_height(0.055)
        cell_obj.set_edgecolor("#e1e6ee")
    ax.set_title("第 13337、13338 行：计算值保留 3 位小数后与 CSV 一致", fontsize=14, color=INK, pad=12)
    _save(fig, "06_verify_table.png")


def figure_residual(points, result) -> None:
    # field-mean measured overlay and field-mean residual
    keys = {}
    for i in range(len(points)):
        keys.setdefault((int(points.field_col[i]), int(points.field_row[i])), []).append(i)
    xc, yc, mx, my, rx, ry = [], [], [], [], [], []
    for idxs in keys.values():
        idxs = np.array(idxs)
        xc.append(points.field_center_x[idxs[0]])
        yc.append(points.field_center_y[idxs[0]])
        mx.append(points.overlay_x[idxs].mean())
        my.append(points.overlay_y[idxs].mean())
        rx.append(result.residual_x[idxs].mean())
        ry.append(result.residual_y[idxs].mean())
    xc, yc, mx, my, rx, ry = map(np.array, (xc, yc, mx, my, rx, ry))
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.5), facecolor=PAPER)
    fig.patch.set_facecolor(PAPER)
    # model arrows at field centers
    model_x, model_y = predict(
        result.coefficients,
        xc,
        yc,
        np.zeros_like(xc),
        np.zeros_like(yc),
    )
    series = [
        (axes[0], mx, my, "各场实测平均", NAVY),
        (axes[1], model_x, model_y, "10 项在场中心的预测", TEAL),
        (axes[2], rx, ry, "减去模型后的场平均", ORANGE),
    ]
    for ax, dx, dy, title, color in series:
        ax.set_facecolor("white")
        ax.set_aspect("equal")
        ax.add_patch(Circle((0, 0), 150, fill=False, ec="#c5d0dc", lw=1.0))
        ax.quiver(xc, yc, dx, dy, angles="xy", scale_units="xy", scale=0.08, color=color, width=0.004)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("X / mm")
        ax.axhline(0, color="#e6ebf1", lw=0.5)
        ax.axvline(0, color="#e6ebf1", lw=0.5)
    axes[1].text(
        0.5,
        0.08,
        "模型最大约 0.3 nm\n同一比例下缩成小点",
        transform=axes[1].transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    axes[0].set_ylabel("Y / mm")
    fig.suptitle(
        f"晶圆 22，角度 0°。残差样本标准差  X {result.res_x:.3f} nm，Y {result.res_y:.3f} nm",
        fontsize=13,
        color=INK,
    )
    fig.text(0.5, 0.01, "箭头长度同一比例。模型只解释其中一小段系统性偏移，大部分留在残差里。", ha="center", fontsize=9, color=MUTED)
    _save(fig, "07_residual_map.png")


def main() -> None:
    points, fit22 = _fit("NFP01255.22", "0")
    _, fit23 = _fit("NFP01255.23", "0")
    figure_flow()
    figure_coordinates(points)
    figure_tis()
    figure_terms()
    extra = figure_why_joint(points, fit22)
    figure_verify(fit22, fit23)
    figure_residual(points, fit22)
    # worked example, first angle-0 point of wafer 22 in file order is not special;
    # use the documented site COL=-6, ROW=-3, target (12.4725, 5.856)
    mask = (
        (points.field_col == -6)
        & (points.field_row == -3)
        & np.isclose(points.target_x, 12.4725)
        & np.isclose(points.target_y, 5.856)
    )
    i = int(np.flatnonzero(mask)[0])
    mx, my = predict(
        fit22.coefficients,
        points.field_center_x[i : i + 1],
        points.field_center_y[i : i + 1],
        points.target_x[i : i + 1],
        points.target_y[i : i + 1],
    )
    print("example index", i)
    print("measured", points.overlay_x[i], points.overlay_y[i])
    print("modeled", float(mx[0]), float(my[0]))
    print("residual", points.overlay_x[i] - float(mx[0]), points.overlay_y[i] - float(my[0]))
    print("extra", {k: float(v) for k, v in extra.items()})
    print("coef22")
    for name, value in fit22.as_dict.items():
        print(f"  {name} {value:.8f}  round {round(value, 3)}")
    print("res22", fit22.res_x, fit22.res_y)


if __name__ == "__main__":
    main()
