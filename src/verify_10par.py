"""Check the 10-parameter fit against the YieldStar lot report."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_yieldstar import load_overlay_points, wafers_in
from overlay_10par import fit_10par

CSV = Path(__file__).resolve().parents[1] / "data" / "overlay_lot.csv"

# Rounded values copied from the lot report. Angle 0, lines 13337-13339.
# Angle 180 is the cross-check on lines 13369-13371.
EXPECTED = {
    ("NFP01255.22", "0"): {
        "coef": [-0.113, -0.097, 0.001, 0.0, -0.001, -0.001, -0.005, 0.027, -0.032, 0.01],
        "res": (0.844, 0.962),
        "n": 3296,
    },
    ("NFP01255.23", "0"): {
        "coef": [-0.09, 0.051, 0.001, -0.003, -0.002, 0.0, -0.006, 0.01, -0.028, 0.018],
        "res": (0.831, 0.882),
        "n": 3296,
    },
    ("NFP01255.22", "180"): {
        "coef": [-0.095, -0.063, 0.001, 0.0, 0.0, -0.001, -0.005, 0.027, -0.032, 0.011],
        "res": (0.839, 0.963),
        "n": 3296,
    },
    ("NFP01255.23", "180"): {
        "coef": [-0.11, 0.093, 0.001, -0.003, -0.001, 0.0, -0.006, 0.009, -0.028, 0.018],
        "res": (0.829, 0.883),
        "n": 3296,
    },
}

# Lot row: coefficients of the stacked fit, residual of the pointwise wafer average.
EXPECTED_LOT = {
    "0": {
        "coef": [-0.102, -0.023, 0.001, -0.002, -0.001, -0.001, -0.005, 0.018, -0.03, 0.014],
        "res": (0.823, 0.907),
    },
    "180": {
        "coef": [-0.103, 0.015, 0.001, -0.002, -0.001, -0.001, -0.006, 0.018, -0.03, 0.014],
        "res": (0.82, 0.907),
    },
}


def fit_points(points):
    return fit_10par(
        points.field_center_x,
        points.field_center_y,
        points.target_x,
        points.target_y,
        points.overlay_x,
        points.overlay_y,
    )


def main() -> int:
    failures = 0
    wafers = wafers_in(CSV)
    print(f"csv: {CSV}")
    print(f"wafers: {wafers}")
    for wafer in wafers:
        for angle in ("0", "180"):
            points = load_overlay_points(CSV, wafer, angle)
            result = fit_points(points)
            expected = EXPECTED[(wafer, angle)]
            rounded = np.round(result.coefficients, 3)
            coef_ok = np.allclose(rounded, expected["coef"], atol=1e-9)
            res_ok = (
                round(result.res_x, 3) == expected["res"][0]
                and round(result.res_y, 3) == expected["res"][1]
            )
            count_ok = len(points) == expected["n"]
            status = "PASS" if coef_ok and res_ok and count_ok else "FAIL"
            if status == "FAIL":
                failures += 1
            print(
                f"{status} {wafer} angle {angle}  n={len(points)}  "
                f"Res=({result.res_x:.6f}, {result.res_y:.6f})"
            )
            if not coef_ok or not res_ok or not count_ok:
                print("  got   ", rounded, round(result.res_x, 3), round(result.res_y, 3))
                print("  expect", expected["coef"], expected["res"])

    for angle, expected in EXPECTED_LOT.items():
        stacked = load_overlay_points(CSV, None, angle)
        stacked_fit = fit_points(stacked)
        per_wafer = [load_overlay_points(CSV, wafer, angle) for wafer in wafers]
        average_x = np.mean([item.overlay_x for item in per_wafer], axis=0)
        average_y = np.mean([item.overlay_y for item in per_wafer], axis=0)
        first = per_wafer[0]
        average_fit = fit_10par(
            first.field_center_x,
            first.field_center_y,
            first.target_x,
            first.target_y,
            average_x,
            average_y,
        )
        coef_ok = np.allclose(np.round(stacked_fit.coefficients, 3), expected["coef"], atol=1e-9)
        res_ok = (
            round(average_fit.res_x, 3) == expected["res"][0]
            and round(average_fit.res_y, 3) == expected["res"][1]
        )
        same_coef = np.allclose(stacked_fit.coefficients, average_fit.coefficients, atol=1e-9)
        status = "PASS" if coef_ok and res_ok and same_coef else "FAIL"
        if status == "FAIL":
            failures += 1
        print(
            f"{status} LOT angle {angle}  "
            f"avg-res=({average_fit.res_x:.6f}, {average_fit.res_y:.6f})  "
            f"stacked-res=({stacked_fit.res_x:.6f}, {stacked_fit.res_y:.6f})"
        )
    print("failures", failures)
    return failures


if __name__ == "__main__":
    sys.exit(main())
