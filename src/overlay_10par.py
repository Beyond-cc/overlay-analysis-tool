"""YieldStar 10-parameter overlay model.

The fit reproduces the ``10par`` rows in an ASML YieldStar lot report:
one ordinary least-squares solve per wafer, using sensor angle 0 degrees
and field-center coordinates for the wafer terms.

Overlay is in nm. Field center and intrafield coordinates are in mm.
With those units the scale coefficients are ppm and the rotation /
non-orthogonality coefficients are µrad, with no extra conversion factor
(1 ppm * 1 mm = 1 nm, and 1 µrad * 1 mm = 1 nm).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PARAM_NAMES = (
    "T_X",
    "T_Y",
    "WRot",
    "WNonOrt",
    "WExp_X",
    "WExp_Y",
    "FRot",
    "FMag",
    "FAsRot",
    "FAsMag",
)

PARAM_UNITS = {
    "T_X": "nm",
    "T_Y": "nm",
    "WRot": "urad",
    "WNonOrt": "urad",
    "WExp_X": "ppm",
    "WExp_Y": "ppm",
    "FRot": "urad",
    "FMag": "ppm",
    "FAsRot": "urad",
    "FAsMag": "ppm",
}


@dataclass(frozen=True)
class FitResult:
    """Coefficients are in ``PARAM_NAMES`` order."""

    coefficients: np.ndarray
    residual_x: np.ndarray
    residual_y: np.ndarray

    @property
    def as_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in zip(PARAM_NAMES, self.coefficients)}

    @property
    def res_x(self) -> float:
        """Sample standard deviation of the X residual, divide by N - 1."""
        return float(self.residual_x.std(ddof=1))

    @property
    def res_y(self) -> float:
        return float(self.residual_y.std(ddof=1))


def design_matrix(
    field_center_x: np.ndarray,
    field_center_y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
) -> np.ndarray:
    """Build the 2N x 10 design matrix.

    Row 2i is the X equation of point i, row 2i+1 is the Y equation.

    Wafer terms are evaluated at the field center ``(Xc, Yc)``. Intrafield
    terms are evaluated at the target position ``(x, y)`` relative to that
    center. The absolute wafer coordinate of the mark, ``(Xc + x, Yc + y)``,
    is not used.
    """
    xc = np.asarray(field_center_x, dtype=float)
    yc = np.asarray(field_center_y, dtype=float)
    x = np.asarray(target_x, dtype=float)
    y = np.asarray(target_y, dtype=float)
    if not (xc.shape == yc.shape == x.shape == y.shape):
        raise ValueError("coordinate arrays must have the same shape")
    n = xc.size
    matrix = np.zeros((2 * n, 10), dtype=float)
    # dx = Tx + WExpX*Xc - (WRot + WNonOrt)*Yc
    #    + (FMag + FAsMag)*x - (FRot + FAsRot)*y
    matrix[0::2, 0] = 1.0
    matrix[0::2, 2] = -yc
    matrix[0::2, 3] = -yc
    matrix[0::2, 4] = xc
    matrix[0::2, 6] = -y
    matrix[0::2, 7] = x
    matrix[0::2, 8] = -y
    matrix[0::2, 9] = x
    # dy = Ty + WRot*Xc + WExpY*Yc
    #    + (FRot - FAsRot)*x + (FMag - FAsMag)*y
    matrix[1::2, 1] = 1.0
    matrix[1::2, 2] = xc
    matrix[1::2, 5] = yc
    matrix[1::2, 6] = x
    matrix[1::2, 7] = y
    matrix[1::2, 8] = -x
    matrix[1::2, 9] = -y
    return matrix


def predict(
    coefficients: np.ndarray,
    field_center_x: np.ndarray,
    field_center_y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = design_matrix(field_center_x, field_center_y, target_x, target_y)
    modeled = matrix @ np.asarray(coefficients, dtype=float)
    return modeled[0::2], modeled[1::2]


def fit_10par(
    field_center_x: np.ndarray,
    field_center_y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    overlay_x: np.ndarray,
    overlay_y: np.ndarray,
) -> FitResult:
    """Unweighted linear least squares of the 10-parameter model.

    Every measurement point is one observation. Points are not averaged by
    field or by intrafield site before the solve.
    """
    overlay_x = np.asarray(overlay_x, dtype=float)
    overlay_y = np.asarray(overlay_y, dtype=float)
    matrix = design_matrix(field_center_x, field_center_y, target_x, target_y)
    observation = np.empty(2 * overlay_x.size, dtype=float)
    observation[0::2] = overlay_x
    observation[1::2] = overlay_y
    coefficients, _, _, _ = np.linalg.lstsq(matrix, observation, rcond=None)
    modeled_x, modeled_y = predict(
        coefficients, field_center_x, field_center_y, target_x, target_y
    )
    return FitResult(
        coefficients=coefficients,
        residual_x=overlay_x - modeled_x,
        residual_y=overlay_y - modeled_y,
    )
