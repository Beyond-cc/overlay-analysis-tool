"""Read the overlay rows of a YieldStar lot CSV."""

from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OverlayPoints:
    wafer_id: str
    angle: str
    slot: np.ndarray
    field_col: np.ndarray
    field_row: np.ndarray
    field_center_x: np.ndarray
    field_center_y: np.ndarray
    target_x: np.ndarray
    target_y: np.ndarray
    overlay_x: np.ndarray
    overlay_y: np.ndarray

    def __len__(self) -> int:
        return int(self.overlay_x.size)


def load_overlay_points(path: str, wafer_id: str | None, angle: str) -> OverlayPoints:
    """Load one sensor angle.

    ``angle`` is the ``SENSOR_TO_TARGET_ANGLE`` value, ``"0"`` or ``"180"``.
    Pass ``wafer_id=None`` to concatenate every wafer. Rows stay in file order,
    wafer by wafer, so two wafers with the same sampling line up.
    """
    with open(path, newline="") as handle:
        lines = handle.readlines()

    header_index = next(i for i, line in enumerate(lines) if line.startswith("SLOT,WAFERID,MEASUREMENT_ID"))
    columns = next(csv.reader([lines[header_index].rstrip("\n")]))
    index = {name: position for position, name in enumerate(columns)}

    selected = []
    for line in lines[header_index + 1 :]:
        if line.startswith("["):
            break
        if not line.strip():
            continue
        row = next(csv.reader([line.rstrip("\n")]))
        if wafer_id is not None and row[index["WAFERID"]] != wafer_id:
            continue
        if row[index["SENSOR_TO_TARGET_ANGLE[degrees]"]] != str(angle):
            continue
        if row[index["VALID_X"]] != "True" or row[index["VALID_Y"]] != "True":
            continue
        selected.append(row)

    if not selected:
        raise ValueError(f"no rows for wafer={wafer_id!r} angle={angle!r}")

    def column(name: str, caster):
        return np.array([caster(row[index[name]]) for row in selected])

    return OverlayPoints(
        wafer_id=wafer_id or "ALL",
        angle=str(angle),
        slot=column("SLOT", str),
        field_col=column("COL", int),
        field_row=column("ROW", int),
        field_center_x=column("FIELDCENTER_X[mm]", float),
        field_center_y=column("FIELDCENTER_Y[mm]", float),
        target_x=column("TARGETPOS_X[mm]", float),
        target_y=column("TARGETPOS_Y[mm]", float),
        overlay_x=column("OV_X[nm]", float),
        overlay_y=column("OV_Y[nm]", float),
    )


def wafers_in(path: str) -> list[str]:
    with open(path, newline="") as handle:
        lines = handle.readlines()
    header_index = next(i for i, line in enumerate(lines) if line.startswith("SLOT,WAFERID,MEASUREMENT_ID"))
    columns = next(csv.reader([lines[header_index].rstrip("\n")]))
    wafer_col = columns.index("WAFERID")
    found: list[str] = []
    for line in lines[header_index + 1 :]:
        if line.startswith("["):
            break
        if not line.strip():
            continue
        wafer = next(csv.reader([line.rstrip("\n")]))[wafer_col]
        if wafer not in found:
            found.append(wafer)
    return found
