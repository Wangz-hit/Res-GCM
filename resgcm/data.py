from __future__ import annotations

from dataclasses import dataclass
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


REQUIRED_COLUMNS = {"engine_id", "cycles", "crack_length"}
TRANSITIONS = ("1->1", "1->2", "2->2", "2->3", "3->3")


def load_damage_csv(path: str) -> pd.DataFrame:
    """Load and validate cumulative inspection records."""
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="raise")
    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Damage data contains missing required values")
    if (df["engine_id"] <= 0).any() or (df["crack_length"] <= 0).any():
        raise ValueError("engine_id and crack_length must be positive")
    if "number" not in df.columns:
        df.insert(0, "number", range(len(df)))

    df = df.sort_values(["engine_id", "cycles"]).reset_index(drop=True)
    for engine_id, group in df.groupby("engine_id", sort=False):
        if not group["cycles"].is_monotonic_increasing:
            raise ValueError(f"Cycles are not monotone for engine {engine_id}")
    return df


@dataclass
class DamageClusterer:
    """Training-only KMeans damage classes with ordered labels."""

    model: KMeans
    label_map: dict[int, int]
    centers: np.ndarray
    boundaries: np.ndarray

    @classmethod
    def fit(
        cls, lengths: np.ndarray | pd.Series, seed: int = 42
    ) -> "DamageClusterer":
        values = np.asarray(lengths, dtype=float).reshape(-1, 1)
        model = KMeans(n_clusters=3, n_init=20, random_state=seed).fit(values)
        order = np.argsort(model.cluster_centers_.ravel())
        label_map = {int(old): int(rank + 1) for rank, old in enumerate(order)}
        centers = model.cluster_centers_.ravel()[order]
        boundaries = (centers[:-1] + centers[1:]) / 2.0
        return cls(model, label_map, centers, boundaries)

    def classify(self, values: np.ndarray | pd.Series | float) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        return np.digitize(arr, self.boundaries, right=False) + 1

    def transition(self, crack_from: float, crack_to: float) -> str:
        start, end = self.classify([crack_from, crack_to])
        return f"{int(start)}->{int(end)}"


def _monotone_engine(group: pd.DataFrame) -> pd.DataFrame:
    """Apply the paper's monotone damage-node filtering rule."""
    group = group.sort_values("cycles").copy()
    keep: list[bool] = []
    maximum = -np.inf
    for value in group["crack_length"].to_numpy(dtype=float):
        valid = value >= maximum
        keep.append(valid)
        if valid:
            maximum = value
    return group.loc[keep].reset_index(drop=True)


def edge_table_from_engine(
    df: pd.DataFrame, clusterer: DamageClusterer | None = None
) -> pd.DataFrame:
    """Convert cumulative engine records into supervised consecutive edges."""
    if clusterer is None:
        clusterer = DamageClusterer.fit(df["crack_length"])

    rows: list[dict[str, float | int | str]] = []
    for engine_id, raw_group in df.groupby("engine_id", sort=True):
        group = _monotone_engine(raw_group)
        values = group[["cycles", "crack_length"]].to_numpy(dtype=float)
        edge_count = max(len(values) - 1, 1)
        crack_start = float(values[0, 1])
        crack_end = float(values[-1, 1])
        for order in range(len(values) - 1):
            crack_from = float(values[order, 1])
            crack_to = float(values[order + 1, 1])
            rows.append(
                {
                    "engine_id": int(engine_id),
                    "edge_order": order,
                    "edge_fraction": order / max(edge_count - 1, 1),
                    "engine_edge_count": edge_count,
                    "engine_crack_start": crack_start,
                    "engine_crack_end": crack_end,
                    "engine_crack_span": crack_end - crack_start,
                    "crack_from": crack_from,
                    "crack_to": crack_to,
                    "delta_crack": crack_to - crack_from,
                    "crack_mid": (crack_from + crack_to) / 2.0,
                    "transition": clusterer.transition(crack_from, crack_to),
                    "actual_cycles": float(values[order + 1, 0] - values[order, 0]),
                }
            )
    return pd.DataFrame(rows)
