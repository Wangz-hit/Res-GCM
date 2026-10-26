from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import TRANSITIONS


BASE_FEATURES = (
    "crack_from",
    "crack_to",
    "delta_crack",
    "crack_mid",
    "crack_from_sq",
    "crack_to_sq",
    "crack_mid_sq",
    "edge_order",
    "edge_fraction",
    "engine_edge_count",
    "engine_crack_start",
    "engine_crack_end",
    "engine_crack_span",
)
PRIOR_FEATURES = (
    "transition_cycle_mean",
    "transition_cycle_median",
    "transition_cycle_std",
    "transition_cycle_count",
    "fleet_cycle_mean",
    "fleet_cycle_std",
)
FEATURE_COLUMNS = BASE_FEATURES + tuple(f"transition_{t}" for t in TRANSITIONS) + PRIOR_FEATURES


@dataclass
class EdgeFeatureBuilder:
    """Build leakage-aware edge features and training-only target priors."""

    train_edges: pd.DataFrame

    def _stats(self, reference: pd.DataFrame) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
        grouped = reference.groupby("transition")["actual_cycles"]
        stats: dict[str, dict[str, float]] = {}
        for transition, values in grouped:
            std = float(values.std(ddof=0)) if len(values) > 1 else 0.0
            stats[str(transition)] = {
                "mean": float(values.mean()),
                "median": float(values.median()),
                "std": std,
                "count": float(len(values)),
            }
        fleet = reference["actual_cycles"].to_numpy(dtype=float)
        fleet_stats = {
            "mean": float(np.mean(fleet)),
            "std": float(np.std(fleet)),
            "median": float(np.median(fleet)),
            "count": float(len(fleet)),
        }
        return stats, fleet_stats

    @staticmethod
    def _base_frame(edges: pd.DataFrame) -> pd.DataFrame:
        frame = edges.copy()
        frame["crack_from_sq"] = frame["crack_from"] ** 2
        frame["crack_to_sq"] = frame["crack_to"] ** 2
        frame["crack_mid_sq"] = frame["crack_mid"] ** 2
        for transition in TRANSITIONS:
            frame[f"transition_{transition}"] = (
                frame["transition"].eq(transition).astype(float)
            )
        return frame

    def _apply_priors(
        self, edges: pd.DataFrame, reference: pd.DataFrame
    ) -> pd.DataFrame:
        frame = self._base_frame(edges)
        stats, fleet = self._stats(reference)
        fallback = {
            "mean": fleet["mean"],
            "median": fleet["median"],
            "std": fleet["std"],
            "count": 0.0,
        }
        selected = [stats.get(str(t), fallback) for t in frame["transition"]]
        frame["transition_cycle_mean"] = [item["mean"] for item in selected]
        frame["transition_cycle_median"] = [item["median"] for item in selected]
        frame["transition_cycle_std"] = [item["std"] for item in selected]
        frame["transition_cycle_count"] = [item["count"] for item in selected]
        frame["fleet_cycle_mean"] = fleet["mean"]
        frame["fleet_cycle_std"] = fleet["std"]
        return frame.loc[:, FEATURE_COLUMNS].astype(float)

    def transform_training(self) -> pd.DataFrame:
        """Use leave-one-engine-out priors for every supervised training edge."""
        parts: list[pd.DataFrame] = []
        for engine_id, engine_edges in self.train_edges.groupby(
            "engine_id", sort=False
        ):
            reference = self.train_edges.loc[
                self.train_edges["engine_id"].ne(engine_id)
            ]
            if reference.empty:
                reference = self.train_edges
            transformed = self._apply_priors(engine_edges, reference)
            transformed.index = engine_edges.index
            parts.append(transformed)
        return pd.concat(parts).sort_index()

    def transform(self, edges: pd.DataFrame) -> pd.DataFrame:
        return self._apply_priors(edges, self.train_edges)
