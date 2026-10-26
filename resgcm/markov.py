from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import regression_metrics


def transition_matrix(
    states: np.ndarray, state_count: int = 3, smoothing: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(states, dtype=int)
    frequency = np.zeros((state_count, state_count), dtype=float)
    for current, following in zip(states[:-1], states[1:]):
        frequency[current, following] += 1.0
    probability = frequency + smoothing
    probability /= probability.sum(axis=1, keepdims=True)
    return probability, frequency


def _state_bins(residuals: np.ndarray, strategy: str) -> np.ndarray:
    residuals = np.asarray(residuals, dtype=float)
    median = float(np.median(residuals))
    std = float(np.std(residuals))
    if strategy == "quantile" and len(residuals) >= 9:
        low, high = np.quantile(residuals, [1 / 3, 2 / 3])
        if high - low >= max(2.0, 0.1 * std):
            return np.array([-np.inf, low, high, np.inf], dtype=float)
    threshold = max(5.0, 0.45 * std)
    if strategy == "signed":
        return np.array([-np.inf, -threshold, threshold, np.inf], dtype=float)
    return np.array(
        [-np.inf, median - threshold, median + threshold, np.inf], dtype=float
    )


def _state_values(
    residuals: np.ndarray, states: np.ndarray, bins: np.ndarray
) -> np.ndarray:
    values = np.zeros(3, dtype=float)
    global_median = float(np.median(residuals))
    spread = max(float(np.std(residuals)), 5.0)
    for state in range(3):
        selected = residuals[states == state]
        if len(selected):
            values[state] = float(np.median(selected))
        elif state == 0:
            values[state] = (
                float(bins[1] - spread / 2) if np.isfinite(bins[1]) else global_median
            )
        elif state == 2:
            values[state] = (
                float(bins[2] + spread / 2) if np.isfinite(bins[2]) else global_median
            )
        else:
            values[state] = global_median
    return values


@dataclass
class ResidualModel:
    relation: str
    residual_mean: float
    residual_median: float
    residual_std: float
    sample_count: int
    bins: np.ndarray
    states: np.ndarray
    transition_matrix: np.ndarray
    transition_frequency: np.ndarray
    state_values: np.ndarray
    state_probabilities: np.ndarray
    last_states: np.ndarray
    stable: bool
    fallback_reason: str


class MarkovResidualCorrector:
    """Transition-specific Markov residual correction with robust fallbacks."""

    def __init__(
        self,
        *,
        strategy: str = "quantile",
        alpha: float = 0.5,
        min_samples: int = 6,
    ):
        if strategy not in {"quantile", "dynamic", "signed"}:
            raise ValueError(f"Unknown residual-state strategy: {strategy}")
        self.strategy = strategy
        self.alpha = float(alpha)
        self.min_samples = int(min_samples)
        self.models: dict[str, ResidualModel] = {}

    @staticmethod
    def _long_form(residual_data: pd.DataFrame) -> pd.DataFrame:
        if {"transition", "residual"}.issubset(residual_data.columns):
            frame = residual_data.copy()
            if "engine_id" not in frame:
                frame["engine_id"] = 0
            if "edge_order" not in frame:
                frame["edge_order"] = np.arange(len(frame))
            return frame

        value_columns = [column for column in residual_data if column != "step"]
        frame = residual_data.melt(
            id_vars=["step"] if "step" in residual_data else None,
            value_vars=value_columns,
            var_name="transition",
            value_name="residual",
        ).dropna(subset=["residual"])
        frame["engine_id"] = 0
        frame["edge_order"] = (
            frame["step"] if "step" in frame else np.arange(len(frame))
        )
        return frame

    def fit(self, residual_data: pd.DataFrame) -> "MarkovResidualCorrector":
        frame = self._long_form(residual_data)
        self.models = {}
        for relation, group in frame.groupby("transition", sort=True):
            group = group.sort_values(["engine_id", "edge_order"])
            residuals = group["residual"].to_numpy(dtype=float)
            if not len(residuals):
                continue
            bins = _state_bins(residuals, self.strategy)
            states = np.digitize(residuals, bins[1:-1], right=False)
            probability, frequency = transition_matrix(states)
            state_probability = (
                np.bincount(states, minlength=3).astype(float) + 0.5
            )
            state_probability /= state_probability.sum()
            occupied_states = int(np.count_nonzero(np.bincount(states, minlength=3)))
            observed_transitions = int(frequency.sum())
            stable = (
                len(residuals) >= self.min_samples
                and occupied_states >= 2
                and observed_transitions >= 3
                and np.isfinite(probability).all()
            )
            reasons = []
            if len(residuals) < self.min_samples:
                reasons.append("too_few_samples")
            if occupied_states < 2:
                reasons.append("single_state")
            if observed_transitions < 3:
                reasons.append("too_few_transitions")
            self.models[str(relation)] = ResidualModel(
                relation=str(relation),
                residual_mean=float(np.mean(residuals)),
                residual_median=float(np.median(residuals)),
                residual_std=float(np.std(residuals)),
                sample_count=int(len(residuals)),
                bins=bins,
                states=states,
                transition_matrix=probability,
                transition_frequency=frequency,
                state_values=_state_values(residuals, states, bins),
                state_probabilities=state_probability,
                last_states=states[-min(3, len(states)) :],
                stable=stable,
                fallback_reason=";".join(reasons) if reasons else "",
            )
        return self

    @staticmethod
    def _expected_residual(model: ResidualModel) -> tuple[float, str]:
        if not model.stable:
            return model.residual_median, "median_fallback"

        next_probability = np.zeros(3, dtype=float)
        weights = np.arange(1, len(model.last_states) + 1, dtype=float)
        weights /= weights.sum()
        for state, weight in zip(model.last_states, weights):
            next_probability += weight * model.transition_matrix[int(state)]
        probability = 0.65 * next_probability + 0.35 * model.state_probabilities
        probability /= probability.sum()
        markov_estimate = float(np.dot(probability, model.state_values))

        reliability = min(1.0, model.sample_count / 18.0)
        estimate = (
            reliability * markov_estimate
            + (1.0 - reliability) * model.residual_median
        )
        estimate = float(
            np.clip(
                estimate,
                model.residual_median - 1.5 * max(model.residual_std, 5.0),
                model.residual_median + 1.5 * max(model.residual_std, 5.0),
            )
        )
        return estimate, "markov"

    def estimate(self, relation: str) -> tuple[float, str]:
        model = self.models.get(str(relation))
        if model is None:
            return 0.0, "missing_transition"
        return self._expected_residual(model)

    def correct_one(self, prediction: float, relation: str) -> float:
        residual, _ = self.estimate(relation)
        return float(max(1.0, prediction + self.alpha * residual))

    def correct_many(
        self, predictions: np.ndarray | pd.Series, relations: pd.Series | list[str]
    ) -> np.ndarray:
        return np.asarray(
            [
                self.correct_one(float(prediction), str(relation))
                for prediction, relation in zip(predictions, relations)
            ],
            dtype=float,
        )

    def summary_frame(self) -> pd.DataFrame:
        rows = []
        for relation, model in sorted(self.models.items()):
            estimate, method = self._expected_residual(model)
            rows.append(
                {
                    "transition": relation,
                    "sample_count": model.sample_count,
                    "residual_mean": model.residual_mean,
                    "residual_median": model.residual_median,
                    "residual_std": model.residual_std,
                    "state_intervals": model.bins.tolist(),
                    "transition_matrix": model.transition_matrix.tolist(),
                    "last_states": model.last_states.tolist(),
                    "stable": model.stable,
                    "fallback_reason": model.fallback_reason,
                    "correction_method": method,
                    "expected_residual": estimate,
                }
            )
        return pd.DataFrame(rows)


def select_correction_alpha(
    oof_predictions: pd.DataFrame,
    alphas: tuple[float, ...] = (0.3, 0.5, 0.7),
    strategy: str = "quantile",
) -> tuple[float, pd.DataFrame]:
    """Select shrinkage by fitting correction on other OOF engines only."""
    frame = oof_predictions.copy()
    frame["residual"] = (
        frame["actual_cycles"] - frame["predicted_cycles"]
    )
    engines = sorted(int(value) for value in frame["engine_id"].unique())
    rows = []
    for alpha in alphas:
        actual_parts = []
        corrected_parts = []
        for engine_id in engines:
            fit_frame = frame.loc[frame["engine_id"].ne(engine_id)]
            eval_frame = frame.loc[frame["engine_id"].eq(engine_id)]
            corrector = MarkovResidualCorrector(
                strategy=strategy, alpha=alpha
            ).fit(fit_frame)
            corrected = corrector.correct_many(
                eval_frame["predicted_cycles"], eval_frame["transition"]
            )
            actual_parts.extend(eval_frame["actual_cycles"].to_numpy(dtype=float))
            corrected_parts.extend(corrected)
        rows.append(
            {
                "correction_alpha": alpha,
                **regression_metrics(actual_parts, corrected_parts),
            }
        )
    table = pd.DataFrame(rows).sort_values(["MSE", "MAE"]).reset_index(drop=True)
    return float(table.iloc[0]["correction_alpha"]), table
