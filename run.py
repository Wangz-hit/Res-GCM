from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from resgcm.data import DamageClusterer, load_damage_csv
from resgcm.graph import make_prediction_pairs
from resgcm.markov import MarkovResidualCorrector, select_correction_alpha
from resgcm.metrics import regression_metrics
from resgcm.training import ModelConfig, cross_validate, train_final_model


def read_test_data(path: Path) -> tuple[pd.DataFrame, bool]:
    frame = pd.read_csv(path)
    has_targets = "cycles" in frame.columns
    if has_targets:
        return load_damage_csv(str(path)), True

    required = {"engine_id", "inspection_order", "crack_length"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing test columns: {sorted(missing)}")
    frame = frame.copy()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame[list(required)].isna().any().any():
        raise ValueError("Test data contains missing values")
    if (frame["engine_id"] <= 0).any() or (frame["crack_length"] <= 0).any():
        raise ValueError("Engine IDs and crack lengths must be positive")
    if frame.duplicated(["engine_id", "inspection_order"]).any():
        raise ValueError("Inspection order must be unique within each engine")
    frame = frame.sort_values(["engine_id", "inspection_order"]).reset_index(drop=True)
    frame["cycles"] = frame.groupby("engine_id").cumcount().astype(float)
    return frame, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Res-GCM damage trend prediction")
    parser.add_argument("--train", required=True, type=Path, help="Training inspections CSV")
    parser.add_argument("--test", required=True, type=Path, help="Test inspections CSV")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--residual-state-strategy", choices=("quantile", "dynamic", "signed"), default="quantile")
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.alpha is not None and not 0 <= args.alpha <= 1:
        parser.error("--alpha must be between 0 and 1")

    train = load_damage_csv(str(args.train))
    test, labeled = read_test_data(args.test)
    if len(train["engine_id"].unique()) < 3:
        raise ValueError("At least three training engines are required")
    if set(train["engine_id"]) & set(test["engine_id"]):
        raise ValueError("Training and test engine IDs must be disjoint")
    if any(len(group) < 2 for _, group in test.groupby("engine_id")):
        raise ValueError("Each test engine must have at least two inspections")

    config = ModelConfig(
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
    )
    oof_result = cross_validate(train, config, epochs=args.epochs, seed=args.seed)
    oof = oof_result.predictions.copy()
    oof["residual"] = oof["actual_cycles"] - oof["predicted_cycles"]

    alpha = args.alpha
    if alpha is None:
        alpha, _ = select_correction_alpha(oof, strategy=args.residual_state_strategy)
    corrector = MarkovResidualCorrector(strategy=args.residual_state_strategy, alpha=alpha).fit(oof)
    fitted = train_final_model(
        train,
        config,
        epochs=args.epochs,
        seed=args.seed,
        best_epochs=oof_result.best_epochs,
    )

    rows = []
    for _, engine in test.groupby("engine_id", sort=True):
        x, adjacency, pairs, edges = make_prediction_pairs(
            fitted.graph, engine, fitted.clusterer
        )
        features = fitted.feature_builder.transform(edges)
        predicted = fitted.training.predict(x, adjacency, pairs, features.to_numpy())
        revised = corrector.correct_many(predicted, edges["transition"])
        result = edges.loc[:, ["engine_id", "crack_from", "crack_to", "transition"]].copy()
        result["predicted_cycles"] = predicted
        result["revised_cycles"] = revised
        if labeled:
            result["actual_cycles"] = edges["actual_cycles"].to_numpy()
        rows.append(result)

    result = pd.concat(rows, ignore_index=True)
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "predictions.csv"
    result.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    print(f"Correction alpha: {alpha:.2f}")
    if labeled:
        metrics = regression_metrics(result["actual_cycles"], result["revised_cycles"])
        print("Res-GCM:", ", ".join(f"{key}={value:.4f}" for key, value in metrics.items()))


if __name__ == "__main__":
    main()
