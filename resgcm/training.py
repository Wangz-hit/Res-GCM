from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from .data import DamageClusterer
from .features import EdgeFeatureBuilder
from .graph import build_merged_graph, make_prediction_pairs
from .metrics import regression_metrics
from .model import TrainingResult, ValidationBatch, train_model


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 32
    lr: float = 0.003
    weight_decay: float = 1e-4
    dropout: float = 0.1


@dataclass
class CrossValidationResult:
    predictions: pd.DataFrame
    best_epochs: list[int]
    fold_metrics: list[dict[str, float | int]]


@dataclass
class FinalModelResult:
    training: TrainingResult
    clusterer: DamageClusterer
    feature_builder: EdgeFeatureBuilder
    graph: object
    training_features: pd.DataFrame
    final_epochs: int


def cross_validate(
    train_df: pd.DataFrame,
    config: ModelConfig,
    *,
    epochs: int,
    seed: int,
    validation_engines: list[int] | None = None,
    verbose: bool = False,
) -> CrossValidationResult:
    """Generate leave-one-engine-out predictions without using final validation data."""
    engines = sorted(int(value) for value in train_df["engine_id"].unique())
    if validation_engines is not None:
        engines = [engine for engine in engines if engine in validation_engines]

    predictions: list[pd.DataFrame] = []
    best_epochs: list[int] = []
    fold_metrics: list[dict[str, float | int]] = []
    for fold_index, engine_id in enumerate(engines):
        fold_train = train_df.loc[train_df["engine_id"].ne(engine_id)].copy()
        fold_validation = train_df.loc[train_df["engine_id"].eq(engine_id)].copy()
        clusterer = DamageClusterer.fit(
            fold_train["crack_length"], seed=seed + fold_index
        )
        graph = build_merged_graph(fold_train, clusterer)
        feature_builder = EdgeFeatureBuilder(graph.edge_meta)
        train_features = feature_builder.transform_training()
        x_val, message_val, pairs_val, edges_val = make_prediction_pairs(
            graph, fold_validation, clusterer
        )
        val_features = feature_builder.transform(edges_val)
        validation_batch = ValidationBatch(
            x=x_val,
            message_edge_index=message_val,
            edge_pairs=pairs_val,
            edge_features=val_features.to_numpy(),
            targets=edges_val["actual_cycles"].to_numpy(dtype=float),
        )
        training = train_model(
            graph,
            train_features.to_numpy(),
            epochs=epochs,
            lr=config.lr,
            weight_decay=config.weight_decay,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            seed=seed + fold_index,
            validation=validation_batch,
            verbose=False,
        )
        fold_prediction = training.predict(
            x_val, message_val, pairs_val, val_features.to_numpy()
        )
        edges_val = edges_val.copy()
        edges_val["predicted_cycles"] = fold_prediction
        predictions.append(edges_val)
        best_epochs.append(training.best_epoch)
        metrics = regression_metrics(edges_val["actual_cycles"], fold_prediction)
        fold_metrics.append({"engine_id": engine_id, **metrics})
        if verbose:
            print(
                f"LOEO engine={engine_id} best_epoch={training.best_epoch} "
                f"RMSE={metrics['RMSE']:.3f} R2={metrics['R2']:.3f}"
            )

    return CrossValidationResult(
        predictions=pd.concat(predictions, ignore_index=True),
        best_epochs=best_epochs,
        fold_metrics=fold_metrics,
    )


def tune_config(
    train_df: pd.DataFrame,
    initial: ModelConfig,
    *,
    epochs: int,
    seed: int,
) -> tuple[ModelConfig, pd.DataFrame]:
    """Run a compact, CPU-friendly search on three internal engines."""
    candidates = [
        initial,
        replace(initial, hidden_dim=16),
        replace(initial, hidden_dim=64),
        replace(initial, lr=0.001),
        replace(initial, lr=0.01),
        replace(initial, weight_decay=1e-3),
        replace(initial, dropout=0.0),
        replace(initial, dropout=0.2),
    ]
    unique_candidates = list(dict.fromkeys(candidates))
    engines = sorted(int(value) for value in train_df["engine_id"].unique())
    tune_engines = engines[-3:]
    rows = []
    tune_epochs = min(epochs, 800)
    for candidate_index, candidate in enumerate(unique_candidates):
        result = cross_validate(
            train_df,
            candidate,
            epochs=tune_epochs,
            seed=seed + candidate_index * 100,
            validation_engines=tune_engines,
        )
        metrics = regression_metrics(
            result.predictions["actual_cycles"],
            result.predictions["predicted_cycles"],
        )
        rows.append({**asdict(candidate), **metrics})
    table = pd.DataFrame(rows).sort_values(["MSE", "MAE"]).reset_index(drop=True)
    best = table.iloc[0]
    selected = ModelConfig(
        hidden_dim=int(best["hidden_dim"]),
        lr=float(best["lr"]),
        weight_decay=float(best["weight_decay"]),
        dropout=float(best["dropout"]),
    )
    return selected, table


def train_final_model(
    train_df: pd.DataFrame,
    config: ModelConfig,
    *,
    epochs: int,
    seed: int,
    best_epochs: list[int],
) -> FinalModelResult:
    clusterer = DamageClusterer.fit(train_df["crack_length"], seed=seed)
    graph = build_merged_graph(train_df, clusterer)
    feature_builder = EdgeFeatureBuilder(graph.edge_meta)
    training_features = feature_builder.transform_training()
    if best_epochs:
        selected_epochs = int(np.median(best_epochs) * 1.15)
        minimum_epochs = min(120, epochs)
        selected_epochs = max(minimum_epochs, min(epochs, selected_epochs))
    else:
        selected_epochs = epochs
    training = train_model(
        graph,
        training_features.to_numpy(),
        epochs=selected_epochs,
        lr=config.lr,
        weight_decay=config.weight_decay,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        seed=seed,
        validation=None,
    )
    return FinalModelResult(
        training=training,
        clusterer=clusterer,
        feature_builder=feature_builder,
        graph=graph,
        training_features=training_features,
        final_epochs=selected_epochs,
    )
