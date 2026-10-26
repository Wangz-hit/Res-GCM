from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
import torch.nn.functional as F

from .graph import GraphData


torch.set_num_threads(1)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class GraphConv(nn.Module):
    """Symmetrically normalized graph convolution with explicit self-loops."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        node_count = x.size(0)
        loop = torch.arange(node_count, device=x.device)
        source = torch.cat([edge_index[0], loop])
        target = torch.cat([edge_index[1], loop])

        degree = torch.zeros(node_count, dtype=x.dtype, device=x.device)
        degree.index_add_(0, target, torch.ones_like(target, dtype=x.dtype))
        inv_sqrt = degree.clamp_min(1.0).pow(-0.5)
        norm = inv_sqrt[source] * inv_sqrt[target]

        transformed = self.linear(x)
        aggregated = torch.zeros(
            (node_count, transformed.size(1)),
            dtype=transformed.dtype,
            device=transformed.device,
        )
        aggregated.index_add_(0, target, transformed[source] * norm.unsqueeze(-1))
        return aggregated + self.bias


class ResGCMPredictor(nn.Module):
    """Residual GCN encoder plus an edge-centric MLP decoder."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = dropout
        self.input_projection = nn.Linear(node_dim, hidden_dim)
        self.conv1 = GraphConv(hidden_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        decoder_dim = hidden_dim * 4 + edge_dim
        self.edge_decoder = nn.Sequential(
            nn.Linear(decoder_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        hidden = self.norm1(F.gelu(self.input_projection(x)))
        first = self.conv1(hidden, edge_index)
        hidden = self.norm2(hidden + F.dropout(F.gelu(first), self.dropout, self.training))
        second = self.conv2(hidden, edge_index)
        hidden = self.norm3(hidden + F.dropout(F.gelu(second), self.dropout, self.training))
        third = self.conv3(hidden, edge_index)
        return hidden + 0.5 * F.dropout(F.gelu(third), self.dropout, self.training)

    def predict_edges(
        self,
        node_embeddings: torch.Tensor,
        pairs: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        source, target = pairs
        source_embedding = node_embeddings[source]
        target_embedding = node_embeddings[target]
        decoder_input = torch.cat(
            [
                source_embedding,
                target_embedding,
                torch.abs(target_embedding - source_embedding),
                source_embedding * target_embedding,
                edge_features,
            ],
            dim=-1,
        )
        return self.edge_decoder(decoder_input).squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        pairs: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.encode(x, edge_index)
        return self.predict_edges(embeddings, pairs, edge_features)


@dataclass
class ValidationBatch:
    x: torch.Tensor
    message_edge_index: torch.Tensor
    edge_pairs: torch.Tensor
    edge_features: np.ndarray
    targets: np.ndarray


@dataclass
class TrainingResult:
    model: ResGCMPredictor
    node_scaler: StandardScaler
    edge_scaler: StandardScaler
    target_scaler: StandardScaler
    best_epoch: int
    history: list[dict[str, float]]

    def predict(
        self,
        x: torch.Tensor,
        message_edge_index: torch.Tensor,
        pairs: torch.Tensor,
        edge_features: np.ndarray,
    ) -> np.ndarray:
        scaled_x = torch.tensor(
            self.node_scaler.transform(x.detach().cpu().numpy()),
            dtype=torch.float32,
        )
        scaled_edge = torch.tensor(
            self.edge_scaler.transform(np.asarray(edge_features, dtype=float)),
            dtype=torch.float32,
        )
        self.model.eval()
        with torch.no_grad():
            scaled_prediction = self.model(
                scaled_x, message_edge_index, pairs, scaled_edge
            ).cpu().numpy()
        prediction = self.target_scaler.inverse_transform(
            scaled_prediction.reshape(-1, 1)
        ).ravel()
        return np.clip(prediction, 1.0, None)


def _scaled_validation(
    batch: ValidationBatch,
    node_scaler: StandardScaler,
    edge_scaler: StandardScaler,
    target_scaler: StandardScaler,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.tensor(
        node_scaler.transform(batch.x.detach().cpu().numpy()), dtype=torch.float32
    )
    edge = torch.tensor(
        edge_scaler.transform(np.asarray(batch.edge_features, dtype=float)),
        dtype=torch.float32,
    )
    target = torch.tensor(
        target_scaler.transform(
            np.asarray(batch.targets, dtype=float).reshape(-1, 1)
        ).ravel(),
        dtype=torch.float32,
    )
    return x, edge, target


def train_model(
    graph: GraphData,
    edge_features: np.ndarray,
    *,
    epochs: int = 2000,
    lr: float = 0.003,
    weight_decay: float = 1e-4,
    hidden_dim: int = 32,
    dropout: float = 0.1,
    seed: int = 42,
    validation: ValidationBatch | None = None,
    patience: int = 180,
    min_epochs: int = 100,
    verbose: bool = False,
) -> TrainingResult:
    """Train on standardized targets and restore the best validation epoch."""
    set_random_seed(seed)
    node_scaler = StandardScaler().fit(graph.x.detach().cpu().numpy())
    edge_scaler = StandardScaler().fit(np.asarray(edge_features, dtype=float))
    target_scaler = StandardScaler().fit(
        graph.edge_weight.detach().cpu().numpy().reshape(-1, 1)
    )

    x = torch.tensor(
        node_scaler.transform(graph.x.detach().cpu().numpy()), dtype=torch.float32
    )
    edge = torch.tensor(
        edge_scaler.transform(np.asarray(edge_features, dtype=float)),
        dtype=torch.float32,
    )
    target = torch.tensor(
        target_scaler.transform(
            graph.edge_weight.detach().cpu().numpy().reshape(-1, 1)
        ).ravel(),
        dtype=torch.float32,
    )
    scaled_validation = (
        _scaled_validation(validation, node_scaler, edge_scaler, target_scaler)
        if validation is not None
        else None
    )

    model = ResGCMPredictor(
        node_dim=x.size(1),
        edge_dim=edge.size(1),
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(patience // 4, 20)
    )

    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        prediction = model(x, graph.message_edge_index, graph.edge_pairs, edge)
        loss = F.smooth_l1_loss(prediction, target, beta=0.5)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            if scaled_validation is not None:
                val_x, val_edge, val_target = scaled_validation
                monitored_prediction = model(
                    val_x,
                    validation.message_edge_index,
                    validation.edge_pairs,
                    val_edge,
                )
                monitored_loss = F.mse_loss(
                    monitored_prediction, val_target
                ).item()
            else:
                monitored_loss = F.mse_loss(prediction, target).item()

        scheduler.step(monitored_loss)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(loss.detach()),
                "validation_loss": float(monitored_loss),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if monitored_loss < best_loss - 1e-6:
            best_loss = monitored_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose and (epoch == 1 or epoch % 200 == 0):
            print(
                f"epoch={epoch} train_loss={float(loss):.5f} "
                f"monitor_loss={monitored_loss:.5f}"
            )
        if (
            validation is not None
            and epoch >= min_epochs
            and epochs_without_improvement >= patience
        ):
            break

    model.load_state_dict(best_state)
    return TrainingResult(
        model=model,
        node_scaler=node_scaler,
        edge_scaler=edge_scaler,
        target_scaler=target_scaler,
        best_epoch=best_epoch,
        history=history,
    )
