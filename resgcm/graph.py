from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .data import DamageClusterer, edge_table_from_engine


NODE_FEATURE_COLUMNS = (
    "crack_length",
    "crack_length_sq",
    "sequence_position",
    "distance_from_start",
    "distance_to_end",
    "engine_node_count",
    "is_global",
)


@dataclass
class GraphData:
    x: torch.Tensor
    message_edge_index: torch.Tensor
    edge_pairs: torch.Tensor
    edge_weight: torch.Tensor
    node_meta: pd.DataFrame
    edge_meta: pd.DataFrame
    global_node_id: int

def _node_features(
    crack_length: float,
    position: int,
    node_count: int,
    crack_start: float,
    crack_end: float,
    is_global: float = 0.0,
) -> dict[str, float]:
    span = max(crack_end - crack_start, 1e-6)
    return {
        "crack_length": crack_length,
        "crack_length_sq": crack_length**2,
        "sequence_position": position / max(node_count - 1, 1),
        "distance_from_start": (crack_length - crack_start) / span,
        "distance_to_end": (crack_end - crack_length) / span,
        "engine_node_count": float(node_count),
        "is_global": is_global,
    }


def build_merged_graph(
    df: pd.DataFrame,
    clusterer: DamageClusterer | None = None,
    add_global_node: bool = True,
) -> GraphData:
    """Build merged engine subgraphs without using cycle labels in node features."""
    if clusterer is None:
        clusterer = DamageClusterer.fit(df["crack_length"])

    nodes: list[dict[str, float | int | str]] = []
    message_edges: list[tuple[int, int]] = []
    supervised_pairs: list[tuple[int, int]] = []
    global_node_id = 0
    next_node_id = 0

    if add_global_node:
        fleet_crack = float(df["crack_length"].mean())
        feature = _node_features(
            fleet_crack,
            position=0,
            node_count=max(len(df), 1),
            crack_start=float(df["crack_length"].min()),
            crack_end=float(df["crack_length"].max()),
            is_global=1.0,
        )
        nodes.append(
            {
                "node_id": next_node_id,
                "engine_id": 0,
                "kind": "global",
                **feature,
            }
        )
        global_node_id = next_node_id
        next_node_id += 1

    edge_meta = edge_table_from_engine(df, clusterer).reset_index(drop=True)
    pair_cursor = 0
    for engine_id, group in df.groupby("engine_id", sort=True):
        group = group.sort_values("cycles")
        crack = group["crack_length"].to_numpy(dtype=float)
        monotone = np.maximum.accumulate(crack) == crack
        group = group.loc[monotone].reset_index(drop=True)
        crack = group["crack_length"].to_numpy(dtype=float)
        node_count = len(group)
        local_ids: list[int] = []
        for position, value in enumerate(crack):
            feature = _node_features(
                float(value),
                position,
                node_count,
                float(crack[0]),
                float(crack[-1]),
            )
            nodes.append(
                {
                    "node_id": next_node_id,
                    "engine_id": int(engine_id),
                    "kind": "damage",
                    **feature,
                }
            )
            local_ids.append(next_node_id)
            if add_global_node:
                message_edges.extend(
                    [(global_node_id, next_node_id), (next_node_id, global_node_id)]
                )
            next_node_id += 1

        for source, target in zip(local_ids[:-1], local_ids[1:]):
            supervised_pairs.append((source, target))
            message_edges.extend([(source, target), (target, source)])
            pair_cursor += 1

    if pair_cursor != len(edge_meta):
        raise RuntimeError("Graph supervised edges are not aligned with edge metadata")

    node_meta = pd.DataFrame(nodes)
    x = torch.tensor(
        node_meta.loc[:, NODE_FEATURE_COLUMNS].to_numpy(), dtype=torch.float32
    )
    message_edge_index = torch.tensor(message_edges, dtype=torch.long).T.contiguous()
    edge_pairs = torch.tensor(supervised_pairs, dtype=torch.long).T.contiguous()
    edge_weight = torch.tensor(
        edge_meta["actual_cycles"].to_numpy(), dtype=torch.float32
    )
    return GraphData(
        x=x,
        message_edge_index=message_edge_index,
        edge_pairs=edge_pairs,
        edge_weight=edge_weight,
        node_meta=node_meta,
        edge_meta=edge_meta,
        global_node_id=global_node_id,
    )


def make_prediction_pairs(
    train_graph: GraphData,
    val_df: pd.DataFrame,
    clusterer: DamageClusterer | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame]:
    """Append a label-free validation subgraph and return its consecutive edges."""
    if clusterer is None:
        clusterer = DamageClusterer.fit(
            train_graph.node_meta.loc[
                train_graph.node_meta["kind"].eq("damage"), "crack_length"
            ]
        )
    val_edges = edge_table_from_engine(val_df, clusterer).reset_index(drop=True)
    val_group = val_df.sort_values("cycles").reset_index(drop=True)
    original_crack = val_group["crack_length"].to_numpy(dtype=float)
    val_group = val_group.loc[
        np.maximum.accumulate(original_crack) == original_crack
    ].reset_index(drop=True)
    crack = val_group["crack_length"].to_numpy(dtype=float)
    base_node_count = len(train_graph.node_meta)
    new_nodes = []
    for position, value in enumerate(crack):
        new_nodes.append(
            _node_features(
                float(value),
                position,
                len(crack),
                float(crack[0]),
                float(crack[-1]),
            )
        )
    x_new = torch.tensor(
        pd.DataFrame(new_nodes).loc[:, NODE_FEATURE_COLUMNS].to_numpy(),
        dtype=torch.float32,
    )
    x_all = torch.cat([train_graph.x, x_new], dim=0)

    message_edges = train_graph.message_edge_index.T.tolist()
    pairs: list[tuple[int, int]] = []
    for offset in range(len(crack)):
        node_id = base_node_count + offset
        message_edges.extend(
            [
                (train_graph.global_node_id, node_id),
                (node_id, train_graph.global_node_id),
            ]
        )
        if offset < len(crack) - 1:
            next_id = node_id + 1
            pairs.append((node_id, next_id))
            message_edges.extend([(node_id, next_id), (next_id, node_id)])

    combined_message_edges = torch.tensor(message_edges, dtype=torch.long).T.contiguous()
    pair_tensor = torch.tensor(pairs, dtype=torch.long).T.contiguous()
    return x_all, combined_message_edges, pair_tensor, val_edges
