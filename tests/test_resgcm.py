import numpy as np
import pandas as pd
import torch

from resgcm.data import DamageClusterer, edge_table_from_engine
from resgcm.graph import build_merged_graph, make_prediction_pairs
from resgcm.markov import MarkovResidualCorrector, transition_matrix
from resgcm.features import EdgeFeatureBuilder
from resgcm.model import ResGCMPredictor


def inspection_frame():
    return pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "cycles": [100, 220, 340, 110, 250, 360, 120, 240, 380],
            "crack_length": [1.2, 2.0, 3.0, 1.3, 2.2, 3.2, 1.4, 2.4, 3.4],
        }
    )


def test_damage_classes_and_edges():
    frame = inspection_frame()
    clusterer = DamageClusterer.fit(frame["crack_length"])
    assert np.all(np.diff(clusterer.centers) > 0)
    edges = edge_table_from_engine(frame, clusterer)
    assert len(edges) == 6
    assert np.all(edges["actual_cycles"].to_numpy() > 0)


def test_graph_prediction_shapes():
    frame = inspection_frame()
    clusterer = DamageClusterer.fit(frame["crack_length"])
    graph = build_merged_graph(frame, clusterer)
    builder = EdgeFeatureBuilder(graph.edge_meta)
    feats = builder.transform_training().to_numpy()
    predictor = ResGCMPredictor(graph.x.shape[1], feats.shape[1])
    output = predictor(
        graph.x,
        graph.message_edge_index,
        graph.edge_pairs,
        torch.tensor(feats, dtype=graph.x.dtype),
    )
    assert output.shape == (6,)


def test_test_subgraph():
    frame = inspection_frame()
    train = frame.loc[frame.engine_id.ne(3)]
    test = frame.loc[frame.engine_id.eq(3)]
    clusterer = DamageClusterer.fit(train["crack_length"])
    graph = build_merged_graph(train, clusterer)
    x, adj, pairs, edges = make_prediction_pairs(graph, test, clusterer)
    assert x.shape[0] == graph.x.shape[0] + 3
    assert adj.shape[0] == pairs.shape[0] == 2
    assert len(edges) == 2


def test_markov_correction():
    matrix, freq = transition_matrix(np.array([0, 1, 2, 1, 0]))
    np.testing.assert_allclose(matrix.sum(axis=1), np.ones(3))
    assert int(freq.sum()) == 4
    residuals = pd.DataFrame(
        {
            "engine_id": [1] * 8,
            "edge_order": range(8),
            "transition": ["1->2"] * 8,
            "residual": [10, -5, 3, 8, -2, 4, 7, -1],
        }
    )
    corrector = MarkovResidualCorrector(alpha=0.5).fit(residuals)
    prediction = corrector.correct_many([100, 120], ["1->2", "3->3"])
    assert prediction.shape == (2,)
    assert np.isfinite(prediction).all()
    assert prediction[1] == 120


def test_unlabeled_input(tmp_path):
    from run import read_test_data

    path = tmp_path / "inspections.csv"
    pd.DataFrame(
        {"engine_id": [11, 11], "inspection_order": [1, 2], "crack_length": [1.2, 1.5]}
    ).to_csv(path, index=False)
    frame, labeled = read_test_data(path)
    assert not labeled
    assert frame["cycles"].tolist() == [0.0, 1.0]


def test_nonmonotone_test_record_is_filtered():
    frame = inspection_frame()
    train = frame.loc[frame.engine_id.ne(3)]
    clusterer = DamageClusterer.fit(train["crack_length"])
    graph = build_merged_graph(train, clusterer)
    test = pd.DataFrame(
        {"engine_id": [11] * 4, "cycles": [1, 2, 3, 4],
         "crack_length": [1.2, 1.8, 1.6, 2.1]}
    )
    x, _, pairs, edges = make_prediction_pairs(graph, test, clusterer)
    assert x.shape[0] == graph.x.shape[0] + 3
    assert pairs.shape[1] == len(edges) == 2
