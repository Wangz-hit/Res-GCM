# Res-GCM

**Res-Graph Convolutional Markov Chains for Aero-Engine Damage Trend Prediction**

[English](README.md) | [简体中文](README.zh-CN.md)

Python implementation of Res-GCM for predicting flight cycles between consecutive damage inspections. The pipeline combines graph-based edge regression and transition-specific Markov residual correction.

## Overall Framework

![Overall framework of Res-GCM](figures/fig1_overall_framework.png)

*Fig. 1. Overall framework of Res-GCM.*

## Method

- **Graph construction:** Inspection records are represented as nodes, with each engine forming a subgraph connected through a global node. Non-increasing damage records are filtered.
- **Graph prediction:** A residual graph convolutional encoder and an edge decoder estimate flight-cycle intervals using damage and graph features.
- **Residual correction:** Damage lengths are grouped into three ordered classes. Leave-one-engine-out predictions supply residual sequences for transition-specific Markov models and corrected cycle estimates.

### Transition-Specific Residual Sequences

![Residual sequences obtained during subgraph merging](figures/fig5_markov_residual_sequences.png)

*Fig. 5. Prediction and residual acquisition for transition classes during subgraph merging.*

## Installation

Python 3.10 or later:

```bash
python -m pip install -r requirements.txt
```

## Usage

Training CSV columns: `engine_id`, `cycles` (cumulative flight cycles), `crack_length` (mm). Each engine must have at least two inspections. Provide at least three training engines.

For evaluation, the test CSV can use the same columns. For prediction without known cycle values, use `engine_id`, `inspection_order`, `crack_length`. Training and test engine IDs must not overlap.

```bash
python run.py --train path/to/train.csv --test path/to/test.csv
```

Optional arguments include `--epochs`, `--seed`, `--hidden-dim`, `--lr`, `--dropout`, `--alpha`, and `--residual-state-strategy`. Results are written to `outputs/predictions.csv`. The `predicted_cycles` column contains graph predictions; `revised_cycles` contains Markov-corrected predictions. When test labels are supplied, evaluation metrics are printed.

Run the tests with:

```bash
python -m pip install pytest
python -m pytest -q
```

## Reference

Wang, Z., Fu, X., & Zhang, J. (2024). *Res-GCM: A Damage Trend Prediction Method for Aero-Engines Based on Res-Graph Convolutional Markov Chains*. 2024 IEEE International Conference on Sensing, Diagnostics, Prognostics, and Control (SDPC). https://doi.org/10.1109/SDPC62810.2024.10707736

```bibtex
@inproceedings{wang2024resgcm,
  title={Res-GCM: A Damage Trend Prediction Method for Aero-Engines Based on Res-Graph Convolutional Markov Chains},
  author={Wang, Zhe and Fu, Xuyun and Zhang, Jinzhu},
  booktitle={2024 IEEE International Conference on Sensing, Diagnostics, Prognostics, and Control (SDPC)},
  year={2024},
  doi={10.1109/SDPC62810.2024.10707736}
}
```
