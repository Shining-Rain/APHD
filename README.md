<h1 align="center">APHD</h1>

<p align="center">
  <strong>Let Each View Shine:</strong><br>
  Role-Preserving Probability-Space Collaboration<br>
  for Heterogeneous Dual-Graph Social Recommendation
</p>

<p align="center">
  <img src="assets/architecture.png" alt="Overview of the APHD framework." width="100%">
</p>

---

![KDD 2027](https://img.shields.io/badge/KDD-2027-4267B2?style=flat-square)

**APHD (Asymmetric Personalized Heterogeneous Dual-Graph Recommendation)** is a social recommendation framework that models interaction and social graphs according to their distinct information roles. It connects individual preferences with neighborhood preference patterns through probability-space collaboration, then adaptively combines the two views according to their reliability to produce personalized recommendations.

## Install

Install [PyTorch](https://pytorch.org/get-started/locally/) for your system and CUDA version, then install the remaining dependencies:

```bash
python -m pip install numpy scipy
```

## Datasets

APHD supports `douban-book`, `douban-movie`, and `yelp`.

## Quick start

From the repository root, extract the data once, then start training:

```bash
python -m zipfile -e Data.zip .
python main.py --dataset yelp --path ./Data/
```

Evaluation reports Recall and NDCG in the terminal.

Run `python main.py --help` for all options.

The three pretrained checkpoints are available in [checkpoint.zip](checkpoint.zip), stored with Git LFS.
