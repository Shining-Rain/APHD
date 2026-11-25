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

To download Douban-Book, Douban-Movie, and Yelp together and prepare their splits in `./Data/`, run:

```bash
python prepare_data.py --download
```

| Dataset | Download | Reference |
| --- | --- | --- |
| Douban-Book | [train.txt](https://raw.githubusercontent.com/BlueGhostYi/DVGRL/2cdb4e0dfcdd129e5c42ac8eb2ec42322b3ae4ff/Data/douban-book/train.txt) · [test.txt](https://raw.githubusercontent.com/BlueGhostYi/DVGRL/2cdb4e0dfcdd129e5c42ac8eb2ec42322b3ae4ff/Data/douban-book/test.txt) · [social.txt](https://raw.githubusercontent.com/BlueGhostYi/DVGRL/2cdb4e0dfcdd129e5c42ac8eb2ec42322b3ae4ff/Data/douban-book/social.txt) | Yu et al., KDD 2021 |
| Douban-Movie | [train.txt](https://raw.githubusercontent.com/BlueGhostYi/DVGRL/2cdb4e0dfcdd129e5c42ac8eb2ec42322b3ae4ff/Data/douban-movie/train.txt) · [test.txt](https://raw.githubusercontent.com/BlueGhostYi/DVGRL/2cdb4e0dfcdd129e5c42ac8eb2ec42322b3ae4ff/Data/douban-movie/test.txt) · [social.txt](https://raw.githubusercontent.com/BlueGhostYi/DVGRL/2cdb4e0dfcdd129e5c42ac8eb2ec42322b3ae4ff/Data/douban-movie/social.txt) | Yu et al., WWW 2021 |
| Yelp | [train.txt](https://raw.githubusercontent.com/BlueGhostYi/DVGRL/2cdb4e0dfcdd129e5c42ac8eb2ec42322b3ae4ff/Data/yelp/train.txt) · [test.txt](https://raw.githubusercontent.com/BlueGhostYi/DVGRL/2cdb4e0dfcdd129e5c42ac8eb2ec42322b3ae4ff/Data/yelp/test.txt) · [social.txt](https://raw.githubusercontent.com/BlueGhostYi/DVGRL/2cdb4e0dfcdd129e5c42ac8eb2ec42322b3ae4ff/Data/yelp/social.txt) | Yin et al., ICDE 2019 |

We thank the authors of these three papers for their contributions to the social recommendation benchmarks used in this work.
