# Supplemental material of'Fast Unsupervised Graph Embedding via Graph Zoom Learning' (submitted to VLDB)

## Reproducibe our result.
1. Install: pip install -m requirement.txt

2. Running GZL model: cd GZL; python train_gzl.py --dataset Cora --coarsening_ratio 0.1

## Additional experiments: embedding visualization (using t-SNE).
We plot the t-SNE 2D projection of the learned node embedding for the CORA dataset with a zoom-out rate of 0.1, and use silhouette scores (SIL) to evaluate the results.

Silhouette refers to a method of interpretation and validation of consistency within clusters of data. The technique provides a succinct graphical representation of how well each object has been classified.

The silhouette score ranges from −1 to +1, where **a high value** indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters


| **Methods** | **Silhouette score (SIL)** | **2D t-SNE projections** |
| ------- | ----------------------|----------------------|
|   GZL   | **0.17586839(Best)**  |<img src="https://github.com/Young0222/pvldb2023/blob/main/figures/gzl_tsne_result.png" width="150">|
|   GRACE | 0.119015895           |<img src="https://github.com/Young0222/pvldb2023/blob/main/figures/grace_tsne_result.png" width="150">|
|   BGRL  | 0.14092626            |<img src="https://github.com/Young0222/pvldb2023/blob/main/figures/bgrl_tsne_result.png" width="150">|
|   GCA   | 0.103519626           |<img src="https://github.com/Young0222/pvldb2023/blob/main/figures/gca_tsne_result.png" width="150">|
| GraphCL | -0.05333803           |<img src="https://github.com/Young0222/pvldb2023/blob/main/figures/graphcl_tsne_result.png" width="150">|
|   DGI   | 0.026580125           |<img src="https://github.com/Young0222/pvldb2023/blob/main/figures/dgi_tsne_result.png" width="150">|
|   MVGRL | 0.12027985            |<img src="https://github.com/Young0222/pvldb2023/blob/main/figures/mvgrl_tsne_result.png" width="150">|

