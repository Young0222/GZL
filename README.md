# Supplemental material of 'Fast Unsupervised Graph Embedding via Graph Zoom Learning' (submitted to VLDB)

## Reproduce our experimental results on Cora.
1. Install: pip install -m requirement.txt

2. Running GZL model: cd GZL; python train_gzl.py --dataset Cora --coarsening_ratio 0.1

Note that due to the limitation of upload size, we only provide an example dataset Cora with a zoom-out rate of 0.1. We will provide more experimental codes and datasets later.

The codes of baselines are in the folders of baseline(contrastive) and baseline(deepwalk+node2vec).

## Additional experiment 1: embedding visualization.
We plot the t-SNE 2D projection of the learned node embedding for the CORA dataset with a zoom-out rate of 0.1, and use silhouette scores (SIL) to evaluate the results.

Silhouette refers to a method of interpretation and validation of consistency within clusters of data. The technique provides a succinct graphical representation of how well each object has been classified.

The silhouette score ranges from −1 to +1, where **a high value** indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters.

The results are as follows (7 colors indicate 7 different classes):

| **Methods** | **Silhouette score (SIL)** | **2D t-SNE projections** |
| ------- | ----------------------|----------------------|
|   GZL   | **0.17586839(Best)**  |<img src="https://github.com/Young0222/pvldb2023/blob/main/figures/gzl_tsne_result.png" width="150">|
|   GRACE | 0.119015895           |<img src="https://github.com/Young0222/pvldb2023/blob/main/figures/grace_tsne_result.png" width="150">|
|   BGRL  | 0.14092626            |<img src="https://github.com/Young0222/pvldb2023/blob/main/figures/bgrl_tsne_result.png" width="150">|
|   GCA   | 0.103519626           |<img src="https://github.com/Young0222/pvldb2023/blob/main/figures/gca_tsne_result.png" width="150">|
| GraphCL | -0.05333803           |<img src="https://github.com/Young0222/pvldb2023/blob/main/figures/graphcl_tsne_result.png" width="150">|
|   DGI   | 0.026580125           |<img src="https://github.com/Young0222/pvldb2023/blob/main/figures/dgi_tsne_result.png" width="150">|
|   MVGRL | 0.12027985            |<img src="https://github.com/Young0222/pvldb2023/blob/main/figures/mvgrl_tsne_result.png" width="150">|

## Additional experiment 2: sensitivity experimental results of other hyperparameters.
Here we provide GZL's sensitivity experimental results of other hyperparameters including: learning rate ![1](http://latex.codecogs.com/svg.latex?\theta), weight decay ![2](http://latex.codecogs.com/svg.latex?\eta), and parameter ![3](http://latex.codecogs.com/svg.latex?\alpha).

Some simple conclusions: 

1. Learning rate ![1](http://latex.codecogs.com/svg.latex?\theta): 5e-4 and 1e-3 are good choices for GZL, but 5e-5 and 1e-4 are bad ones.

2. Weight decay ![2](http://latex.codecogs.com/svg.latex?\eta): Using 1e-7, 1e-6, and 1e-5 bring better results than others.

3. Parameter ![4](http://latex.codecogs.com/svg.latex?\alpha): Using 0.0 and 1.0 bring better results than others.

For other datasets, their conclusions are similar to Cora, e.g., using a learning rate of 1e-5 is a good choice for GZL on CS, using a weight decay of 1e-5 brings better results than other choices.


| **Learning rate** | **Weight decay** |**Parameter ![4](http://latex.codecogs.com/svg.latex?\alpha)** |
| ------- | ----------------------|----------------------|
| <img src="https://github.com/Young0222/pvldb2023/blob/main/figures/LR.png" width="150"> | <img src="https://github.com/Young0222/pvldb2023/blob/main/figures/WD.png" width="150">|<img src="https://github.com/Young0222/pvldb2023/blob/main/figures/alpha_new.png" width="150">|
