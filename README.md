# 'Fast Unsupervised Graph Embedding via Graph Zoom Learning'

## Reproduce our experimental results on Cora.
1. Install: pip install -m requirement.txt

2. Running GZL model: cd GZL; python train_gzl.py --dataset Cora --coarsening_ratio 0.1

Note that due to the limitation of upload size, we only provide an example dataset Cora with a zoom-out rate of 0.1. We will provide more experimental codes and datasets later.

The codes of baselines are in the folders of baseline(contrastive) and baseline(deepwalk+node2vec).



