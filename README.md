# Self-Attention LSTM for Human Motion Analysis

This is a Tensorflow implementation of a bidirectional, layer-normalized, self-attentive LSTM as proposed by Coskun et al. (2018) in their paper [_Human Motion Analysis with Deep Metric Learning_](https://arxiv.org/abs/1807.11176v2).

The self-attention mechanism is based on Lin et al. (2017): [_A Structured Self-attentive Sentence Embedding_](https://arxiv.org/abs/1703.03130v1)

Layer normalization is based on Ba et al. (2016): [_Layer Normalization_](https://arxiv.org/abs/1607.06450v1)


## Network architecture
| ![alt text](/img/architecture.png) |
|:--:|
| A-LSTM network architecture - figure taken from Coskun et al. (2018) |


## TO DO
- [ ] Penalization term using Frobenius norm
- [ ] MMD-NCA loss
- [ ] L2 norm


## References
<div class="csl-entry">Ba, J. L., Kiros, J. R., &#38; Hinton, G. E. (2016). <i>Layer Normalization</i>. https://doi.org/10.48550/arxiv.1607.06450</div>

<div class="csl-entry">Coskun, H., Tan, D. J., Conjeti, S., Navab, N., &#38; Tombari, F. (2018). Human Motion Analysis with Deep Metric Learning. <i>Lecture Notes in Computer Science (Including Subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics)</i>, <i>11218 LNCS</i>, 693–710. https://doi.org/10.48550/arxiv.1807.11176</div>

<div class="csl-entry">Lin, Z., Feng, M., dos Santos, C. N., Yu, M., Xiang, B., Zhou, B., &#38; Bengio, Y. (2017). A Structured Self-attentive Sentence Embedding. <i>5th International Conference on Learning Representations, ICLR 2017 - Conference Track Proceedings</i>. https://doi.org/10.48550/arxiv.1703.03130</div>