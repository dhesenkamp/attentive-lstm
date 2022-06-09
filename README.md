# Self-Attention LSTM for Human Motion Analysis

This is a Tensorflow implementation of a bidirectional, layer-normalized, self-attentive LSTM as proposed by Coskun et al. (2018) in their paper [_Human Motion Analysis with Deep Metric Learning_](https://arxiv.org/abs/1807.11176v2).

The self-attention mechanism is based on Lin et al. (2017): [_A Structured Self-attentive Sentence Embedding_](https://arxiv.org/abs/1703.03130v1)

Layer normalization is based on Ba et al. (2016): [_Layer Normalization_](https://arxiv.org/abs/1607.06450v1)

### Network architecture
| ![alt text](/img/architecture.png){align=center} |
|:--:|
| A-LSTM network architecture - figure taken from Coskun et al. (2018) |