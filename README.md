# Weighted Normalization

https://arxiv.org/abs/1602.07868

## dataset

CIFAR-10

## network

### conv-plain.py

3-layers CNN.
every layer-output will be BatchNormalized.

### conv-wn.py

every CNN and Linear (Dense) layer's weights are normalized.

that is

parameters a scalar $L >0$ and a vector $V$ are used instead of the vector $W$.
And $W = L V / |V|$.

