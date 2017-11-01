import numpy as np
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

class FFNN(Chain):
    def __init__(self, n_source_units, n_target_units):
        super().__init__(
            W=L.Linear(n_source_units, n_target_units),
        )
        self.n_sentences = 0
        
    def __call__(self, x, y):
        batch = len(x)
        self.n_sentences += batch
        pred = self.fwd(x)
        loss = F.sum(F.softmax_cross_entropy(pred, y, reduce='no')) / batch
        return loss
    
    def fwd(self, x):
        return self.W(x)