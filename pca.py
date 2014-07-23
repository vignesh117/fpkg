__author__ = 'vignesh'

import numpy as np
from scikits.learn.decomposition import PCA


class PCAMODEL(object):
    n_components = None
    trainX = None
    trainY = None
    testX = None
    model = None
    def __init__(self, n='mle', X = None, Y = None):
        self.n_components = n
        self.trainX = X
        self.trainY = Y

    def build_model(self):
        self.model = PCA(self.n_components)
        self.model.fit(self.trainX)

    def reduce_dim(self, data):
        return self.model.transform(data)


def main():

    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    pcamodel = PCAMODEL('mle',X)
    pcamodel.build_model()
    print pcamodel.reduce_dim(X)






