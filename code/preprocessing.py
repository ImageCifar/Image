import math as ma
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline

datapath = "./public_data/"

class RmZero(FunctionTransformer):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[~np.all(X.T == 0, axis=0)]

class RmCorrelated(FunctionTransformer):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        correlations = np.corrcoef(X, rowvar=False)-np.eye(data.shape[1])
        n = X.shape[1]
        #print(np.where(correlations>0.8))
        #for i in range(n):
        #    for j in range(n):
        #        corr = np.triu(correlations)[i,j]
        #        if corr > 0.5:
        #            print(corr)
        return X

if __name__ == '__main__':
    with open(datapath+"cifar10_valid.data", "r") as f:
        data = []
        for line in f:
            bits = []
            for bit in line.split(" "):
                bits.append(float(bit))
            data.append(bits)
        data = np.array(data)

    ppl = Pipeline([('rmz', RmZero()),
        ('rmcorr', RmCorrelated()),
        ('norm', Normalizer())])

    print(ppl.fit_transform(data))
