import numpy as np   # We recommend to use numpy arrays
import sklearn

from sklearn import pipeline as ppl
from sklearn import preprocessing as pp
from sklearn import feature_selection as fs
from sklearn import cluster as cls
from sklearn import model_selection as ms

class Preprocessing(pp.FunctionTransformer):
    def __init__(self):
        self.steps = [('rm-variance', fs.VarianceThreshold()),
                                 ('kbest', fs.SelectKBest().set_params(k = 75)),
                                 ('ft-agglo', cls.FeatureAgglomeration(20))]
        self.ppl = ppl.Pipeline(self.steps)
        print("Prepro successfully loaded.")

    def fit(self, X, y=None):
        return self.ppl.fit(X, y)

    def transform(self, X, y=None):
        return self.ppl.transform(X)

if __name__ == '__main__':
    ## Here we are testing our preprocessing class
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import datasets
    mod = RandomForestClassifier(n_estimators = 100)
    prep = Preprocessing()
    ppl = ppl.Pipeline(prep.steps + [('mod', mod)])
    gscv = ms.GridSearchCV(prep.ppl, 
            {'kbest__k' : [50, 75, 100, 150, 200], 'rm-variance__threshold' : [p*(1-p) for p in [0.8, 0.9]], 'ft-agglo__n_clusters' : [5, 10, 15, 20]},
            scoring='accuracy',
                verbose=2) 
    data = datasets.load_iris()
    gscv.fit(data.data, data.target)
    print(gscv.best_score_)
    print(gscv.best_params_)

