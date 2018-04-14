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

    def fit(self, X, y):
        return self.ppl.fit(X, y)

    def transform(self, X, y=None):
        return self.ppl.transform(X)

def convert_to_num(Ybin, verbose=True):
    ''' Convert binary targets to numeric vector (typically classification target values)'''
    if verbose: print("Converting to numeric vector")
    Ybin = np.array(Ybin)
    if len(Ybin.shape) ==1: return Ybin
    classid=range(Ybin.shape[1])
    Ycont = np.dot(Ybin, classid)
    if verbose: print(Ycont)
    return Ycont

   

def parseFile(path):
    
    with open(path, "r") as f:
        data = []
        for line in f:
            bits = []
            for bit in line.split(" "):
                bits.append(float(bit))
            data.append(bits)
        print("Successfully parsed data at \" {}\"".format(path))
        return np.array(data)


 
if __name__ == '__main__':
    ## Here we are testing our preprocessing class
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import datasets
    prep = Preprocessing()
    ppl = ppl.Pipeline([('rm-variance', fs.VarianceThreshold()),
                                 ('kbest', fs.SelectKBest().set_params(k = 75)),
                                 ('ft-agglo', cls.FeatureAgglomeration(20)),
                                 ('mod', RandomForestClassifier(n_estimators = 100))])
    gscv = ms.GridSearchCV(ppl, 
            {'kbest__k' : [50, 75, 100, 150, 200], 'rm-variance__threshold' : [p*(1-p) for p in [0.8, 0.9]], 'ft-agglo__n_clusters' : [5, 10, 15, 20]},
            scoring='accuracy',
                verbose=2,
                n_jobs=9) 
    datapath = "../public_data/"
    X, y = parseFile(datapath+"cifar10_train.data"), convert_to_num(parseFile(datapath+"cifar10_train.solution"))
    gscv.fit(X, y)
    print(gscv.best_score_)
    print(gscv.best_params_)

