import numpy as np   # We recommend to use numpy arrays
import sklearn

from sklearn import pipeline as ppl
from sklearn import preprocessing as pp
from sklearn import feature_selection as fs
from sklearn import cluster as cls
from sklearn import model_selection as ms
from sklearn import decomposition as dc

class Preprocessing(pp.FunctionTransformer):
    '''
    Custom preprocessing class implementing
    '''
    def __init__(self):
        self.steps = [('rm-variance', fs.VarianceThreshold(0.1275)),
                                 ('kbest', fs.SelectKBest().set_params(k = 200)),
                                 ('ft-agglo', dc.PCA(26))]
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
    '''
    Helper function which parses a data file into an np.array.
    In :
      path  sting representing path to file
    Out :
      np.array representing the file
    '''
    with open(path, "r") as f:
        data = []
        for line in f:
            bits = []
            for bit in line.split(" "):
                bits.append(float(bit))
            data.append(bits)
        print("Successfully parsed data at \" {}\"".format(path))
        return np.array(data)

def bac(est, data, label):
    '''
    Ballanced Accuracy metric function. It is sklearn compatible
    In :
      est   The estimator on which the metric is applied
      data  Input (test/validation) data
      label 1-D array of numbers representinc the class of each sample
    Out :
      Floating point representing the score of the estimator. Higher is better.
    '''
    pred = est.predict(data)
    conf = sklearn.metrics.confusion_matrix(label, pred)
    #print(np.array(conf, dtype=float))
    return np.mean(np.diag(np.array(conf, dtype=float))/conf.sum(axis=1))
 
if __name__ == '__main__':
    ## Here we are testing our preprocessing class
    from sklearn.ensemble import RandomForestClassifier
    prep = Preprocessing()
    ppl = ppl.Pipeline([('rm-variance', fs.VarianceThreshold()),
                                 ('kbest', fs.SelectKBest().set_params(k=200)),
                                 ('pca', dc.PCA()),
                                 ('mod', RandomForestClassifier(n_estimators = 100))])
    # Declaring the GridSearch object instance
    gscv = ms.GridSearchCV(ppl, 
            {'rm-variance__threshold' : [p*(1-p) for p in [0.9, 0.85]], 
                'kbest__k' : ['all', 200], 
                'pca__n_components' : [24, 25, 26]},
                scoring=bac,
                verbose=2,
                n_jobs=9)

    # Fetching the data
    datapath = "../public_data/"
    X, y = parseFile(datapath+"cifar10_train.data"), convert_to_num(parseFile(datapath+"cifar10_train.solution"))
    # Perform the search
    gscv.fit(X, y)
    print(gscv.best_score_)
    print(gscv.best_params_)

