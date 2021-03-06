'''Examples of organizer-provided metrics.
You can just replace this code by your own.
Make sure to indicate the name of the function that you chose as metric function
in the file metric.txt. E.g. mse_metric, because this file may contain more 
than one function, hence you must specify the name of the function that is your metric.'''

import numpy as np
import scipy as sp
import sklearn
def accuracy_per_class(pred, label):
    conf = sklearn.metrics.confusion_matrix(label, pred)
    #print(conf)
    return np.diag(conf)/conf.sum(axis=1)
