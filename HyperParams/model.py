'''
This code helps us to choose the hyper-parameters of our model(RandomForestClassifier) 
using ParameterGrid  
'''
import pickle
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile
import sklearn
import matplotlib.pyplot as plt 

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import  KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import ParameterGrid

# CROSS-VALIDATION ERROR
from sklearn.model_selection import KFold
from numpy import zeros, mean

class model:
    def __init__(self, x, y):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation. 
        '''
        self.num_train_samples=0
        self.num_feat=1
        self.num_labels=1
        self.is_trained=False
        self.mod = RandomForestClassifier(n_estimators = x, max_depth = y)
    
    def define_model(self, name, C=1.0):
        if self.is_trained == False:
            if name == 'GaussianNB':
                self.mod = GaussianNB()
            elif name == 'MultinomialNB':
                self.mod = MultinomialNB()
            elif name == 'Tree':
                self.mod = DecisionTreeClassifier()
            elif name == 'Forest':
                self.mod = RandomForestClassifier(max_depth=4, random_state=0, n_estimators=500)
            elif name == 'SVM_rbf':
                self.mod = SVC(C=C, kernel = 'rbf')
            elif name == 'SVM_linear':
                self.mod = SVC(C=C, kernel = 'linear')
            elif name == 'SVM_poly':
                self.mod = SVC(C=C, kernel = 'poly', degree=2)
            elif name == 'Softmax':
                self.mod = LogisticRegression(multi_class='multinomial', solver='saga')
            else:
                print('Error selecting the model, choose by default Decision Tree Model')
                self.mod = DecisionTreeClassifier()
        else:
            print("Model already load")
        
    def fit(self, X, Y):
        '''
        This function should train the model parameters.
        Here we do nothing in this example...
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        For classification, labels could be either numbers 0, 1, ... c-1 for c classe
        or one-hot encoded vector of zeros, with a 1 at the kth position for class k.
        The AutoML format support on-hot encoding, which also works for multi-labels problems.
        Use data_converter.convert_to_num() to convert to the category number format.
        For regression, labels are continuous values.
        '''
        # For multi-class problems, convert target to be scikit-learn compatible
        # into one column of a categorical variable
        y=self.convert_to_num(Y, verbose=False)     
        
        self.num_train_samples = X.shape[0]
        if X.ndim>1: self.num_feat = X.shape[1] # Does not work for sparse matrices
        print("\nFIT: dim(X)= [{:d}, {:d}]").format(self.num_train_samples, self.num_feat)
        num_train_samples = y.shape[0]
        if y.ndim>1: self.num_labels = y.shape[1]
        print("FIT: dim(y)= [{:d}, {:d}]").format(num_train_samples, self.num_labels)
        if (self.num_train_samples != num_train_samples):
            print("ARRGH: number of samples in X and y do not match!")

        ###### Baseline models ######
        from sklearn.naive_bayes import GaussianNB
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neighbors import KNeighborsClassifier
        # Comment and uncomment right lines in the following to choose the model
        #self.mod = GaussianNB()
        #self.mod = LogisticRegression() # linear regression for classification
        #self.mod = DecisionTreeClassifier()
        #self.mod = RandomForestClassifier()
        #self.mod = KNeighborsClassifier()
        
        self.mod.fit(X, y)
        self.is_trained=True

    def predict(self, X):
        '''
        This function should provide predictions of labels on (test) data.
        Here we just return zeros...
        Make sure that the predicted values are in the correct format for the scoring
        metric. For example, binary classification problems often expect predictions
        in the form of a discriminant value (if the area under the ROC curve it the metric)
        rather that predictions of the class labels themselves. For multi-class or multi-labels
        problems, class probabilities are often expected if the metric is cross-entropy.
        Scikit-learn also has a function predict-proba, we do not require it.
        The function predict eventually can return probabilities.
        '''
        num_test_samples = X.shape[0]
        if X.ndim>1: num_feat = X.shape[1]
        print("PREDICT: dim(X)= [{:d}, {:d}]".format(num_test_samples, num_feat))
        if (self.num_feat != num_feat):
            print("ARRGH: number of features in X does not match training data!")
        print("PREDICT: dim(y)= [{:d}, {:d}]".format(num_test_samples, self.num_labels))
        
        # Return predictions as class probabilities
        y = self.mod.predict_proba(X)
        return y

    def save(self, path="./"):
        with open(path + '_model.pickle', 'wb') as f:
            print('modele name : ', path + '_model.pickle')
            pickle.dump(self , f)

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self
        
    def convert_to_num(self, Ybin, verbose=True):
        ''' Convert binary targets to numeric vector (typically classification target values)'''
        if verbose: print("Converting to numeric vector")
        Ybin = np.array(Ybin)
        if len(Ybin.shape) ==1: return Ybin
        classid=range(Ybin.shape[1])
        Ycont = np.dot(Ybin, classid)
        if verbose: print (Ycont)
        return Ycont
 


######## Main function ########
if __name__ == "__main__":
    # Find the files containing corresponding data
    # To find these files successfully:
    # you should execute this "model.py" script in the folder "sample_code_submission"
    # and the folder "public_data" should be in the SAME folder as the starting kit
    path_to_training_data   =  "C:/Users/zakar/Desktop/starting_kit/vision/public_data/cifar10_train.data"
    path_to_training_label  =  "C:/Users/zakar/Desktop/starting_kit/vision/public_data/cifar10_train.solution"
    path_to_testing_data    =  "C:/Users/zakar/Desktop/starting_kit/vision/public_data/cifar10_test.data"
    path_to_validation_data =  "C:/Users/zakar/Desktop/starting_kit/vision/public_data/cifar10_valid.data"
    graphe_x = []
    graphe_y = []
    # Find the program computing balanced accuracy score
    path_to_metric = "C:/Users/zakar/Desktop/starting_kit/scoring_program/libscores.py"
    import imp
    bac_multiclass = imp.load_source('metric', path_to_metric).bac_multiclass

    # use numpy to load data
    X_train = np.loadtxt(path_to_training_data)
    y_train = np.loadtxt(path_to_training_label)
    X_test = np.loadtxt(path_to_testing_data)
    X_valid = np.loadtxt(path_to_validation_data)
    
    #Here we created a grid to combine the hyper_parameters of our model (the Random forest)      
    param_grid = {"n_estimators": [5,10, 20, 30, 50,100], "max_depth": [10,5,100,50]}
    max = 0
    for a in list(ParameterGrid(param_grid)):
        # TRAINING ERROR
        # generate an instance of our model (clf for classifier)
        clf = model(a["n_estimators"], a["max_depth"])
        # train the model
        clf.fit(X_train, y_train)
        # to compute training error, first make predictions on training set
        y_hat_train = clf.predict(X_train)
        # then compare our prediction with true labels using the metric
        training_error = bac_multiclass(y_train, y_hat_train)


        # 3-fold cross-validation
        n = 3
        kf = KFold(n_splits=n)
        kf.get_n_splits(X_train)
        i=0
        scores = zeros(n)
        for train_index, test_index in kf.split(X_train):
            Xtr, Xva = X_train[train_index], X_train[test_index]
            Ytr, Yva = y_train[train_index], y_train[test_index]
            M = model(a["n_estimators"],a["max_depth"])
            M.fit(Xtr, Ytr)
            Yhat = M.predict(Xva)
            scores[i] = bac_multiclass(Yva, Yhat)
            print ('Fold', i+1, 'example metric = ', scores[i])
            i=i+1
        cross_validation_error = mean(scores)

        # Print results
        
        print("\nThe scores are: ")
        print("**********************************")
        print(a)
        print("Training: ", training_error)
        print("Cross Validation:", cross_validation_error)
        print("*********************************")
        #graphe_y = list.append(cross_validation_error)
        #graphe_x = list.append(a)
        if cross_validation_error > max :
            min = cross_validation_error
            b = a 
    print("The best score is :", min, "and the hyper parameters for this score are :", b)     
    #x = np.linspace(1,20,1)
    
        
        
        
