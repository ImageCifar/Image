# -*- coding: utf-8 -*-
from model import model 
import numpy as np

"""
Created on Fri Mar 02 11:14:00 2018

@author: zakar
"""
""" la création du modele """

"""with open("cifar10_train.data",'r') as f:
    X = np.array ([[float (i) for i in l.split(" ")]for l in f])
    
#print(X) 

with open("cifar10_train.solution",'r') as f:
    Y = np.array ([[float (i) for i in l.split(" ")]for l in f])
    
with open("cifar10_valid.data",'r') as f:
    X2 = np.array ([[float (i) for i in l.split(" ")]for l in f])
    
with open("cifar10_valid.solution",'r') as f:
    Y2 = np.array ([[float (i) for i in l.split(" ")]for l in f])"""
    
with open("cifar10_train.predict",'r') as f:
    X = np.array ([[float (i) for i in l.split(" ")]for l in f])
    
#print(X) 

with open("cifar10_train.solution",'r') as f:
    Y = np.array ([[float (i) for i in l.split(" ")]for l in f])    
   

def compute_accuracy(prediction, Y_test):
    right = 0
    wrong = 0
    max = 0.
    for i in range(prediction.shape[0]):
        for j in range(prediction.shape[1]):
            if prediction[i][j] > max :
                getj = j
                max = prediction[i][j]
        if Y_test[i][getj] == 1:
             right += 1
        else:
            wrong += 1      
    return right, wrong    
            
            
        

m = model() 
m.define_model("KNeighborsClassifier") #on peux choisir plusieurs modeles dans notre cas on prend le KNN
"""X,Y =  preprocessing(**) #on récupère les donner réorganiser par le groupe preprocessing """
m.fit(X,Y) #on entraine les data grace à la fonction fit de la classe model
 
p = m.predict(X2) # on predit les classes des données X3(qui représente "the test data" ) """

# Y_test = convert_to_num(Ybin, verbose=True)
right, wrong = compute_accuracy(X, Y) 
print("right : ",right)
print("wrong : ",wrong)
print("accuracy : ",float(right)/float(right+wrong))         
print("wrong : ",float(wrong)/float(right+wrong)) 

#m.predict(X2) # on predit les classes des données X2(qui représente "the validation data ) """