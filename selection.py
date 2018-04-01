#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 07:43:46 2018

@author: Mariam Barhoumi <mariam.barhoumi@u-psud.com>
"""
import math
import numpy as np

datapath="./public_data/"

with open(datapath+"cifar10_valid.data", "r") as f:
    matrice = np.empty((1000, 256))
    ligne = f.readline()
    i = 0
    while ligne != "":
        matrice[i] = ligne.split(None,256)
        ligne = f.readline()
        i+=1

def isZero(mat,c):
    b = True
    for j in range(mat[c].size):
        if mat[c][j] != 0:
            b = False
    return b

def moy(mat, c):
    moy = 0; cpt = 0
    for j in range(mat[c].size):
        moy += mat[c][j]
        cpt += 1
    moy /= cpt
    return moy

def cov(mat, x, y):
    r = 0
    for j in range(mat[x].size):
        r+=(mat[x][j]-moy(mat,x))*(mat[y][j]-moy(mat,y))
    return r

def ecartType(mat, c):
    r = 0
    for j in range(mat[c].size):
        r+=(mat[c][j]-moy(mat,c))**2
    r = math.sqrt(r)
    return r
    

def corr(mat,x,y):
   return (cov(mat,x,y)/(ecartType(mat,x))*(ecartType(mat,y)))

def add(l,e):
    if e not in l :
        l.append(e)

def badZeros(mat):
    badCols = []
    for j in range(matrice[0].size):
        if(isZero(matrice,j)):
            add(badCols,j)
    return badCols
        
def badCorr(mat, eps):
    badCols = []
    for i in range(matrice.size):
        for j in range(matrice[i].size):
            if(abs(corr(matrice,i,j)) >= eps):
                add(badCols,j)
    return badCols

def remove(mat, l):
    sorted(l)
    for j in range(mat[0].size):
        if j in l:
            np.delete(mat,j,axis=1)

List = badZeros(matrice)+badCorr(matrice,7e-2)
mat_result = remove(matrice,List)

#np.savetxt('selected_valid.txt', mat_result)
