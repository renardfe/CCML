# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 10:05:38 2012

@author: bouthillon
"""
"""
Script grouping function fo ploting
"""

import matplotlib.pyplot as pl
import numpy as np

def scatter_2D(X,l,names,file_name = "./scatter.png"):
    """
    To plot the individu in a 2D space (from feature selection, isomap...), with
    the identification of the individus.
    Input : X : array, shape[nb_ind,2]. Contains the (float) coordinates of the
                individu in the 2D space.
            l : array, shape[nb_ind,]label of the individu : 0 for controls (at
                the begining of the array) and 1 for comas
            names : array, shape[nb_ind,] strings refering to the identification
                    of the individus
            file_name : string ending by .png giving the name of the file to
                        save the plot
    """
    #Initialisations
    NPat = int(sum(l))
    NCont = len(l)-NPat
    xmin, xmax = np.min(X[:,0]),np.max(X[:,0])
    ymin, ymax = np.min(X[:,1]),np.max(X[:,1])
    #plot
    pl.figure()
    pl.title("2D representation, controls in blue, comas in red")
    pl.axis([xmin, xmax, ymin, ymax])
    for i in range(NCont):
        pl.text(X[i,0],X[i,1],names[i],fontsize= 20, color = '0.5')
    for i in range(NCont,NCont+NPat):
        pl.text(X[i,0],X[i,1],names[i],fontsize= 20 ,color = 'black')
    pl.savefig(file_name)
    pl.close

def classif_2D(X,l,clf,names,file_name = "./classif.png"):
    """
    To plot the classifier in a 2D space (from feature selection, isomap...), with
    the individus and their identification.
    Input : X : array, shape[nb_ind,2]. Contains the (float) coordinates of the
                individu in the 2D space.
            l : array, shape[nb_ind,]label of the individu : 0 for controls (at
                the begining of the array) and 1 for comas
            clf : the trained classifier, from scikit-learn toolbox
            names : array, shape[nb_ind,] strings refering to the identification
                    of the individus
            file_name : string ending by .png giving the name of the file to
                        save the plot
    """
    #initialisations
    h=0.01
    x_min, x_max = X[:, 0].min() - h, X[:, 0].max() + h
    y_min, y_max = X[:, 1].min() - h, X[:, 1].max() + h
    NPat = int(sum(l))
    NCont = len(l)-NPat
    #prediction
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    #plot
    fig = pl.figure() 
    pl.set_cmap(pl.cm.Paired)
    pl.contourf(xx, yy, Z)
    pl.title("2D representation, controls in blue, comas in red")
    for i in range(NCont):
        pl.text(X[i,0],X[i,1],names[i],color = 'blue')
    for i in range(NCont,NCont+NPat):
        pl.text(X[i,0],X[i,1],names[i],color = 'red')
    pl.savefig(file_name)
    pl.close
    
def classif_2D_var(X,l,clf,names,file_name = "./classif.png"):
    """
    To plot the classifier in a 2D space (from feature selection, isomap...), with
    the individus and their identification.
    Input : X : array, shape[nb_ind,2]. Contains the (float) coordinates of the
                individu in the 2D space.
            l : array, shape[nb_ind,]label of the individu : 0 for controls (at
                the begining of the array) and 1 for comas
            clf : the trained classifier, from scikit-learn toolbox
            names : array, shape[nb_ind,] strings refering to the identification
                    of the individus
            file_name : string ending by .png giving the name of the file to
                        save the plot
    """
    #initialisations
    h=0.01
    x_min, x_max = X[:, 0].min() - h, X[:, 0].max() + h
    y_min, y_max = X[:, 1].min() - h, X[:, 1].max() + h
    NPat = int(sum(l))
    NCont = len(l)-NPat
    #prediction
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    #plot
    fig = pl.figure() 
    # plot the decision function for each datapoint on the grid
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    pl.imshow(Z, interpolation='nearest',
          extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
          origin='lower', cmap=pl.cm.PuOr_r)
    contours = pl.contour(xx, yy, Z, levels=[0], linewidths=2,
                      linetypes='--')    
    pl.title("2D representation, controls in blue, comas in red")
    for i in range(NCont):
        pl.text(X[i,0],X[i,1],names[i],color = 'blue')
    for i in range(NCont,NCont+NPat):
        pl.text(X[i,0],X[i,1],names[i],color = 'red')
    pl.savefig(file_name)
    pl.close
