# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 09:35:25 2013

@author: Felix Renard
felixrenard@gmail.com
"""


import numpy as np
import scipy.optimize as opt
import sklearn as sk
import pandas as pd
#The three optimisation functions

def f_coord1(x,X2,cov2,alpha2,dist2):
    X2 = np.vstack([X2,x])
    cov2 = cov2.reshape(X2.shape)
    X_tmp2 = np.hstack((alpha2*cov2,X2))
    D2 = sk.metrics.pairwise_distances(X_tmp2)
    return np.linalg.norm(dist2-D2)

def f_glob(X2,cov2,alpha2,dist2):
    cov2 = cov2.reshape(X2.shape)
    X_tmp2 = np.vstack((alpha2*cov2,X2)).T
    D2 = sk.metrics.pairwise_distances(X_tmp2)
    return np.linalg.norm(dist2-D2)

def f_alpha(alpha,X2,cov2,dist2):
    cov2 = cov2.reshape(X2.shape)
    X_tmp = np.vstack((alpha*cov2,X2))
    D2 = sk.metrics.pairwise_distances(X_tmp.T)
    return np.linalg.norm(dist2-D2)

# Read the dataset - the graph metrics of the different subject
# 
X = pd.read_csv("file.csv",index = False)
cov = pd.read_csv("file_covariate.csv", index = False)

# Speficic for the comatose case
# Labels of the subject
Indiv = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12',
             'C13','C14','C15','C16','C17','C18','C19','P0','P3','P7','P8','P9',
             'P10','P11','P13','P15','P16','P18','P19','P20','P22','P24']
# Classification label, 0 for the controls, 1 for the patients
l = np.zeros(35)
l[20:] = 1


#Estimation of the Euclidean distance between subject in the natural space
D = sk.metrics.pairwise_distances(X)

#initialisation aleatoire
#list_extract = sk.utils.shuffle(np.arange(X.shape[0]))

#initialize with the most remote couple of points
ll = []
ll.append(np.where(D==np.max(D))[0][0])
ll.append(np.where(D==np.max(D))[1][0])

# Add sequentially the furthest point of the previous set
while(len(ll)!=D.shape[0]):
    D_tmp = D[ll]
    D_tmp[:,ll] = 0
    ll.append(np.where(D_tmp == np.max(D_tmp))[1][0])

list_extract = np.array(ll)

# Initialize alpha parameter with the classic Isomap
iso = sk.manifold.isomap.Isomap(n_components=2,n_neighbors=4)
iso.fit(X)
tmp = iso.embedding_[:,0]
dist = iso.dist_matrix_
delta_1 = np.max(tmp) -np.min(tmp) 
delta_2 = np.max(cov) -np.min(cov)
alpha = delta_1/delta_2

#Initialize the first point at 0
X_tmp = np.zeros([1])

for i in range(1,X.shape[0]):
    print "one coord",i
    cov_tmp = cov[list_extract[0:i+1]]
    dist_tmp = dist[list_extract[0:i+1]][:,list_extract[0:i+1]]
    opt_test = 10000    
    #Multi-start opitmisation
    for j in range(-5,5):
        xtmptmp,B,tmp,tmp1,tmp3 = opt.fmin(f_coord1,j,(X_tmp,cov_tmp,alpha,dist_tmp),xtol=0.000000001, ftol=0.000000001,maxiter=1000000,maxfun=1000000,disp = 0 , full_output = 1)
        if opt_test>B:
            xtmp = xtmptmp
            opt_test = B
            print j
    X_tmp = np.vstack([X_tmp, xtmp])
# We ca add an additional global optimisation before the estimation of the alpha coefficient
#    X_tmp2,B,tmp,tmp1,tmp3 = opt.fmin(f_glob,X_tmp,(cov_tmp,alpha,dist_tmp), xtol=0.000000001, ftol=0.000000001,maxiter=1000000,maxfun=1000000,disp = 1 , full_output = 1)
#    print "all_coord"
#    X_tmp = X_tmp2.reshape(-1,1)
    alpha,B,tmp,tmp1,tmp3 = opt.fmin(f_alpha,alpha,(X_tmp,cov_tmp,dist_tmp), xtol=0.000000001, ftol=0.00000001,disp = 1 , full_output = 1)
    X_tmp2,B,tmp,tmp1,tmp3 = opt.fmin(f_glob,X_tmp,(cov_tmp,alpha,dist_tmp), xtol=0.000000000001, ftol=0.00000000001,maxiter=1000000,maxfun=1000000,disp = 1 , full_output = 1)
    print "all_coord"
    X_tmp = X_tmp2.reshape(-1,1)
    
# Merge the covariate modulated by alpha and the first component
cov2 = cov.reshape(X_tmp.shape)
X_iso = np.hstack((alpha*cov2[list_extract],X_tmp))

# Re-order the rows of the matrix to correspond to the initial order of the subject
I = np.argsort(list_extract)
X_iso_f = X_iso[I]  


###### Visualization

Xt =  cov
scaling = 5

from scipy.interpolate import griddata
import PlotFunction as pf

grid_x , grid_y = np.mgrid[-1:1:1000j,-1:1:1000j]*scaling
grid_lin = griddata(X_iso_f,Xt,(grid_x,grid_y),method='linear')
grid_lin = grid_lin.reshape(1000,1000)
pf.scatter_2D(X_iso_f,l,Indiv)

#plt.imshow(grid_lin.T, extent=(-1,1,-1,1), origin='lower')
plt.imshow(grid_lin.T,extent=(-scaling,scaling,-scaling,scaling), origin='lower')
plt.colorbar().ax.tick_params(labelsize=20)
plt.title('')

plt.tick_params(labelsize=20)
