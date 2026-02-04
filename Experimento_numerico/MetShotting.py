#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  Jan 27 10:12:58 2026

@author: Fernando

PROBLEM
=======
Using shooting method for find periodic solutios of

x“=(-sen(x)-cx') dλ + g(t,x) dμ,

μ: algebraic sum of deltas

"""

#############################################
############ libraries  ##################### 
############################################# 


import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
import numpy as np
from numpy.matlib import repmat
import time
from itertools import product
from multiprocessing import Pool  # para paraleliza



####  Heaveside Function
def H(t,a):
    if t>a:
        return 1
    else:
        return 0

############## BV function generating the  singular measure  ###########  
g2=lambda t: .1*sum(H(t,j) for j in range(1,6))
g2=np.vectorize(g2)




############## BV function generating the Lebesgue measure  ###########

g1=lambda t: t
g1=np.vectorize(g1)


##interval and parameters
a=0.0
b=2*np.pi
c=0.0

## number of  times in the discretization
n=1000
#vector tiempos 
t=np.linspace(a,b,n)

# Measures of intervals vector 
dg2=g2(t[1:])-g2(t[:-1])
dg1=g1(t[1:])-g1(t[:-1]) 


 
# number iteration in algorithm
nro_iter=100


#################  FIEDS ##################

def f2(t,x):
    Y = np.zeros_like(x)
    Y[1, :] = np.maximum(1.9-np.sqrt(np.sum(x**2, axis=0)), Y[0, :])
    return Y


def f1(t,x):
    Y=np.zeros_like(x)
    Y[0,:]=x[1,:]
    Y[1,:]=-np.sin(x[0,:])-c*x[1,:]
    return Y


############  INTEGRALES  #########################
def stieltjes_integral(f, dg):
    result = np.cumsum(f[:,:-1].dot(np.diag(dg)),axis=1)  
    Z=np.zeros( (len(f[:,0]) , 1) )
    return np.concatenate((Z,result),axis=1)


############# Integral mediante la regla del trapecio ######

def trapesio_integral(f, dg):
    result = np.cumsum(((f[:,:-1]+f[:,1:])/2).dot(np.diag(dg)),axis=1)  
    Z=np.zeros( (len(f[:,0]) , 1) )
    return np.concatenate((Z,result),axis=1)



def IteraPicard_bis(f1,f2, x0, t, dg1, dg2,nro_iter ):
   # Inicilizo la iteración
    x0=np.array([[x0[0]],[x0[1]]])
    phi0=repmat(x0,1,len(t))
    phi=phi0 
    #iteracion
    error=1
    j_it=0
    umbral=0.00001 
    while error>umbral and j_it<nro_iter:
        f_phi1=f1(t,phi)
        f_phi2=f2(t,phi)
        I=stieltjes_integral(f_phi1, dg1)\
            +stieltjes_integral(f_phi2, dg2)
        phi1=phi0+I
        err=phi-phi1
        phi=phi1
        error=np.sum(err**2)
        j_it+=1
       
    
    return phi


def PoincareMap(x0):
    phi=IteraPicard_bis(f1,f2, x0,t, dg1, dg2,nro_iter )
    return [phi[0][-1],phi[1][-1]]


def Error_PM(x, y):
    x0 = [x, y]
    return (np.array(PoincareMap(x0))-np.array(x0)).dot(np.array(PoincareMap(x0))-np.array(x0))


Error_PM_vect = np.vectorize(Error_PM)


start_time = time.time()  # inicia reloj
# Espezor de la Malla
# esp=1
x0 = np.arange(-np.pi, np.pi, 0.1)  # np.arange(-np.pi,np.pi,.02)
v0 = np.arange(-2, 2, .1)
X0, V0 = np.meshgrid(x0, v0)

if __name__ == "__main__":
    with Pool(8) as pool:
        error = pool.starmap(Error_PM_vect, product(x0, v0))
error = np.reshape(error, np.shape(X0.T))
tarda = time.time() - start_time

###########################################
### BORRAR! SI SE SUBE AL SSH ############
#########################################


print("--- %s seconds ---" % (tarda))
print('---%s minutos' %(tarda/60))
print('---%s horas' %(tarda/3600))


#ax.set_aspect(aspect=1.5)
#fig.colorbar(s, ax=ax)
# A = np.load("ExperimentoMedidasE2.npz")
# X0 = A["X0"]
# V0 = A["V0"]
# error = A["error"]

#fig = plt.figure()
#ax = fig.add_axes([0.05, .1, .95, .8])
fig,ax = plt.subplots()
Z = error.T**.1
s = ax.imshow(Z, extent=[X0[0, 0], X0[0, -1], V0[0, 0], V0[-1, 0]], origin='lower',
               cmap='Greys')
fig.colorbar(s, ax=ax)
s=ax.contour(X0,V0,error.T,[0, .01, .1, .25, .5,.75, 1, 2], colors="k" ) 
ax.clabel(s, fontsize=10)   


Info="g2=lambda t: sum((-1)**j*H(t,j) for j in range(1,10))\n \
     c=2.0\n\
     def f2(t,x):\n\
         Y = np.zeros_like(x)\n\
         Y[1, :] = np.maximum(1.9-np.sqrt(np.sum(x**2, axis=0)), Y[0, :])\n\
         return Y\n\
       "

f=open("DataPendAmort.npz",'wb')
np.savez(f, X0=X0, V0=V0,error=error, Info=Info)














def Error_PM_opt(x):
    return np.sum((PoincareMap(x)-x)**2)
rangos=((-1,1),(-2,2))
















opt=minimize(Error_PM_opt,[0.26,0.56])#,bounds=rangos)
#opt=differential_evolution(Error_PM_opt,rangos,workers=8)
    
y0=opt["x"]

#y0=np.array([0,0])
def g2b(t):
    if t<=b:
        z=g2(t)
    else:
        z=g2(t-b)+g2(b)
    return z
g2b=np.vectorize(g2b)

t=np.linspace(a,2*b,10000 )
dg2=g2b(t[1:])-g2b(t[:-1])
dg1=g1(t[1:])-g1(t[:-1])  


phi=IteraPicard_bis(f1,f2, y0,t, dg1, dg2,nro_iter )
#fig1= plt.figure()
#ax1=fig1.add_axes([0.05,.1,.95,.8])
#ax.plot(phi[0,:],phi[1,:])
fig1= plt.figure()
ax1=fig1.add_axes([0.05,.1,.95,.8])
ax1.plot(t,phi.T, '.')


######################  Experimentos #######################################