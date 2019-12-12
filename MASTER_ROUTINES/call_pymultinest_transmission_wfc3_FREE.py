from __future__ import absolute_import, unicode_literals, print_function
import pymultinest
import math, os
from fm import *
import pdb
import numpy as np
import pickle
from matplotlib.pyplot import *



#output path--note this will be name of folder
#for multinest output and also the generated
#pickle 
outpath="./OUTPUT/pmn_transmission_wfc3_free"
if not os.path.exists(outpath): os.mkdir(outpath)  #creating folder to dump MultiNest output


#load crosssections between wnomin and wnomax
xsects=xsects_HST(5800,9100) #5800 cm-1 (1.72 um) to 9100 cm-1 (1.1 um)

# log-likelihood
def loglike(cube, ndim, nparams):

    #setting default parameters---will be fixed to these values unless replaced with 'theta'
    #planet/star system params--xRp is the "Rp" free parameter, M right now is fixed, but could be free param
    Rp= 1.036#0.930#*x[4]# Planet radius in Jupiter Radii--this will be forced to be 10 bar radius--arbitrary (scaling to this is free par)
    Rstar=0.667#0.598   #Stellar Radius in Solar Radii
    M =2.034#1.78    #Mass in Jupiter Masses

    #TP profile params (3--Guillot 2010, Parmentier & Guillot 2013--see Line et al. 2013a for implementation)
    Tirr=1400#1500#x[0]#544.54 #terminator **isothermal** temperature--if full redistribution this is equilibrium temp
    logKir=-5  #TP profile IR opacity controlls the "vertical" location of the gradient
    logg1=-0.0     #single channel Vis/IR opacity. Controls the delta T between deep T and TOA T
    Tint=0.
    
    #A&M Cloud parameters--includes full multiple scattering (for realzz) in both reflected and emitted light
    logKzz=7 #log Rayleigh Haze Amplitude (relative to H2)
    fsed=3.0 #haze slope--4 is Rayeigh, 0 is "gray" or flat.  
    logPbase=-1.0  #gray "large particle" cloud opacity (-35 - -25)
    logCldVMR=-25.0 #cloud fraction
    
    #simple 'grey+rayleigh' parameters--non scattering--just pure extinction
    logKcld = -40
    logRayAmp = -30
    RaySlope = 0
    
    H2O=-15.
    CH4=-15.
    CO=-15.  
    CO2=-15. 
    NH3=-15.  
    N2=-15.   
    HCN=-15.   
    H2S=-15.  
    PH3=-15.  
    C2H2=-15. 
    C2H6=-15. 
    Na=-15.    
    K=-15.   
    TiO=-15.   
    VO=-15.   
    FeH=-15.  
    H=-15.     
    em=-15. 
    hm=-15.

    
    #unpacking parameters to retrieve (these override the fixed values above)
    Tirr, H2O,CO,CO2,CH4, logKcld, xRp=cube[0],cube[1],cube[2],cube[3],cube[4],cube[5],cube[6]
   
    ##all values required by forward model go here--even if they are fixed
    x=np.array([Tirr, logKir,logg1, Tint,0,0,0,0, Rp*xRp, Rstar, M, logKzz, fsed,logPbase,logCldVMR, logKcld, logRayAmp, RaySlope])
    # 0   1    2    3   4    5    6     7    8    9   10    11   12   13    14   15   16   17   18  19 20   21
    #H2O  CH4  CO  CO2 NH3  N2   HCN   H2S  PH3  C2H2 C2H6  Na    K   TiO   VO   FeH  H    H2   He   e- h-  mmw
    gas_scale=np.array([H2O,CH4,CO,CO2,NH3,N2,HCN,H2S,PH3,C2H2,C2H6,Na,K,TiO,VO ,FeH,H,-50.,-50.,em, hm,-50.]) #
    foo=fx_trans_free(x,wlgrid,gas_scale,xsects)
    y_binned=foo[0]
    
    loglikelihood=-0.5*np.sum((y_meas-y_binned)**2/err**2)  #your typical "quadratic" or "chi-square"
    
    return loglikelihood


# prior transform
def prior(cube,ndim,nparams):
    #prior ranges...
    cube[0] = 1900 * cube[0] + 100  #Tiso
    cube[1]=12*cube[1]-12
    cube[2]=12*cube[2]-12
    cube[3]=12*cube[3]-12
    cube[4]=12*cube[4]-12
    cube[5]=20*cube[5]-45
    cube[6]=1*cube[6]+0.5

 

#####loading in data##########
wlgrid, y_meas, err=np.loadtxt('w43b_trans.txt').T
outname=outpath+'.pic'  #dynesty output file name (saved as a pickle)
Nparam=7  #number of parameters--make sure it is the same as what is in prior and loglike
Nlive=500 #number of nested sampling live points


#calling pymultinest
pymultinest.run(loglike, prior, Nparam, outputfiles_basename=outpath+'/template_',resume=False, verbose=True,n_live_points=Nlive, importance_nested_sampling=False)

#converting pymultinest output into pickle format (weighting sampling points by volumes to get "true" posterior)
a = pymultinest.Analyzer(n_params = Nparam, outputfiles_basename=outpath+'/template_')
s = a.get_stats()
output=a.get_equal_weighted_posterior()
pickle.dump(output,open(outname,"wb"))




