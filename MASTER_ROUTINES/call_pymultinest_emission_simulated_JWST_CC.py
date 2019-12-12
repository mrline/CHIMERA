from __future__ import absolute_import, unicode_literals, print_function
import pymultinest
import math, os
from fm import *
import pdb
import numpy as np
import pickle
from matplotlib.pyplot import *
outpath="./OUTPUT/pmn_transmission_jwst_cc"
if not os.path.exists(outpath): os.mkdir(outpath)  #creating folder to dump MultiNest output



#load crosssections between wnomin and wnomax
xsects=xsects_JWST(750, 15000)  #5800 cm-1 (1.72 um) to 9100 cm-1 (1.1 um)

# log-likelihood
def loglike(cube, ndim, nparams):

    #setting default parameters---will be fixed to these values unless replaced with 'theta'
    #planet/star system params--typically not free parameters in retrieval
    Rp= 1.036#0.930#*x[4]# Planet radius in Jupiter Radii--this will be forced to be 10 bar radius--arbitrary (scaling to this is free par)
    Rstar=0.667#0.598   #Stellar Radius in Solar Radii
    M =2.034#1.78    #Mass in Jupiter Masses
    D=0.01562  #semimajor axis in AU
    #TP profile params (3--Guillot 2010, Parmentier & Guillot 2013--see Line et al. 2013a for implementation)
    Tirr=1400#1500#x[0]#544.54 #terminator **isothermal** temperature--if full redistribution this is equilibrium temp
    logKir=-0.7  #TP profile IR opacity controlls the "vertical" location of the gradient
    logg1=-0.6     #single channel Vis/IR opacity. Controls the delta T between deep T and TOA T
    Tint=200.
    #Composition parameters---assumes "chemically consistnat model" described in Kreidberg et al. 2015
    logMet=0.0#x[1]#1.5742E-2 #.   #Metallicity relative to solar log--solar is 0, 10x=1, 0.1x = -1 used -1.01*log10(M)+0.6
    logCtoO=-0.26#x[2]#-1.97  #log C-to-O ratio: log solar is -0.26
    logPQCarbon=-5.5  #CH4, CO, H2O Qunech pressure--forces CH4, CO, and H2O to constant value at quench pressure value
    logPQNitrogen=-5.5  #N2, NH3 Quench pressure--forces N2 and NH3 to ""  --ad hoc for chemical kinetics--reasonable assumption
    #A&M Cloud parameters--includes full multiple scattering (for realzz) in both reflected and emitted light
    logKzz=7 #log Rayleigh Haze Amplitude (relative to H2)
    fsed=3.0 #haze slope--4 is Rayeigh, 0 is "gray" or flat.  
    logPbase=-1.0  #gray "large particle" cloud opacity (-35 - -25)
    logCldVMR=-25.0 #cloud fraction
    #simple 'grey+rayleigh' parameters--non scattering--just pure extinction
    logKcld = -40
    logRayAmp = -30
    RaySlope = 0

    #unpacking parameters to retrieve
    Tirr, logKir,logg1, logMet, logCtoO, logKzz, fsed ,logPbase,logCldVMR=cube[0],cube[1],cube[2],cube[3],cube[4],cube[5],cube[6],cube[7],cube[8]

    ##all values required by forward model go here--even if they are fixed
    x=np.array([Tirr, logKir,logg1, Tint,logMet, logCtoO, logPQCarbon,logPQNitrogen, Rp, Rstar, M, D, logKzz, fsed,logPbase,logCldVMR, logKcld, logRayAmp, RaySlope])
    gas_scale=np.array([1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1., 1., 1.]) #can be made free params if desired (won't affect mmw)#can be made free
    wlgrid=-1
    foo=fx_emis(x,wlgrid,gas_scale,xsects)  #note the call to fx_emis here...
    y_binned=foo[0]

    #y_mod=y_meas+x[1]
    loglikelihood=-0.5*np.sum((y_meas-y_binned)**2/err**2)
    return loglikelihood


# prior transform
def prior(cube,ndim,nparams):
    #prior ranges...
    cube[0] = 1400 * cube[0] + 400  
    cube[1] = 3*cube[1]-3.0
    cube[2] = 4*cube[2]-2.0
    cube[3] = 5*cube[3]-2
    cube[4] = 2.3*cube[4]-2
    cube[5] = 6*cube[5]+5
    cube[6] = 5.5*cube[6]+0.5
    cube[7] = 7.5*cube[7]-6.0
    cube[8] = 8*cube[8]-10



#####loading in data##########
junk, y_meas, err=np.loadtxt('simulated_emission_JWST.txt').T
outname=outpath+'.pic'  #dynesty output file name (saved as a pickle)
Nparam=9  #number of parameters--make sure it is the same as what is in prior and loglike
Nlive=500 #number of nested sampling live points

#calling pymultinest
pymultinest.run(loglike, prior, Nparam, outputfiles_basename=outpath+'/template_',resume=False, verbose=True,n_live_points=Nlive, importance_nested_sampling=False)

#converting pymultinest output into pickle format (weighting sampling points by volumes to get "true" posterior)
a = pymultinest.Analyzer(n_params = Nparam, outputfiles_basename=outpath+'/template_')
s = a.get_stats()
output=a.get_equal_weighted_posterior()
pickle.dump(output,open(outname,"wb"))




pdb.set_trace()



