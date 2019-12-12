import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib.pyplot import *
import pickle
from matplotlib.ticker import FormatStrFormatter
from fm import *
import time
rc('font',family='serif')

'''
#for dynesty & emcee
module purge
module load anaconda3/4.4.0


#for multinest
module purge
module load anaconda3/4.2.0
module load multinest/3.10

'''


#load crosssections between wnomin and wnomax (in cm-1)
xsects=xsects_JWST(2000,29000)

pdb.set_trace()
#the parameters
#planet/star system params--typically not free parameters in retrieval
Rp=1.036 #1.380#0.930#*x[4]# Planet radius in Jupiter Radii--this will be forced to be 10 bar radius--arbitrary (scaling to this is free par)
Rstar=0.667#1.162#0.598   #Stellar Radius in Solar Radii
M =2.034#0.714    #Mass in Jupiter Masses
D=0.01526#0.04747  #semimajor axis in AU

#TP profile params (3--Guillot 2010, Parmentier & Guillot 2013--see Line et al. 2013a for implementation)
Tirr=1400#0#1500#x[0]#544.54 #terminator **isothermal** temperature--if full redistribution this is equilibrium temp
logKir=-1.5  #TP profile IR opacity controlls the "vertical" location of the gradient
logg1=-0.7     #single channel Vis/IR opacity. Controls the delta T between deep T and TOA T
Tint=200.

#Composition parameters---assumes "chemically consistnat model" described in Kreidberg et al. 2015, but with Quenching like in Morley+2017
logMet=0.0#x[1]#1.5742E-2 #.   #Metallicity relative to solar log--solar is 0, 10x=1, 0.1x = -1 used -1.01*log10(M)+0.6
logCtoO=-0.26#x[2]#-1.97  #log C-to-O ratio: log solar is -0.26
logPQCarbon=-5.5  #CH4, CO, H2O Qunech pressure--forces CH4, CO, and H2O to constant value at quench pressure value
logPQNitrogen=-5.5  #N2, NH3 Quench pressure--forces N2 and NH3 to ""  --ad hoc for chemical kinetics--reasonable assumption

#A&M Cloud parameters--includes full multiple scattering (for realzz) in both reflected and emitted light
logKzz=7 #log Rayleigh Haze Amplitude (relative to H2)
fsed=2.0 #haze slope--4 is Rayeigh, 0 is "gray" or flat.  
logPbase=-1.0  #gray "large particle" cloud opacity (-35 - -25)
logCldVMR=-5.5 #cloud fraction

#simple 'grey+rayleigh' parameters--non scattering--just pure extinction
logKcld = -40
logRayAmp = -30
RaySlope = 0

#'''
#EMISSION
#data
wlgrid, y_meas, err=np.loadtxt('w43b_emis.txt').T

#seting up input state vector. Must be in this order as indicies are hard wired in fx inside fm
x=np.array([Tirr, logKir,logg1,Tint, logMet, logCtoO, logPQCarbon,logPQNitrogen, Rp, Rstar, M, D, logKzz, fsed,logPbase,logCldVMR, logKcld, logRayAmp, RaySlope])

#calling forward model
#thermochemical gas profile scaling factors
# 0   1    2    3   4    5    6     7    8    9   10    11   12   13    14   15   16   17   18  19 20   21
#H2O  CH4  CO  CO2 NH3  N2   HCN   H2S  PH3  C2H2 C2H6  Na    K   TiO   VO   FeH  H    H2   He   e- h-  mmw
gas_scale=np.array([1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1., 1., 1.]) #can be made free params if desired (won't affect mmw)#can be made free params if desired (won't affect mmw)
y_binned,y_mod,wnocrop,atm,Ftoa,Fstar,Fstar_TOA,Fup_therm,Fup_ref=fx_emis(x,wlgrid,gas_scale, xsects)  #returns model spectrum, wavenumber grid, and vertical 


ymin=1E-5
ymax=np.max(y_mod)*1E3*1.5
fig1, ax=subplots()
xlabel('$\lambda$ ($\mu$m)',fontsize=14)
ylabel('F$_p$/F$_{star}$ [$\\times 10^{-3}$]',fontsize=14)
minorticks_on()
errorbar(wlgrid, y_meas*1E3, yerr=err*1E3, xerr=None, fmt='Dk')
plot(wlgrid, y_binned*1E3,'ob')
plot(1E4/wnocrop, y_mod*1E3,color='black',label='Total')
#reflected component
plot(1E4/wnocrop, Fup_ref/Fstar*1E3*(Rp/Rstar*0.10279)**2 ,color='blue',label='Reflected Stellar')
#emission component
plot(1E4/wnocrop, Fup_therm/Fstar*1E3*(Rp/Rstar*0.10279)**2 ,color='red',label='Thermal Emission ')


legend(frameon=False)
subplots_adjust(left=0.2, right=0.9, top=0.8, bottom=0.2)
ax.set_xscale('log')
#ax.set_yscale('log')
ax.set_xticks([0.3, 0.5,0.8,1, 2, 3, 5])
ax.axis([0.3,5,ymin,ymax])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.tick_params(length=5,width=1,labelsize='large',which='major')
savefig('emission_spectrum.pdf',fmt='pdf')

show()
close()
#'''

#doing transmission spectrum
xRp=1.0#1.00035
#data
wlgrid, y_meas, err=np.loadtxt('w43b_trans.txt').T


x=np.array([Tirr, logKir,logg1,Tint, logMet, logCtoO, logPQCarbon,logPQNitrogen, Rp*xRp, Rstar, M, logKzz, fsed,logPbase,logCldVMR, logKcld, logRayAmp, RaySlope])
#thermochemical gas profile scaling factors
# 0   1    2    3   4    5    6     7    8    9   10    11   12   13    14   15   16   17   18  19 20   21
#H2O  CH4  CO  CO2 NH3  N2   HCN   H2S  PH3  C2H2 C2H6  Na    K   TiO   VO   FeH  H    H2   He   e- h-  mmw
gas_scale=np.array([1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1., 1., 1.]) #can be made free params if desired (won't affect mmw)

y_binned,y_mod,wnocrop,atm=fx_trans(x,wlgrid,gas_scale, xsects)  #returns model spectrum, wavenumber grid, and vertical abundance profiles from chemistry

ymin=np.min(y_mod)*1E2*0.99
ymax=np.max(y_mod)*1E2*1.01
fig1, ax=subplots()
xlabel('$\lambda$ ($\mu$m)',fontsize=14)
ylabel('(R$_{p}$/R$_{*}$)$^{2} [\%]$',fontsize=14)
minorticks_on()
plot(1E4/wnocrop, y_mod*1E2,color='black')
#errorbar(wlgrid, y_meas*1E2, yerr=err*1E2, xerr=None, fmt='Dk')
ax.set_xscale('log')
#ax.set_yscale('log')
ax.set_xticks([0.3, 0.5,0.8,1, 2, 3, 5])
ax.axis([0.3,5,ymin,ymax])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.tick_params(length=5,width=1,labelsize='large',which='major')

savefig('transmission_spectrum.pdf',fmt='pdf')
show()
close()

#pdb.set_trace()

#PLOTTING ATMOPHERE.............
#unpacking variables
#P is in bars
#T is in K
#H2O, CH4,CO,CO2,NH3,Na,K,TiO,VO,C2H2,HCN,H2S,FeH,H2,He are gas mixing ratio profiles
#qc is the condensate abundance profile given an "f_sed" value and cloud base pressure
#r_eff is the effective cloud droplet radius given (see A&M 2001 or Charnay et al. 2017)
#f_r is the mixing ratio array for each of the cloud droplet sizes.
P,T, H2O, CH4,CO,CO2,NH3,Na,K,TiO,VO,C2H2,HCN,H2S,FeH,H2,He,H,e, Hm,qc,r_eff,f_r=atm

'''
fig2, ax1=subplots()
#feel free to plot whatever you want here....
ax1.semilogx(H2O,P,'b',ls='--',lw=2,label='H2O')
ax1.semilogx(CH4,P,'black',ls='--',lw=2,label='CH4')
ax1.semilogx(CO,P,'g',ls='--',lw=2,label='CO')
ax1.semilogx(CO2,P,'orange',ls='--',lw=2,label='CO2')
ax1.semilogx(NH3,P,'darkblue',ls='--',lw=2,label='NH3')
ax1.semilogx(Na,P,'b',lw=2,label='Na')
ax1.semilogx(K,P,'g',lw=2,label='K')
ax1.semilogx(TiO,P,'k',lw=2,label='TiO')
ax1.semilogx(VO,P,'orange',lw=2,label='VO')
ax1.set_xlabel('Mixing Ratio',fontsize=20)
ax1.set_ylabel('Pressure [bar]',fontsize=20)
ax1.semilogy()
ax1.legend(loc=4,frameon=False)
ax1.axis([1E-9,1,100,1E-7])

#plotting TP profile on other x-axis
ax2=ax1.twiny()
ax2.semilogy(T,P,'r-',lw='4',label='TP')
ax2.set_xlabel('Temperature [K]',color='r',fontsize=20)
ax2.axis([0.8*T.min(),1.2*T.max(),100,1E-6])
for tl in ax2.get_xticklabels(): tl.set_color('r')
savefig('atmosphere.pdf',fmt='pdf')
show()
close()
'''



pdb.set_trace()






