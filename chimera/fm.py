import math
import numpy as np
import scipy as sp
from array import *
from scipy import interpolate
from scipy import signal
from scipy import special
from scipy import interp
from scipy import ndimage
import pdb
from pickle import *
from numba import jit
import time
from scipy.interpolate import RegularGridInterpolator
import h5py
from scipy.linalg import solve_banded
import pdb
import scipy
import datetime
import os
import pysynphot as psyn
### stellar spec ### 

def make_stellar(temp, logMH, logg, database, outfile):
    """
    Make stellar using pysynphot

    Parameters
    ----------
    temp : float 
        temperature 
    logMH : float 
        log stellar metallicity 
    logg : float 
        stellar logg
    database : str 
        Stellar database to choose from 
    outfile : str 
        stellar h5 output file

    Returns
    -------
    N/A just output file
    """
    sp = psyn.Icat(database, temp, logMH, logg)
    sp.convert('flam') ## initial units erg/cm^2/s/ Angst
    wave=sp.wave*1e-10  #in meters
    fluxmks = sp.flux*1e-7*1e4*1e10 #in W/m2/m
    f = h5py.File(outfile,'w')
    f.create_dataset('lambdastar',data=wave)
    f.create_dataset('Fstar0',data=fluxmks)
    f.close()


##############################   OPACITY ROUTINES  #################################
####################################################################################
####################################################################################

def xsects(wnomin, wnomax, observatory, directory, stellar_file=None,
    cond_name='MgSiO3',gauss_pts='default'):
    """Get correlated-K opacities, stellar spectrum, and chemistry 

    Routine that loads in the correlated-K opacities
    applicable to JWST. Here we assume the data will be binned 
    to an R=100 and span 50 - 28000 cm-1 (200 - 0.3 um)
    Also loads in Mie scattering properties, stellar spectrum
    for emission, and grid chemistry.  Note, to change Mie condensate, 
    change the name in this routine.  Have a look in ../ABSCOEFF_CK/MIES/
    for a current list of condensates.  Feel free to also
    load in a different stellar spectrum (for emission). 
    Does not matter where you get it from just so long as you can
    save it as a .h5 file in wavelength (from low to high)
    and flux, both in MKS units (m, W/m2/m)


    Parameters
    ----------
    wnomin : float 
        minimum wavenumber for computation (min 50 for JWST, 2000 for HST)
    wnomax : float 
        maximum wavenumber for computation (max 28000 for JWST, 30000 for HST)
    observatory : str 
        Options are 'JWST' or 'HST'
    directory : str
        Directory path to zip file from 
        https://www.dropbox.com/sh/o4p3f8ukpfl0wg6/AADBeGuOfFLo38MGWZ8oFDX2a?dl=0&lst=
    stellar_file : str, optional 
        Stellar file for emission spectra
    cond_name : str , optional
        Name of condensate to compute cloud, if using Ackerman Marley cloud scheme
    gauss_pts : str , optional
        Number of gauss points for ck opacity tables. Default is to use 
        10 for HST and 20 for JWST. 
        Options are : ['default', '10', '20']


    Returns
    -------
    P
        pressure grid for C-K-coefficient relaevant properties

    T
        temperature grid for C-K-coefficient relaevant properties (not the same as chemistry)
    wno
        wavenumber grid of xsecs (here, R=100)
    g
        CK gauss quad g-ordinate
    wts
        CK gauss quad weights
    xsecarr 
        Pre-computed grid of CK coefficients as a function of
        Gases x Pressure x Temperature x Wavenumber x GaussQuad Points. 
    radius
        Mie scattering/condensate relevant properties. 
        Condensate particle sizes in micron (0.01 - 316 um)
    mies_arr
        mie properties array.  It contains the total extinction
        cross-section (first element--Qext*pi*r^2), single scatter albedo (second,
        Qs/Qext),and assymetry parameter (third) as a function of condensate (in
        this case, just one), particle radius (0.01 - 316 um), and wavenumber. These
        were generated offline with the "pymiecoated" routine using
        the indicies of refraction given in Wakeford & Sing 2017.
        (Condensates x MieProperties x size bins x wavenumber)
    Fstar 
        Stellar Flux spectrum binned to cross-section wavenumber grid
    logCtoO
        chemistry C/O grid in log10 (-2.0 - +0.3, -0.26 = solar)
    logMet
        chemistry metallicity grid in log10 (-2 - +3, 0=solar)
    Tarr
        Chemistry Temperature grid (400 - 3400K)
    logParr
        Chemistry Pressure grid (-7.0 - 2.4, log10 in bar)
    gases
        log of Chemistry grid of molecular gas abundances (and 
        mean molecular weight) as a function of Metallicity, C/O, T, and P.
        (CtoO x logMet x Tarr x Gases x logParr). Gases: H2O  CH4  CO  CO2 NH3  N2  
        HCN   H2S  PH3  C2H2 C2H6  Na    K   TiO   VO   FeH  H    H2   He   e- h-  mmw. 
        This grid was generated with NASA CEA2 routine and assumes pure
        equilibrium+condensate chemistry (no rainout here).
    """
    ### Read in CK arrays (can switch between 10 & 20 CK gp's )

    
    available_opacities = ['H2H2', 'H2He', 'H2O','CH4','CO','CO2','NH3',
                          'Na_allard', 'K_allard','TiO','VO','C2H2','HCN',
                          'H2S','FeH','HMFF','HMBF']

    #set up filename tag 
    if observatory == 'HST':
        if gauss_pts == 'default': 
            gp = '10'
        elif gauss_pts in ['10','20']:
            gp = gauss_pts
        else: 
            raise Exception('Invalid gauss point choice.')

        filename = '_CK_STIS_WFC3_'+gp+'gp_2000_30000wno.h5'
        filename_MIE = '_r_0.01_300um_wl_0.3_200um_interp_STIS_WFC3_2000_30000wno.h5'

    elif observatory =='JWST':
        if gauss_pts == 'default': 
            gp = '20'
        elif gauss_pts in ['10','20']:
            gp = gauss_pts
        else: 
            raise Exception('Invalid gauss point choice.')
        filename = '_CK_R100_'+gp+'gp_50_30000wno.h5'
        filename_MIE = '_r_0.01_300um_wl_0.3_200um_interp_R100_20gp_50_30000wno.h5'

    else: 
        raise Exception ('Pick a valid observatory: HST or JWST')

    #set up cross section array list 
    xsecarr = []
    for mol in available_opacities: 
        file=os.path.join(directory,mol+filename)
        
        #open up h5 file
        hf=h5py.File(file, 'r')

        #get parameters that are uniform for everything
        if mol == 'H2H2':
            wno=np.array(hf['wno'])
            T=np.array(hf['T'])
            P=np.array(hf['P'])
            g=np.array(hf['g'])
            wts=np.array(hf['wts'])

        #get kcoefficients for each molecule
        kcoeff=np.array(hf['kcoeff'])
        xsecarr_mol=10**(kcoeff-4.)
        hf.close()

        #add it to master list 
        xsecarr += [xsecarr_mol]

    #super big array stuffing all the gases in    
    xsecarr = np.log10(np.array(xsecarr))



    #stellar flux file------------------------------------
    if stellar_file != None:
        hf=h5py.File(stellar_file, 'r')  #user should generate their own stellar spectrum from whatever database they choose, save as .h5 
        lambdastar=np.array(hf['lambdastar'])  #in MKS (meters)
        Fstar0=np.array(hf['Fstar0'])  #in MKS (W/m2/m)
        hf.close()

        lambdastar=lambdastar*1E6
        loc=np.where((lambdastar >= 1E4/wno[-1]) & (lambdastar <=1E4/wno[0]))
        lambdastar=lambdastar[loc]
        lambdastar_hi=np.arange(lambdastar.min(),lambdastar.max(),0.0001)
        Fstar0=Fstar0[loc]
        Fstar0=interp(np.log10(lambdastar_hi), np.log10(lambdastar), Fstar0) 

        #smooth stellar spectrum to CK bins
        szmod=len(wno)
        Fstar_smooth=np.zeros(szmod)
        dwno=wno[1:]-wno[:-1]
        for i in range(szmod-1):
            i=i+1
            loc=np.where((1E4/lambdastar_hi >= wno[i]-0.5*dwno[i-1]) & (1E4/lambdastar_hi < wno[i]+0.5*dwno[i-1]))
            Fstar_smooth[i]=np.mean(Fstar0[loc])

        Fstar_smooth[0]=Fstar_smooth[1]
        Fstar_smooth[-1]=Fstar_smooth[-2]
        Fstar=Fstar_smooth
    else: 
        Fstar = [np.nan]

    #loading mie coefficients-----------------------------
    file=os.path.join(directory, 'MIE_COEFFS',cond_name+filename_MIE)
    hf=h5py.File(file, 'r')
    wno_M=np.array(hf['wno_M'])
    radius=np.array(hf['radius'])
    Mies=np.array(hf['Mies'])
    hf.close()
  
    SSA=Mies[1,:,:]/Mies[0,:,:]#single scatter albedo
    Mies[1,:,:]=SSA  #single scatter albedo
    Mg2SiO4=Mies  #Mies = Qext, Qs, asym
    xxsec=Mg2SiO4[0,:,:].T*np.pi*radius**2*1E-12 #scattering cross-section
    Mg2SiO4[0,:,:]=xxsec.T
    mies_arr=np.array([Mg2SiO4])

    #cropping in wavenumber 
    loc=np.where((wno <= wnomax) & (wno >= wnomin))[0]
    wno=wno[loc]
    xsecarr=xsecarr[:,:,:,loc,:]
    mies_arr=mies_arr[:,:,:,loc]    
    if stellar_file != None: 
        Fstar=Fstar[loc]
    else: 
        Fstar = np.array(Fstar*len(loc))

    #loading interpolatedable chemistry grid as a function of C/O, Metallicity, T, and P (for many gases)
    hf=h5py.File(os.path.join(directory,'CHEM','chem_grid.h5'), 'r')
    logCtoO=np.array(hf['logCtoO'])  #-2.0 - 0.3
    logMet=np.array(hf['logMet']) #-2.0 - 3.0
    Tarr=np.array(hf['Tarr'])  #400 - 3400
    logParr=np.array(hf['logParr'])   #-7.0 - 2.4
    gases=np.array(hf['gases'])  ##H2O  CH4  CO  CO2 NH3  N2   HCN   H2S  PH3  C2H2 C2H6  Na    K   TiO   VO   FeH  H    H2   He   e- h-  mmw
    hf.close()

    print('Cross-sections Loaded')
    return P,T,wno,g,wts,xsecarr,radius*1E-6,mies_arr,Fstar,logCtoO, logMet, Tarr, logParr, np.log10(gases)


@jit(nopython=True)
def kcoeff_interp(logPgrid, logTgrid, logPatm, logTatm, wnogrid, kcoeff):
    """
    This routine interpolates the correlated-K tables
    to the appropriate atmospheric P & T for each wavenumber and 
    g-ordinate for each gas. It uses a standard bi-linear 
    interpolation scheme.
     
    Parameters
    ---------- 
    logPgrid : ndarray
        pressure grid (log10) on which the CK coeff's are pre-computed (Npressure points)
    logTgrid : ndarray
        temperature grid (log10) on which the CK coeffs are pre-computed (Ntemperature points)
    logPatm : ndarray 
        atmospheric pressure grid (log10) 
    logTatm : ndarray 
        atmospheric temperature grid (log10)
    wnogrid : ndarray 
        CK wavenumber grid (Nwavenumber points) (actually, this doesn't need to be passed...it does nothing here...)
    kcoeff : ndarray 
        massive CK coefficient array (in log10)--Ngas x Npressure x Ntemperature x Nwavenumbers x Ngordinates
    
    Returns
    ------- 
    kcoeff_int 
        the interpolated-to-atmosphere CK coefficients (in log). 
        This will be Nlayers x Nwavenumber x Ngas x Ngordiantes
    """

    Ng, NP, NT, Nwno, Nord=kcoeff.shape
    Natm=len(logTatm)
    kcoeff_int=np.zeros((Natm,Nwno,Ng,Nord))

    for i in range(Natm):  #looping through atmospheric layers

        y=logPatm[i]
        x=logTatm[i]

        p_ind_hi=np.where(logPgrid>=y)[0][0]
        p_ind_low=np.where(logPgrid<y)[0][-1]
        T_ind_hi=np.where(logTgrid>=x)[0][0]
        T_ind_low=np.where(logTgrid<x)[0][-1]

        y2=logPgrid[p_ind_hi]
        y1=logPgrid[p_ind_low]
        x2=logTgrid[T_ind_hi]
        x1=logTgrid[T_ind_low]
    
        for j in range(Ng): #looping through gases
            for k in range(Nwno): #looping through wavenumber
                for l in range(Nord): #looping through g-ord
                    arr=kcoeff[j,:,:,k,l]
                    Q11=arr[p_ind_low,T_ind_low]
                    Q12=arr[p_ind_hi,T_ind_low]
                    Q22=arr[p_ind_hi,T_ind_hi]
                    Q21=arr[p_ind_low,T_ind_hi]
                    fxy1=(x2-x)/(x2-x1)*Q11+(x-x1)/(x2-x1)*Q21
                    fxy2=(x2-x)/(x2-x1)*Q12+(x-x1)/(x2-x1)*Q22
                    fxy=(y2-y)/(y2-y1)*fxy1 + (y-y1)/(y2-y1)*fxy2
                    kcoeff_int[i,k,j,l]=fxy

    return kcoeff_int


@jit(nopython=True)
def mix_two_gas_CK(k1,k2,VMR1,VMR2,gord, wts):
    """
    Key function that properly mixes the CK coefficients
    for two individual gases via the "resort-rebin" procedure
    as descrbied in Lacis & Oinas 1991, Molliere et al. 2015 and 
    Amundsen et al. 2017. Each pair of gases can be treated as a
    new "hybrid" gas that can then be mixed again with another
    gas. That is the resort-rebin magic.  This is all for a *single*
    wavenumber bin for a single pair of gases.

    Parameters
    ---------- 
    k1 : ndarray 
        k-coeffs for gas 1 (on Nk ordinates)
    k2 : ndarray 
        k-coeffs for gas 2
    VMR1 : ndarray 
        volume mixing ratio of gas 1
    VMR2 : ndarray 
        volume mixing ratio for gas 2
    gord : ndarray
        g-ordinate array for gauss. quad.
    wts : ndarray 
        gauss quadrature wts--same for both gases
    Returns
    -------
    kmix_bin
        mixed CK coefficients for the given pair of gases
    VMR
        volume mixing ratio of "mixed gas".
    """
    VMR=VMR1+VMR2  #"new" VMR is sum of individual VMR's
    Nk=len(wts)
    kmix=np.zeros(Nk**2)   #Nk^2 mixed k-coeff array
    wtsmix=np.zeros(Nk**2) #Nk^2 mixed weights array
    #mixing two gases weighting by their relative VMR
    for i in range(Nk):
        for j in range(Nk):
            kmix[i*Nk+j]=(VMR1*k1[i]+VMR2*k2[j])/VMR #equation 9 Amundsen 2017 (equation 20 Mollier 2015)
            wtsmix[i*Nk+j]=wts[i]*wts[j]    #equation 10 Amundsen 2017

    #resort-rebin procedure--see Amundsen et al. 2016 or section B.2.1 in Molliere et al. 2015
    sort_indicies=np.argsort(kmix)  #sort new "mixed" k-coeff's from low to high--these are indicies
    kmix_sort=kmix[sort_indicies]  #sort k-coeffs from low to high
    wtsmix_sort=wtsmix[sort_indicies]  #sort mixed weights using same indicie mapping from sorted mixed k-coeffs
    #combining w/weights--see description on Molliere et al. 2015--not sure why this works..similar to Amundson 2016 weighted avg?
    int=np.cumsum(wtsmix_sort)
    x=int/np.max(int)*2.-1
    logkmix=np.log10(kmix_sort)
    #kmix_bin=10**np.interp(gord,x,logkmix)  #interpolating via cumulative sum of sorted weights...
    kmix_bin=np.zeros(len(gord))
    for i in range(len(gord)):
        loc=np.where(x >= gord[i])[0][0]
        kmix_bin[i]=10**logkmix[loc]

    return kmix_bin, VMR


@jit(nopython=True)
def mix_multi_gas_CK(CK,VMR,gord, wts):
    """
    Key function that properly mixes the CK coefficients
    for multiple gases by treating a pair of gases at a time.
    Each pair becomes a "hybrid" gas that can be mixed in a pair
    with another gas, succesively. This is performed at a given
    wavenumber and atmospheric layer.
    
    Parameters
    ----------
    CK : ndarray 
        array of CK-coeffs for each gas: Ngas x nordinates at a given wavenumber and pressure level
    VMR : ndarray 
        array of mixing ratios for Ngas.
    gord : ndarray 
        g-ordinates
    wts : ndarray 
        gauss quadrature wts--same for both gases
    
    Returns
    -------
    kmix_bin 
        mixed CK coefficients for the given pair of gases
    VMR 
        Volume mixing ratio of "mixed gas".
    """
    ngas=CK.shape[0]
    #begin by mixing first two gases
    kmix,VMRmix=mix_two_gas_CK(CK[0,:],CK[1,:],VMR[0],VMR[1],gord,wts)
    loc1=np.where((VMR > 1E-12) & (CK[:,-1] > 1E-50))[0]
    ngas=len(loc1)
    #mixing in rest of gases inside a loop
    for j in range(2,ngas):
        kmix,VMRmix=mix_two_gas_CK(kmix,CK[loc1[j],:],VMRmix,VMR[loc1[j]],gord,wts)
    #kmix,VMRmix=mix_two_gas_CK(kmix,CK[j,:],VMRmix,VMR[j],gord,wts)
    
    
    return kmix, VMRmix


@jit(nopython=True)
def compute_tau(CK,xsecContinuum,Mies, mass_path, Fractions,Fractions_Continuum, Fractions_Cond, gord, wts):  #this is the bottleneck right here...
    """
    Key function that computes the layer optical depths
    at each wavenumber,and g-ordiante. It also does the confusing mixing
    for the single-scatter abledo and asymetry parameter by
    appropriately weighting each by the scattering/extincition
    optical depths.  Each g-bin is treated like a psuedo"wavenumber" bin.
    Check out: https://spacescience.arc.nasa.gov/mars-climate-modeling-group/brief.html

    Parameters
    ----------
    CK : ndarray 
        array of interpolated-to-atmosphere grid CK-coeffs for each gas (Nlayers x Nwavenumbers x Ngas x Ngordinates)
    xsecContinuum : ndarray 
        CK coefficients for continuum gases, here just the rayleigh scattering opacities. Each g-bin is 'flat'
    Mies : ndarray 
        Condensate Mie scattering extenction cross sections (e.g., Qe*pi*r^2) for each condensate, particle size, wavenumber
    wts : ndarray 
        gauss quadrature wts--same for both gases

    Returns
    -------
    kmix_bin 
        mixed CK coefficients for the given pair of gases
    VMR
        Volume mixing ratio of "mixed gas".
    """
    Nlay=CK.shape[0]
    Nord=CK.shape[2]
    dtau_gas=np.zeros((Nlay,Nord))
    dtau_cont=np.zeros((Nlay,Nord))
    dtau_cond=np.zeros((Nlay,Nord))
    ssa=np.zeros((Nlay,Nord))
    asym=np.zeros((Nlay,Nord))
    
    for j in range(Nlay):
        k, VMR=mix_multi_gas_CK(CK[j,:,:],Fractions[:,j],gord,wts)
        dtau_gas[j,:]=VMR*k*mass_path[j]
        
        #add continuum opacities here--just add to dtau linearly
        weighted_xsec=np.sum(xsecContinuum[j,:]*Fractions_Continuum[:,j])  #summing xsecs x abundances
        xsec_cont=np.zeros(Nord)+weighted_xsec  #crating array that is size of n-ordiantes to fill with continuum xsecs (flat k-dist)
        dtau_cont[j,:]=xsec_cont*mass_path[j]   #computing continuum optical depth
        
        #everything condensate scattering here!!
        #total extinction cross-section of condensates (in g-space)
        weighted_cond_xsec=np.sum(Fractions_Cond[:,j]*Mies[0,0,:])
        xsec_cond=np.zeros(Nord)+weighted_cond_xsec  #crating array that is size of n-ordiantes to fill with continuum xsecs (flat k-dist)
        dtau_cond[j,:]=xsec_cond*mass_path[j]
        
        #weighted ssa and asym--this is hard--espceially after 2 beers....it's a weird "weighted in g-space ssa"
        #ssa
        weighted_ssa=np.sum(Fractions_Cond[:,j]*Mies[0,1,:]*Mies[0,0,:])  #what did I do here???
        ssa_cond=np.zeros(Nord)+weighted_ssa
        ssa[j,:]=(dtau_cont[j,:]*0.999999+ssa_cond*mass_path[j])/(dtau_cont[j,:]+dtau_cond[j,:]+dtau_gas[j,:])
        #(gotta keep that ray scattering under control..thus the 0.99999)
        #asym, aka <cos(theta)>
        weighted_asym=np.sum(Fractions_Cond[:,j]*Mies[0,0,:]*Mies[0,2,:]*Mies[0,1,:])
        asym_cond=np.zeros(Nord)+weighted_asym
        asym[j,:]=(asym_cond*mass_path[j])/(ssa_cond*mass_path[j]+dtau_cont[j,:])
    
    #print '--------------'
    #print ssa[:,0]
    
    dtau=dtau_cont+dtau_gas+dtau_cond*1.
    return dtau,dtau_cond,dtau_gas, ssa*1., asym*1.


##############################   EMISSION ROUTINES  ################################
####################################################################################
####################################################################################

@jit(nopython=True)
def blackbody(T,wl):
    """
    This function takes in a temperature (T) and a
    wavelength grid (wl) and returns a blackbody flux grid.
    This is used to compute the layer/slab "emission". All in MKS units.

    Parameters
    ---------- 
    T : float 
        temperature of blackbody in kelvin
    wl : ndarray 
        wavelength grid in meters

    Returns
    -------
    B : ndarray 
        an array of blackbody fluxes (W/m2/m/ster) at each wavelength (size Nwavelengths)
    """
    # Define constants used in calculation
    h = 6.626E-34
    c = 3.0E8
    k = 1.38E-23
    
    # Calculate Blackbody Flux (B) at each wavelength point (wl)
    B = ((2.0*h*c**2.0)/(wl**5.0))*(1.0/(sp.exp((h*c)/(wl*k*T)) - 1.0))
    
    # Return blackbody flux
    return B


@jit(nopython = True)
def tri_diag_solve(l, a, b, c, d):
    '''
    Tri-diagnoal matrix inversion solver. This is used
    for the two-stream radiative transver matrix inversions
    to solve for the boundary-condition coefficents on the layer
    interfaces.  
    A, B, C and D refer to: A(I)*X(I-1) + B(I)*X(I) + C(I)*X(I+1) = D(I)

    Parameters
    ---------- 
    L : int 
        array size
    A : array or list
    B : array or list
    C : array or list
    C : array or list

    Returns
    -------
    solution coefficients, XK
    '''   
    AS, DS, CS, DS,XK = np.zeros(l), np.zeros(l), np.zeros(l), np.zeros(l), np.zeros(l) # copy arrays
    
    AS[-1] = a[-1]/b[-1]
    DS[-1] = d[-1]/b[-1]
    
    for i in range(l-2, -1, -1):
        x = 1.0 / (b[i] - c[i] * AS[i+1])
        AS[i] = a[i] * x
        DS[i] = (d[i]-c[i] * DS[i+1]) * x
    XK[0] = DS[0]
    for i in range(1,l):
        XK[i] = DS[i] - AS[i] * XK[i-1]
    return XK


@jit(nopython=True)
def src_func_loop(B,tautop,Bsurf,ubari,nlay,lam,dtau,taump,taut,ssa,hg,k1,k2,B0,B1,uarr,w):
    '''
    Two stream source function technique described in 
    Toon, McKay, & Ackerman, 1989, JGR, 94.

    Parameters
    ----------
    B : ndarray 
        planck function array (with optical depth)
    tautop : ndarray 
        optical depth of top most layer
    Bsurf : ndarray 
        "surface" blackbody
    ubari : float 
        0.5 for hemispheric mean approximation
    nlay : int 
        number of model layers
    lam : narray
        Wavelength 
    dtau : ndarray
        optical depth per layer
    taump : ndarray 
    taut : ndarray 
    ssa : ndarray 
        single scattering albedo 
    hg : ndarray
        asymmetry 
    k1 : ndarray 
    k2 : ndarray 
    B0 : ndarray 
    B1 : ndarray 
    uarr : ndarray 
    w : ndarray

    Returns
    -------
    Fu 
        layer interface monochromatic upward flux
    Fd
        layer interface monochromatic downward flux
    '''
    twopi=2.*np.pi

    Fd=np.zeros(nlay+1)
    Fu=np.zeros(nlay+1)
 
    alphax=((1.-ssa)/(1.-ssa*hg))**0.5
    g=twopi*ssa*k1*(1+hg*alphax)/(1.+alphax)
    h=twopi*ssa*k2*(1-hg*alphax)/(1.+alphax)
    xj=twopi*ssa*k1*(1-hg*alphax)/(1.+alphax)
    xk=twopi*ssa*k2*(1+hg*alphax)/(1.+alphax)
    alpha1=twopi*(B0+B1*(ubari*ssa*hg/(1.-ssa*hg)))
    alpha2=twopi*B1
    sigma1=twopi*(B0-B1*(ubari*ssa*hg/(1.-ssa*hg)))
    sigma2=alpha2
    
    #so overflow/underflows don't happen
    g[ssa < 0.01]=0.
    h[ssa < 0.01]=0.
    xj[ssa < 0.01]=0.
    xk[ssa < 0.01]=0.
    alpha1[ssa < 0.01]=twopi*B0[ssa < 0.01]
    alpha2[ssa < 0.01]=twopi*B1[ssa < 0.01]
    sigma1[ssa < 0.01]=alpha1[ssa < 0.01]
    sigma2[ssa < 0.01]=alpha2[ssa < 0.01]
    
    #more array definitions
    fpt=np.zeros(nlay+1)
    fmt=np.zeros(nlay+1)
   
    em=np.exp(-lam*dtau)
    obj=lam*dtau
    obj[obj > 35.]=35.
    epp=np.exp(obj)
    em4_mp=np.exp(-lam*dtau)
    obj2=0.5*lam*dtau
    obj2[obj2 > 35.]=35.
    epp2=np.exp(obj2)
    ngauss=len(uarr)
    
    for i in range(ngauss):
        ugauss=uarr[i]
        fpt[:]=0.
        fmt[:]=0.
       
        fpt[-1]=twopi*(Bsurf+B1[-1]*ugauss)  #bottom BD
        fmt[0]=twopi*(1.-np.exp(-tautop/ugauss))*B[0]  #top BC
        em2=np.exp(-dtau/ugauss)
        em3=em*em2
        #ray tracing intensities from bottom to top (upwards intensity, fpt) and from top to bottom (downwards intensity, fmt)     
        for j in range(nlay):  #j is from TOA "down"
        	#downards emission intensity from TOA
            fmt[j+1]=fmt[j]*em2[j]+(xj[j]/(lam[j]*ugauss+1.))*(epp[j]-em2[j])+(xk[j]/(lam[j]*ugauss-1.))*(em2[j]-em[j])+sigma1[j]*(1.-em2[j])+sigma2[j]*(ugauss*em2[j]+dtau[j]-ugauss)
            
            #upwards emission intensity from bottom of atmosphere (nlay-1)
            z=nlay-1-j  #flipping indicies to integrate from bottom up
            fpt[z]=fpt[z+1]*em2[z]+(g[z]/(lam[z]*ugauss-1))*(epp[z]*em2[z]-1)+(h[z]/(lam[z]*ugauss+1))*(1.-em3[z])+alpha1[z]*(1.-em2[z])+alpha2[z]*(ugauss-(dtau[z]+ugauss)*em2[z])
        
        #gauss quadrature integration for fluxes
        Fu=Fu+fpt*uarr[i]*w[i]
        Fd=Fd+fmt*uarr[i]*w[i]
    
    return Fu, Fd



@jit(nopython=True)
def toon(dtau1, ssa1, hg1, B):
    """
    Monochromatic two-stream radiative transfer solver described in
    Toon, McKay, & Ackerman, 1989, JGR, 94.  Modified from
    https://github.com/adamkovics/atmosphere/blob/master/atmosphere/rt/twostream.py
    (described in Adamkovics et al. 2016) and the Ames Mars GCM radiative transfer 
    from https://spacescience.arc.nasa.gov/mars-climate-modeling-group/models.html,
    Hollingsworth et al. This then calls the src_func_loop which recomputes the l
    ayer intensities at select cos(ZA) using the two stream solution as the intial
    source function.

    Parameters
    ----------
    dtau : ndarray
        layer/slab optical depths
    ssa : ndarray 
        layer/slab single scatter albedo
    hg  : ndarray 
        layer/slab asymmetry parameter
    B : ndarray 
        planck function at each *level* (N levels, N-1 layers/slabs)

    Returns
    -------
    Fup : ndarray
        monochromatic upwards flux at each layer interface
    Fdown : ndarray
        monochromatic downwards flux at each layer interface
    """
    dtau1[dtau1 < 1E-5]=1E-5

    #delta eddington correction for peaky scatterers
    ssa=(1.-hg1**2)*ssa1/(1.-ssa1*hg1**2)
    dtau=(1.-ssa1*hg1**2)*dtau1
    hg=hg1/(1.+hg1)

    ubari=0.5#1./np.sqrt(3)#0.5
    nlay = len(dtau)
    
    taub = np.cumsum(dtau)    # Cumulative optical depth at layer bottoms
    taut=np.zeros(len(dtau))      
    taut[1:]=taub[:-1] # Cumulative optical depth at layer tops (right? the bottom of one is the top of the one below it =)
    
    taump=taut+0.5*dtau #midpoint optical depths
    
    twopi = np.pi+np.pi  #2pi
    
    
    #AMES MARS CODE equations--Hemispheric Mean Approximation for plankian source (ubari=0.5 in IR)
    #see also Table 1 in Toon et al. 1989, plus some algebra
    alpha = ((1.-ssa)/(1.-ssa*hg))**0.5
    lam = alpha*(1.-ssa*hg)/ubari
    gamma = (1.-alpha)/(1.+alpha)
    term = ubari/(1.-ssa*hg)
    
    #computing linearized planck function (the glorious linear-in-tau)
    B0=B[0:-1]
    B1=(B[1:]-B[:-1])/dtau
    loc=np.where(dtau <= 3E-6)[0]
    B1[loc]=0.
    B0[loc][:-1]=0.5*(B0[loc][1:]+B0[loc][:-1])
    
    # Cpm1 and Cmm1 are the C+ and C- terms evaluated at the top of the layer (at dtau=0).
    Cpm1 =B0+B1*term  #ames code
    Cmm1 =B0-B1*term
    
    # Cp and Cm are the C+ and C- terms evaluated at the bottom of the layer.
    Cp =B0+B1*dtau+B1*term #ames code
    Cm =B0+B1*dtau-B1*term
    
    #
    tautop=dtau[0]*np.exp(-1)
    Btop=(1.-np.exp(-tautop/ubari))*B[0]
    Bsurf=B[-1]
    bottom=Bsurf+B1[-1]*ubari
    
    # Solve for the coefficients of system of equations using boundary conditions
    exptrm = lam*dtau
    exptrm[exptrm>35] = 35 # clipped so that exponential doesn't explode
    Ep = np.exp(exptrm)
    Em = 1./Ep
    
    E1 = Ep + gamma*Em
    E2 = Ep - gamma*Em
    E3 = gamma*Ep + Em
    E4 = gamma*Ep - Em
    
    L = nlay+nlay
    Af = np.zeros(L)
    Bf = np.zeros(L)
    Cf = np.zeros(L)
    Df = np.zeros(L)
    
    # First Term
    Af[0] = 0.0
    Bf[0] = gamma[0] + 1.
    Cf[0] = gamma[0] - 1.
    Df[0] = Btop - Cmm1[0]
    
    AA = (E1[:-1]+E3[:-1])*(gamma[1:]-1)
    BB = (E2[:-1]+E4[:-1])*(gamma[1:]-1)
    CC = 2.*(1.-gamma[1:]*gamma[1:])
    DD = (gamma[1:]-1) * (Cpm1[1:] - Cp[:-1]) + (1-gamma[1:]) * (Cm[:-1]-Cmm1[1:])
    Af[1:-1:2]=AA
    Bf[1:-1:2]=BB
    Cf[1:-1:2]=CC
    Df[1:-1:2]=DD
    
    AA = 2.*(1.-gamma[:-1]*gamma[:-1])
    BB = (E1[:-1]-E3[:-1])*(gamma[1:]+1.)
    CC = (E1[:-1]+E3[:-1])*(gamma[1:]-1.)
    DD = E3[:-1]*(Cpm1[1:] - Cp[:-1]) + E1[:-1]*(Cm[:-1] - Cmm1[1:])
    Af[2::2]=AA
    Bf[2::2]=BB
    Cf[2::2]=CC
    Df[2::2]=DD
    
    # Last term:
    rsf=0
    Af[-1] = E1[-1]-rsf*E3[-1]
    Bf[-1] = E2[-1]-rsf*E4[-1]
    Cf[-1] = 0.0
    Df[-1] = Bsurf - Cp[-1]+rsf*Cm[-1]
    
    k=tri_diag_solve(L, Af, Bf, Cf, Df)
    
    # Unmix coefficients
    even = np.arange(0,2*nlay,2)
    odd  = even+1
    k1 = k[even] + k[odd]
    k2 = k[even] - k[odd]
    
    #this would be the raw two stream solution Fluxes, but we
    #won't use these. Will use source function technique
    Fupraw = np.pi*(k1*Ep + gamma*k2*Em + Cpm1) #
    Fdownraw=np.pi*(k1*Ep*gamma + k2*Em + Cmm1)
    
    #source function part
    uarr=np.array([0.1834346,0.5255324,0.7966665,0.9602899])  #angluar gauss quad cos zenith angles
    w=np.array([0.3626838,0.3137066, 0.2223810, 0.1012885 ])  #gauss quad weights
   
    Fup, Fdown=src_func_loop(B,tautop,Bsurf,ubari,nlay,lam,dtau,taump,taut, ssa,hg,k1,k2,B0,B1,uarr,w)
    
    return Fup,Fdown#, Fupraw, Fdownraw


@jit(nopython=True)
def toon_solar(dtau1, ssa1, hg1, rsurf, MU0, F0PI, BTOP):
    """
    Monochromatic two-stream radiative transfer solver described in
    Toon, McKay, & Ackerman, 1989, JGR, 94.  Modified from
    https://github.com/adamkovics/atmosphere/blob/master/atmosphere/rt/twostream.py
    (described in Adamkovics et al. 2016) and the Ames Mars GCM radiative transfer 
    from https://spacescience.arc.nasa.gov/mars-climate-modeling-group/models.html,
    Hollingsworth et al. This then calls the src_func_loop which recomputes the l
    ayer intensities at select cos(ZA) using the two stream solution as the intial
    source function.

    Parameters
    ----------
    dtau : ndarray 
        layer/slab optical depths
    ssa : ndarray 
        layer/slab single scatter albedo
    hg  : ndarray 
        layer/slab asymmetry parameter
    rsurf : ndarray 
        surface reflectivity
    MU0 : ndarray 
        cos(theta) where theta is both the incidence an emission angle.
    F0PI : ndarray 
        top of atmosphere incident flux * pi (the direct beam)
    BTOP : ndarray 
        top of the atmosphere diffuse flux.

    Returns
    -------
    Fup 
        monochromatic upwards reflected stellar flux at each layer interface
    Fdn
        monochromatic downwards direct and diffuse stellar flux at each layer interface
    """
    #delta eddington scaling
    dtau1[dtau1 < 1E-5]=1E-5
    ssa=(1.-hg1**2)*ssa1/(1.-ssa1*hg1**2)
    dtau=(1.-ssa1*hg1**2)*dtau1
    hg=hg1/(1.+hg1)
    
    
    nlay = len(dtau)
    
    # Cumulative optical depth
    taub = np.cumsum(dtau)
    taut=np.zeros(len(dtau))
    taut[1:]=taub[:-1]
    
    taump=taut+0.5*dtau
    
    # Surface reflectance and lower boundary condition
    bsurf = rsurf * MU0 * F0PI * np.exp(-taub[-1]/MU0)
    
    twopi = np.pi+np.pi
    #
    g1 = 0.86602540378 * (2.-ssa*(1+hg))
    g2 = (1.7320508075688772*ssa/2.) * (1-hg)
    g2[g2 == 0.] = 1E-10
    g3 = (1.-1.7320508075688772*hg*MU0)/2
    g4 = 1. - g3
    
    lam = np.sqrt(g1*g1 - g2*g2)
    gamma = (g1-lam)/g2
    alpha = np.sqrt( (1.-ssa) / (1.-ssa*hg) )
    
    Am = F0PI * ssa *(g4 * (g1 + 1./MU0) + g2*g3 )/ (lam*lam - 1./(MU0*MU0))
    Ap = F0PI * ssa *(g3 * (g1 - 1./MU0) + g2*g4 )/ (lam*lam - 1./(MU0*MU0))
    
    # Cpm1 and Cmm1 are the C+ and C- terms evaluated at the top of the layer.
    Cpm1 = Ap * np.exp(-taut/MU0)
    Cmm1 = Am * np.exp(-taut/MU0)
    # Cp and Cm are the C+ and C- terms evaluated at the bottom of the layer.
    Cp = Ap * np.exp(-taub/MU0)
    Cm = Am * np.exp(-taub/MU0)
    
    #  Solve for the coefficients of system of equations using boundary conditions
    # Exponential terms:
    exptrm = lam*dtau
    exptrm[exptrm>35] = 35 # clipped so that exponential doesn't explode
    Ep = np.exp(exptrm)
    Em = 1./Ep
    
    E1 = Ep + gamma*Em
    E2 = Ep - gamma*Em
    E3 = gamma*Ep + Em
    E4 = gamma*Ep - Em
    
    L = nlay+nlay
    Af = np.empty(L)
    Bf = np.empty(L)
    Cf = np.empty(L)
    Df = np.empty(L)
    
    # First Term
    Af[0] = 0.0
    Bf[0] = gamma[0] + 1.
    Cf[0] = gamma[0] - 1.
    Df[0] = BTOP - Cmm1[0]
    
    AA = (E1[:-1]+E3[:-1])*(gamma[1:]-1)
    BB = (E2[:-1]+E4[:-1])*(gamma[1:]-1)
    CC = 2.*(1.-gamma[1:]*gamma[1:])
    DD = (gamma[1:]-1) * (Cpm1[1:] - Cp[:-1]) + (1-gamma[1:]) * (Cm[:-1]-Cmm1[1:])
    Af[1:-1:2]=AA
    Bf[1:-1:2]=BB
    Cf[1:-1:2]=CC
    Df[1:-1:2]=DD
    
    AA = 2.*(1.-gamma[:-1]*gamma[:-1])
    BB = (E1[:-1]-E3[:-1])*(gamma[1:]+1.)
    CC = (E1[:-1]+E3[:-1])*(gamma[1:]-1.)
    DD = E3[:-1]*(Cpm1[1:] - Cp[:-1]) + E1[:-1]*(Cm[:-1] - Cmm1[1:])
    Af[2::2]=AA
    Bf[2::2]=BB
    Cf[2::2]=CC
    Df[2::2]=DD
    # Last term:
    Af[-1] = E1[-1] - rsurf*E3[-1]
    Bf[-1] = E2[-1] - rsurf*E4[-1]
    Cf[-1] = 0.0
    Df[-1] = bsurf - Cp[-1] + rsurf*Cm[-1]
    
    k=tri_diag_solve(L, Af, Bf, Cf, Df)
    
    # Unmix coefficients
    even = np.arange(0,2*nlay,2)
    odd  = even+1
    k1 = k[even] + k[odd]
    k2 = k[even] - k[odd]
    
    Fup0=k1*Ep+gamma*k2*Em+Cpm1
    Fdn0=k1*Ep*gamma+k2*Em+Cmm1
    
    Cpmid = Ap * np.exp(-taump/MU0)
    Cmmid = Am * np.exp(-taump/MU0)
    Fup_diffuse=k1*Ep+gamma*k2*Em+Cpmid
    Fdn_diffuse=k1*Ep*gamma+k2*Em+Cmmid
    
    Fdn=np.zeros(nlay+1)
    Fup=np.zeros(nlay+1)
    Fdn[0:-1]=Fdn_diffuse+MU0*F0PI*np.exp(-taump/MU0)
    Fup[0:-1]=Fup_diffuse
    
    return Fup, Fdn




@jit(nopython=True)
def compute_RT(wnocrop,T,kcoeffs_interp,xsecContinuum,Mies,mass_path,
    Fractions,Fractions_Continuum,Fractions_Cond, gord,wts, Fstar):
    """
    Calls all requisite radiative transfer routines for emission

    Parameters
    ----------
    wnocrop : ndarray 
        wavenumber grid 
    T : ndarray 
        temperatures 
    kcoeffs_interp : ndarray 
        Interpolated k-coefficients 
    xsecContinuum : ndarray 
        cross sections for continuum opacity 
    Mies : ndarray 
        Mie scattering parameters 
    mass_path : ndarray 
    Fractions : ndarray
        volume mixing ratios of all absorbers (excluding continuum)
    Fractions_Continuum : ndarray
        volume mixing ratios for continuum 
    Fractions_Cond : ndarray
        volume mixing ratios for condensates 
    gord : ndarray
        g-oordinates for k tables
    wts : ndarray 
        weights for k tables 
    Fstar : ndarray
        Flux of parent star

    Returns
    -------

    """
    mu0=0.5#1./np.sqrt(3.) #what should I do with this???
    Nwno=len(wnocrop)
    Nlay=kcoeffs_interp.shape[0]
    Fuparr=np.zeros((Nwno, Nlay+1))
    Fdnarr=np.zeros((Nwno, Nlay+1))
    Fdnarr_star=np.zeros((Nwno, Nlay+1))
    Fuparr_star=np.zeros((Nwno, Nlay+1))
    dtauarr=np.zeros((Nlay,len(gord),Nwno))
    dtau_condarr=np.zeros((Nlay,len(gord),Nwno))
    dtau_gasarr=np.zeros((Nlay,len(gord),Nwno))
    ssaarr=np.zeros((Nlay,len(gord),Nwno))
    asymarr=np.zeros((Nlay,len(gord),Nwno))
    for i in range(Nwno): #looping over wavenumber
        #print 'BEGIN compute_RT WNO Loop: ', datetime.datetime.now().time()
        B=blackbody(T,1E4/wnocrop[i]*1E-6)
        #print 'Compute tau: ', datetime.datetime.now().time()
        dtau,dtau_cond,dtau_gas,ssa, asym=compute_tau(kcoeffs_interp[:,i,:,:],xsecContinuum[:,i,:],Mies[:,:,:,i], mass_path, Fractions,Fractions_Continuum, Fractions_Cond,gord, wts)
        dtauarr[:,:,i]=dtau
        ssaarr[:,:,i]=ssa
        asymarr[:,:,i]=asym
        dtau_condarr[:,:,i]=dtau_cond
        dtau_gasarr[:,:,i]=dtau_gas
    
        #print 'Looping over Gords: ', datetime.datetime.now().time()
        for j in range(len(gord)): #looping over g-space
            Fup_k,Fdn_k=toon(dtau[:,j], ssa[:,j]*1. ,asym[:,j]*1. , B)  #toon
            Fup_star_k, Fdn_star_k=stellar_flux(dtau[:,j], ssa[:,j]*1. ,asym[:,j]*1.,Fstar[i],mu0)
            Fuparr[i,:]+=Fup_k*wts[j]
            Fdnarr[i,:]+=Fdn_k*wts[j]
            Fdnarr_star[i,:]+=Fdn_star_k*wts[j]*1.
            Fuparr_star[i,:]+=Fup_star_k*wts[j]*1.

    return 0.5*Fuparr+0.5*Fuparr_star, 0.5*Fdnarr+0.5*Fdnarr_star, 0.5*Fuparr,0.5*Fuparr_star , dtauarr, ssaarr, asymarr #0.5 is b/c g-ordinates go from -1 - 1 instead of 0 -1



@jit(nopython=True)
def stellar_flux(dtau,ssa,hg, Fstar0, mu0):  #need to give this dtau and mu0
    """ 
    """
    rsfc=0.0
    Fup_s,Fdn_s=toon_solar(dtau, ssa, hg, rsfc,mu0,Fstar0,0.)
    return 0.5*Fup_s, 0.5*Fdn_s


def rad(xsects, T, P, mmw,Ps,CldOpac,alphaH2O,alphaCH4,alphaCO,alphaCO2,
        alphaNH3,alphaNa,alphaK,alphaTiO,alphaVO, alphaC2H2, alphaHCN, 
        alphaH2S,alphaFeH,fH,fe,fHm,fH2,fHe,amp,power,f_r,M,Rstar,Rp, D):
    """
    """
    t1=time.time()
    Frac_Cond=f_r
    #renaming variables, bbecause why not
    fH2=fH2
    fHe=fHe
    fH2O=alphaH2O
    fCH4=alphaCH4
    fCO=alphaCO
    fCO2=alphaCO2
    fNH3=alphaNH3
    fNa=alphaNa
    fK=alphaK
    fTiO=alphaTiO
    fVO=alphaVO
    fC2H2=alphaC2H2
    fHCN=alphaHCN
    fH2S=alphaH2S
    fFeH=alphaFeH
    mmw=mmw
    
    
    Fractions = np.array([fH2*fH2,fHe*fH2,fH2O, fCH4, fCO, fCO2, fNH3,fNa,fK,fTiO,fVO,fC2H2,fHCN,fH2S,fFeH,fH*fe, fHm])  #gas mole fraction profiles
                        #H2Ray,HeRay  Filling for power lay   Filling for gray opacity
    Frac_Cont = np.array([fH2,  fHe,  fH2*0.+1.,               fH2*0.+1])  #continuum mole fraction profiles

    #Load measured cross-sectional values and their corresponding
    #T,P,and wno grids on which they were measured
    Pgrid, Tgrid, wno, gord, wts, xsecarr, radius, Mies, Fstar0=xsects[0:9]
    Fstar_TOA=Fstar0*(Rstar*6.95508E8)**2/(D*1.496E11)**2  #for reflected light...
  
    #Hydrostatic grid
    n = len(P)
    nv = len(wno)
    
    Z=np.zeros(n)  #level altitudes
    dZ=np.zeros(n)  #layer thickness array
    r0=Rp*71492.*1.E3  #converting planet radius to meters
    mmw=mmw*1.660539E-27  #converting mmw to Kg
    kb=1.38E-23
    G=6.67428E-11
    M=M*1.89852E27

    #Compute avg Temperature at each grid
    Tavg=0.5*(T[1:]+T[:-1])
    Pavg=0.5*(P[1:]+P[:-1])
    g0=np.array([0.0]*(n))

    #create hydrostatic altitutde grid from P and T
    Phigh=P.compress((P>Ps).flat)  #deeper than reference pressure
    Plow=P.compress((P<=Ps).flat)   #shallower than reference pressure
    for i in range(Phigh.shape[0]):  #looping over levels above ref pressure
        i=i+Plow.shape[0]-1
        g=G*M/(r0+Z[i])**2
        g0[i]=g
        H=kb*Tavg[i]/(mmw[i]*g)  #scale height
        dZ[i]=H*np.log(P[i+1]/P[i]) #layer thickness, dZ is negative
        Z[i+1]=Z[i]-dZ[i]   #level altitude
    for i in range(Plow.shape[0]-1):  #looping over levels below ref pressure
        i=Plow.shape[0]-i-1
        g=G*M/(r0+Z[i])**2
        g0[i]=g
        H=kb*Tavg[i]/(mmw[i]*g)
        dZ[i]=H*np.log(P[i+1]/P[i])
        Z[i-1]=Z[i]+dZ[i]

    gavg=0.5*(g0[1:]+g0[:-1])  #layer average
    gavg[0]=gavg[1]   #annoying bottom layer
    gavg[-1]=gavg[-2]  #annoying top layer
    dP=P[1:]-P[:-1]
    mass_path=1./mmw/gavg*dP*1.E5

    #Interpolate values of measured cross-sections 
    TT=np.zeros(len(Tavg))
    TT[:]=Tavg
    TT[Tavg < 300] = 400.
    TT[Tavg > 3000] = 2800.
    PP=np.zeros(len(Pavg))
    PP[:]=Pavg
    PP[Pavg < 3E-6]=3E-6
    PP[Pavg >=300 ]=300
    kcoeffs_interp=10**kcoeff_interp(np.log10(Pgrid), np.log10(Tgrid), np.log10(PP), np.log10(TT), wno, xsecarr)
    t3=time.time()

    #continuum opacities (nlayers x nwnobins x ncont)***********
    xsec_cont=kcoeffs_interp[:,:,0,0]
    wave = (1/wno)*1E8
    sigmaH2 = xsec_cont*0.+1*((8.14E-13)*(wave**(-4.))*(1+(1.572E6)*(wave**(-2.))+(1.981E12)*(wave**(-4.))))*1E-4  #H2 gas Ray
    sigmaHe = xsec_cont*0.+1*((5.484E-14)*(wave**(-4.))*(1+(2.44E5)*(wave**(-2.))))*1E-4   #He gas Ray
    #"Rayleigh Haze" from des Etangs 2008--just a hacked parameterization
    wno0=1E4/0.43
    sigmaRay=xsec_cont*0.+2.E-27*amp*(wno/wno0)**power*1E-4

    #"grey" cloud opacity (constant in wl and altitude--totally not realistic)
    sigmaCld=xsec_cont*0.+CldOpac

    #mie scattering 
    xsecMie=Mies[0,0,:,:].T
    sigmaMie=np.repeat(xsecMie[np.newaxis,:,:],len(Pavg),axis=0)

    xsecContinuum=np.array([sigmaH2.T,sigmaHe.T,sigmaRay.T,sigmaCld.T]).T #building continuum xsec array (same order as cont_fracs)

    Fuparr, Fdnarr, Fup_therm,Fup_ref, dtau,ssa, asym=compute_RT(wno,T,kcoeffs_interp,xsecContinuum,Mies,mass_path,Fractions, Frac_Cont,Frac_Cond, gord,wts, Fstar_TOA)

    return wno, Fuparr, Fdnarr, dtau,ssa,asym, wts, Fstar0,Fup_therm,Fup_ref


def instrument_emission_non_uniform(wlgrid,wno, Fp, Fstar):
    """
    Rebin emission spectroscopy on new wavelength grid 

    Parameters
    ----------
    wlgrid : ndarray or int
        new wavelength grid 
    wno : ndarray
        old wavenumber grid 
    Fp : ndarray
        Flux of planet 
    Fstar : ndarray
        Flux of star 

    Returns
    -------
    fp/fs on new wavelength grid
    """
    if isinstance(wlgrid,int):
        return Fp/Fstar

    else:
        szmod=wlgrid.shape[0]
        delta=np.zeros(szmod)
        Fp_binned=np.zeros(szmod)
        Fstar_binned=np.zeros(szmod)
        for i in range(szmod-1):
            delta[i]=wlgrid[i+1]-wlgrid[i]  

        delta[szmod-1]=delta[szmod-2] 

        for i in range(szmod-1):
            i=i+1
            loc=np.where((1E4/wno > wlgrid[i]-0.5*delta[i-1]) & (1E4/wno < wlgrid[i]+0.5*delta[i]))
            Fp_binned[i]=np.mean(Fp[loc])
            Fstar_binned[i]=np.mean(Fstar[loc])


        loc=np.where((1E4/wno > wlgrid[0]-0.5*delta[0]) & (1E4/wno < wlgrid[0]+0.5*delta[0]))
        Fp_binned[0]=np.mean(Fp[loc])
        Fstar_binned[0]=np.mean(Fstar[loc])

        Fratio_binned=Fp_binned/Fstar_binned
        return Fratio_binned



##############################   TRANSMISSION ROUTINES  ############################
####################################################################################
####################################################################################    

@jit(nopython=True)
def CalcTauXsecCK(kcoeffs,Z,Pavg,Tavg, Fractions, r0,gord, wts, 
                Fractions_Continuum, xsecContinuum):
    """
    Calculate opacity using correlated-k tables 
    
    Parameters
    ----------
    kcoeffs : ndarray 
        correlated-k tables 
    Z : ndarray
        height above ref pressure
    Pavg : ndarray
        pressure grid 
    Tavg : ndarray
        temperature grid 
    Fractions : ndarray
        volume mixing ratios of species 
    Fractions_Continuum : ndarray
        volume mixing ratios of continuum species 
    xsecContinuum : ndarray
        cross sections for continuum

    Returns
    -------
    transmission
    """
    ngas=Fractions.shape[0]
    nlevels=len(Z)
    nwno=kcoeffs.shape[1]
    trans=np.zeros((nwno, nlevels))+1.
    dlarr=np.zeros((nlevels,nlevels))
    ncont=xsecContinuum.shape[-1]
    uarr=np.zeros((nlevels,nlevels))
    kb=1.38E-23
    kbTavg=kb*Tavg
    Pavg_pascal=1E5*Pavg
    for i in range(nlevels):
        for j in range(i):
            index=i-j-1
            r1=r0+Z[i]
            r2=r0+Z[i-j]
            r3=r0+Z[index]
            dlarr[i,j]=(r3**2-r1**2)**0.5-(r2**2-r1**2)**0.5
            uarr[i,j]=dlarr[i,j]*Pavg_pascal[index]/kbTavg[index]
    
    for v in range(nwno):
        for i in range(nlevels):
            transfull=1.
            #for CK gases--try to do ALL gases as CK b/c of common interpolation
            for k in range(ngas):
                transtmp=0.
                for l in range(len(wts)):
                    tautmp=0.
                    for j in range(i):
                        index=i-j-1
                        tautmp+=2.*Fractions[k,index]*kcoeffs[index,v,k,l]*uarr[i,j]
                    transtmp+=np.exp(-tautmp)*wts[l]/2.
                transfull*=transtmp
            #for continuum aborbers (gas rayligh, condensate scattering etc.--nlayers x nwno x ncont
            #'''
            for k in range(ncont):
                tautmp=0.
                for j in range(i):

                    index=i-j-1
                    tautmp+=2.*Fractions_Continuum[k,index]*xsecContinuum[index,v,k]*uarr[i,j]

                transfull*=np.exp(-tautmp)
            #'''
            trans[v,i]=transfull
    return trans


def tran(xsects, T, P, mmw,Ps,CldOpac,alphaH2O,alphaCH4,alphaCO,alphaCO2,alphaNH3,alphaNa,alphaK,alphaTiO,alphaVO, alphaC2H2, alphaHCN, alphaH2S,alphaFeH,fH,fe,fHm,fH2,fHe,amp,power,f_r,M,Rstar,Rp):
    """Runs all requisite routins for transmisison spectroscopy

    Returns
    -------
    wno
        Wavenumber grid 
    F
        (rp/rs)^2
    Z
        height above ref pressure
    """
    t1=time.time()

    #renaming variables, bbecause why not
    fH2=fH2
    fHe=fHe
    fH2O=alphaH2O
    fCH4=alphaCH4
    fCO=alphaCO
    fCO2=alphaCO2
    fNH3=alphaNH3
    fNa=alphaNa
    fK=alphaK
    fTiO=alphaTiO
    fVO=alphaVO
    fC2H2=alphaC2H2
    fHCN=alphaHCN
    fH2S=alphaH2S
    fFeH=alphaFeH
    mmw=mmw
    #pdb.set_trace()
    #Na and K are fixed in this model but can be made free parameter if desired
    #If T < 800 K set these equal to 0!!!--they condense out below this temperature (roughly)
   
    
    Fractions = np.array([fH2*fH2,fHe*fH2,fH2O, fCH4, fCO, fCO2, fNH3,fNa,fK,fTiO,fVO,fC2H2,fHCN,fH2S,fFeH,fH*fe, fHm])  #gas mole fraction profiles
                        #H2Ray, HeRay  Ray General,
    Frac_Cont = np.array([fH2,fHe,fH2*0.+1.,fH2*0.+1])  #continuum mole fraction profiles
    Frac_Cont=np.concatenate((Frac_Cont, f_r),axis=0)

    #Load measured cross-sectional values and their corresponding
    #T,P,and wno grids on which they were measured
    Pgrid, Tgrid, wno, gord, wts, xsecarr, radius, Mies, Fstar=xsects[0:9]
    
    
    #Calculate Temperature, Pressure and Height grids on which
    #transmissivity will be computed
    n = len(P)
    nv = len(wno)
    
    
    Z=np.zeros(n)  #level altitudes
    dZ=np.zeros(n)  #layer thickness array
    r0=Rp*71492.*1.E3  #converting planet radius to meters
    mmw=mmw*1.660539E-27  #converting mmw to Kg
    kb=1.38E-23
    G=6.67428E-11
    M=M*1.89852E27

    
    #Compute avg Temperature at each grid
    Tavg = np.array([0.0]*(n-1))
    Pavg = np.array([0.0]*(n-1))
    for z in range(n-1):
        Pavg[z] = np.sqrt(P[z]*P[z+1])
        Tavg[z] = interp(np.log10(Pavg[z]),sp.log10(P),T)
    #create hydrostatic altitutde grid from P and T
    Phigh=P.compress((P>Ps).flat)  #deeper than reference pressure
    Plow=P.compress((P<=Ps).flat)   #shallower than reference pressure
    for i in range(Phigh.shape[0]):  #looping over levels above ref pressure
        i=i+Plow.shape[0]-1
        g=G*M/(r0+Z[i])**2#g0*(Rp/(Rp+Z[i]/(69911.*1E3)))**2
        H=kb*Tavg[i]/(mmw[i]*g)  #scale height
        dZ[i]=H*np.log(P[i+1]/P[i]) #layer thickness, dZ is negative
        Z[i+1]=Z[i]-dZ[i]   #level altitude
        #print(P[i], H/1000, Z[i]/1000, g)
    for i in range(Plow.shape[0]-1):  #looping over levels below ref pressure
        i=Plow.shape[0]-i-1
        g=G*M/(r0+Z[i])**2#g0*(Rp/(Rp+Z[i]/(69911.*1E3)))**2
        H=kb*Tavg[i]/(mmw[i]*g)
        dZ[i]=H*np.log(P[i+1]/P[i])
        Z[i-1]=Z[i]+dZ[i]
        #print(P[i], H/1000., Z[i]/1000, g)

    #pdb.set_trace()
    #Interpolate values of measured cross-sections at their respective
    #temperatures pressures to the temperature and pressure of the
    #levels on which the optical depth will be computed
    t2=time.time()
    #print('Setup', t2-t1)
    #make sure   200 <T <4000 otherwise off cross section grid
    TT=np.zeros(len(Tavg))
    TT[:]=Tavg
    TT[Tavg < 500] = 500.
    TT[Tavg > 3000] = 3000.
    PP=np.zeros(len(Pavg))
    PP[:]=Pavg
    PP[Pavg < 3E-6]=3E-6
    PP[Pavg >=300 ]=300


    kcoeffs_interp=10**kcoeff_interp(np.log10(Pgrid), np.log10(Tgrid), np.log10(PP), np.log10(TT), wno, xsecarr)
    t3=time.time()
    #print('Kcoeff Interp', t3-t2)
    #continuum opacities (nlayers x nwnobins x ncont)***********
    xsec_cont=kcoeffs_interp[:,:,0,0]
    wave = (1/wno)*1E8
    sigmaH2 = xsec_cont*0.+1*((8.14E-13)*(wave**(-4.))*(1+(1.572E6)*(wave**(-2.))+(1.981E12)*(wave**(-4.))))*1E-4  #H2 gas Ray
    sigmaHe = xsec_cont*0.+1*((5.484E-14)*(wave**(-4.))*(1+(2.44E5)*(wave**(-2.))))*1E-4   #He gas Ray
    #Rayleigh Haze from des Etangs 2008
    wno0=1E4/0.43
    sigmaRay=xsec_cont*0.+2.E-27*amp*(wno/wno0)**power*1E-4
    #grey cloud opacity
    sigmaCld=xsec_cont*0.+CldOpac

    #mie scattering 
    xsecMie=Mies[0,0,:,:].T
    sigmaMie=np.repeat(xsecMie[np.newaxis,:,:],len(Pavg),axis=0)

    xsecContinuum=np.array([sigmaH2.T,sigmaHe.T,sigmaRay.T,sigmaCld.T]).T #building continuum xsec array (same order as cont_fracs)
    xsecContinuum=np.concatenate((xsecContinuum, sigmaMie),axis=2)
    #(add more continuum opacities here and in fractions)
    t4=time.time()
    #print("Continuum Xsec Setup ", t4-t3)
    #********************************************
    #Calculate transmissivity as a function of
    #wavenumber and height in the atmosphere
    t=CalcTauXsecCK(kcoeffs_interp,Z,Pavg,Tavg, Fractions, r0,gord,wts,Frac_Cont,xsecContinuum)
    t5=time.time()
    #print('Transmittance', t5-t4)    

    #Compute Integral to get (Rp/Rstar)^2 (equation in brown 2001, or tinetti 2012)
    F=((r0+np.min(Z))/(Rstar*6.95508E8))**2+2./(Rstar*6.95508E8)**2.*np.dot((1.-t),(r0+Z)*dZ)
    t6=time.time()
    #print('Total in Trans', t6-t1)

    return wno, F, Z#, TauOne


def instrument_tran_non_uniform(wlgrid,wno, Fp):
    """
    Rebins transmission spectra on new wavelength grid 

    Parameters
    ----------
    wlgrid : ndarray
        New wavelength grid 
    wno : ndarray
        Old wavenumber grid 
    Fp : ndarray
        (rp/rs)^2

    Returns
    -------
    Fratio_int
        new regridded spectrum
    Fp
        old high res spectrum
    """
    if isinstance(wlgrid,int):
        return Fp, Fp

    else:
        szmod=wlgrid.shape[0]
        delta=np.zeros(szmod)
        Fratio=np.zeros(szmod)
        for i in range(szmod-1):
            delta[i]=wlgrid[i+1]-wlgrid[i]  
        delta[szmod-1]=delta[szmod-2] 

        for i in range(szmod-1):
            i=i+1
            loc=np.where((1E4/wno > wlgrid[i]-0.5*delta[i-1]) & (1E4/wno < wlgrid[i]+0.5*delta[i]))
            Fratio[i]=np.mean(Fp[loc])

        loc=np.where((1E4/wno > wlgrid[0]-0.5*delta[0]) & (1E4/wno < wlgrid[0]+0.5*delta[0]))
        Fratio[0]=np.mean(Fp[loc])

        Fratio_int=Fratio
        return Fratio_int, Fp





##############################   CLOUD ROUTINES  ###################################
####################################################################################
####################################################################################    

@jit(nopython=True)
def cloud_profile(fsed,cloud_VMR, Pavg, Pbase):
    """
    A&M2001 eq 7., but in P-coordinates (using hydrostatic) and definition of f_sed

    Parameters
    ----------
    fsed : float 
        sedimentation efficiency 
    cloud_VMR : float 
        cloud base volume mixing ratio 
    Pavg : ndarray
        pressure grid 
    Pbase : float 
        location of cloud base

    Returns
    -------
    Condensate mixing ratio as a function of pressure
    """
    cond=cloud_VMR
    loc0=np.where(Pbase >= Pavg)[0][-1]
    cond_mix=np.zeros(len(Pavg))+1E-50
    cond_mix[0:loc0+1]=cond*(Pavg[0:loc0+1]/Pavg[loc0])**fsed  #
    return cond_mix


@jit(nopython=True)
def particle_radius(fsed,Kzz,mmw,Tavg, Pavg,g, rho_c,mmw_c, qc,rr):
    """
    Computes particle radius based on A&M2011

    Parameters
    ----------
    fsed : float 
        sedimentation efficiency 
    Kzz : float 
        vertical mixing 
    mmw : float 
        mean molecular weight of the atmosphere
    Tavg : float
        temperature 
    Pavg : float 
        pressure 
    g : float 
        gravity 
    rho_c : float 
        density of condensate 
    mmw_c : float 
        molecular weight of condensate 
    qc : float 
        mass mixing ratio of condensate 
    rr : float 
    """
    dlnr=np.abs(np.log(rr[1])-np.log(rr[0]))
    kb=1.38E-23  #boltzman constant
    mu0=1.66E-27  #a.m.u.
    d=2.827E-10  #bath gas molecule diameter (m)
    alpha=1.4  #alpha factor from A&M 2001 (don't need to change this)
    sig_eff=2  #log-normal particle size distribution width
    
    #atmosphere properties
    H=kb*Tavg/(mmw*mu0*g)  #scale height
    rho_a=Pavg*mmw*mu0*1E5/(kb*Tavg)  #atmospheric mass density
    
    wmix=Kzz/H  #vertical mixing velocity
    mfp=kb*Tavg/(2**0.5*np.pi*d**2*Pavg*1E5)   #mean free path
    eta=5./16.*np.sqrt(np.pi*2.3*mu0*kb*Tavg)*(Tavg/59.7)**.16/(1.22*np.pi*d**2) #dynamic viscosity of bath gas
    
    #computing varius radius profiles
    r_sed=2./3.*mfp*((1.+10.125*eta*wmix*fsed/(g*(rho_c-rho_a)*mfp**2))**.5-1.)  #sedimentation radius
    r_eff=r_sed*np.exp(-0.5*(alpha+1)*np.log(sig_eff)**2)  #A&M2011 equation 17 effective radius
    r_g=r_sed*np.exp(-0.5*(alpha+6.)*np.log(sig_eff)**2) #A&M formula (13)--lognormal mean (USE THIS FOR RAD)
    
    #droplet VMR
    f_drop=3.*mmw_c*mu0*qc/(4.*np.pi*rho_c*r_g**3)*np.exp(-4.5*np.log(sig_eff)**2)  #
    prob_lnr=np.zeros((len(rr),len(r_g)))
    for i in range(len(prob_lnr)): prob_lnr[i,:]=1./((2.*np.pi)**0.5*np.log(sig_eff))*np.exp(-0.5*np.log(rr[i]/r_g)**2/np.log(sig_eff)**2)*dlnr
    f_r=prob_lnr*f_drop
    
    return r_sed, r_eff, r_g, f_r



def TP(Teq, Teeff, g00, kv1, kv2, kth, alpha):
    """Guillot 2010 PT profile parameterization

    Returns
    -------
    temperature , pressure
    """
    Teff = Teeff
    f = 1.0  # solar re-radiation factor
    A = 0.0  # planetary albedo
    g0 = g00
    
    # Compute equilibrium temperature and set up gamma's
    T0 = Teq
    gamma1 = kv1/kth
    gamma2 = kv2/kth
    
    # Initialize arrays
    logtau =np.arange(-10,20,.1)
    tau =10**logtau
    
    #computing temperature
    T4ir = 0.75*(Teff**(4.))*(tau+(2.0/3.0))
    f1 = 2.0/3.0 + 2.0/(3.0*gamma1)*(1.+(gamma1*tau/2.0-1.0)*sp.exp(-gamma1*tau))+2.0*gamma1/3.0*(1.0-tau**2.0/2.0)*special.expn(2.0,gamma1*tau)
    f2 = 2.0/3.0 + 2.0/(3.0*gamma2)*(1.+(gamma2*tau/2.0-1.0)*sp.exp(-gamma2*tau))+2.0*gamma2/3.0*(1.0-tau**2.0/2.0)*special.expn(2.0,gamma2*tau)
    T4v1=f*0.75*T0**4.0*(1.0-alpha)*f1
    T4v2=f*0.75*T0**4.0*alpha*f2
    T=(T4ir+T4v1+T4v2)**(0.25)
    P=tau*g0/(kth*0.1)/1.E5
    
    
    # Return TP profile
    return T, P


def fx_trans(x,wlgrid,gas_scale, xsects):
    """Transmission spectrscopy

    Parameters
    ----------
    x : list 
        See tutorials for description of list and order of list 
    wlgrid : ndarray
        Array to regrid final specturm on (micron)
    gas_scale : ndarray
        array to scale mixing ratio of gases 
    xsects : list 
        cross section array from `xsecs` function 

    Returns
    -------
    y_binned,F,wno,chemarr

    binned array of spectrum, high res spectrum, og wavenumber grid, chemistry array
    which includes: 
    chemarr = [P,T, H2Oarr, CH4arr,COarr,CO2arr,NH3arr,Naarr,Karr,TiOarr,VOarr,C2H2arr,HCNarr,H2Sarr,FeHarr,H2arr,Hearr,Harr, earr, Hmarr,qc,r_eff,f_r])
    """

    #print(x)
    #UNPACKING PARAMETER VECTOR.......
    #Unpacking Guillot 2010 TP profile params (3 params)
    Tirr=x[0]
    logKir=x[1]
    logg1=x[2]
    Tint=x[3]
    #Unpacking Chemistry Parms
    Met=10.**x[4]  #metallicity
    CtoO=10.**x[5] #C/O
    logPQC=x[6]  #carbon quench pressure
    logPQN=x[7]  #nitrogen quench pressure
    #unpacking planet params
    Rp=x[8]  #planet radius (in jupiter)
    Rstar=x[9]   #stellar radius (in solar)
    M=x[10]   #planet mass (in jupiter)
    #unpacking and converting A&M cloud params
    Kzz=10**x[11]*1E-4  #Kzz for A&M cloud
    fsed=x[12]  #sedimentation factor for A&M cloud
    Pbase=10.**x[13]  #cloud top pressure
    Cld_VMR=10**x[14]  #Cloud Base Condensate Mixing ratio
    #unpacking and converting simple cloud params
    CldOpac=10**x[15]
    RayAmp=10**x[16]
    RaySlp=x[17]

    #Setting up atmosphere grid****************************************
    logP = np.arange(-6.8,1.5,0.1)+0.1
    P = 10.0**logP
    g0=6.67384E-11*M*1.898E27/(Rp*71492.*1.E3)**2
    kv=10.**(logg1+logKir)
    kth=10.**logKir
    tp=TP(Tirr, Tint,g0 , kv, kv, kth, 0.5)
    T = interp(logP,np.log10(tp[1]),tp[0])
    t1=time.time()
    Tavg=0.5*(T[1:]+T[:-1])
    Pavg=0.5*(P[1:]+P[:-1])

    #interpolation chem
    logCtoO, logMet, Tarr, logParr, loggas=xsects[9:]  #see xsects_HST/JWST routine...
    Ngas=loggas.shape[-2]
    gas=np.zeros((Ngas,len(P)))+1E-20

    #capping T at bounds
    TT=np.zeros(len(T))
    TT[:]=T[:]
    TT[TT>3400]=3400
    TT[TT<500]=500

    for j in range(Ngas):
        gas_to_interp=loggas[:,:,:,j,:]
        IF=RegularGridInterpolator((logCtoO, logMet, np.log10(Tarr),logParr),gas_to_interp,bounds_error=False)
        for i in range(len(P)):
            gas[j,i]=10**IF(np.array([np.log10(CtoO), np.log10(Met), np.log10(TT[i]), np.log10(P[i])]))*gas_scale[j]

    H2Oarr, CH4arr, COarr, CO2arr, NH3arr, N2arr, HCNarr, H2Sarr,PH3arr, C2H2arr, C2H6arr, Naarr, Karr, TiOarr, VOarr, FeHarr, Harr,H2arr, Hearr,earr, Hmarr,mmw=gas


    #Super simplified non-self consistent quenching based on quench pressure
    #Carbon
    PQC=10.**logPQC
    loc=np.where(P <= PQC)
    CH4arr[loc]=CH4arr[loc][-1]
    COarr[loc]=COarr[loc][-1]
    H2Oarr[loc]=H2Oarr[loc][-1]
    CO2arr[loc]=CO2arr[loc][-1]

    #Nitrogen
    PQN=10.**logPQN
    loc=np.where(P <= PQN)
    NH3arr[loc]=NH3arr[loc][-1]
    N2arr[loc]=N2arr[loc][-1]
    t2=time.time()

    #hacked rainout (but all rainout is...).if a mixing ratio profile hits '0' (1E-12) set it to 1E-20 at all layers above that layer
    rain_val=1E-8
    loc=np.where(TiOarr <= rain_val)[0]
    if len(loc>1): TiOarr[0:loc[-1]-1]=1E-20
    #loc=np.where(VOarr <= rain_val)[0]
    if len(loc>1):VOarr[0:loc[-1]-1]=1E-20 #VO and TiO rainout togather
    loc=np.where(Naarr <= rain_val)[0]
    if len(loc>1): Naarr[0:loc[-1]-1]=1E-20
    loc=np.where(Karr <= rain_val)[0]
    if len(loc>1):Karr[0:loc[-1]-1]=1E-20
    loc=np.where(FeHarr <= rain_val)[0]
    if len(loc>1):FeHarr[0:loc[-1]-1]=1E-20


    #ackerman & Marley cloud model here
    mmw_cond=100.39#molecular weight of condensate (in AMU)  MgSiO3=100.39
    rho_cond=3250#density of condensate (in kg/m3)           MgSiO3=3250.
    rr=10**(np.arange(-2,2.6,0.1))  #Droplet radii to compute on: MUST BE SAME AS MIE COEFF ARRAYS!!!!!!!!! iF YOU CHANGE THIS IT WILL BREAK
    qc=cloud_profile(fsed,Cld_VMR, P,Pbase)
    r_sed, r_eff, r_g, f_r=particle_radius(fsed,Kzz,mmw,T, P,g0, rho_cond,mmw_cond,qc, rr*1E-6)
   
    Pref=1.1#10.1  #reference pressure bar-keep fixed
    #computing transmission spectrum-----------
   
    spec = tran(xsects, T,P,mmw, Pref,CldOpac, H2Oarr, CH4arr,COarr,CO2arr,NH3arr,Naarr,Karr,TiOarr,VOarr,C2H2arr,HCNarr,H2Sarr,FeHarr,Harr,earr,Hmarr,H2arr,Hearr,RayAmp,RaySlp,f_r, M, Rstar, Rp)
    wno = spec[0]
    F = spec[1]
    
    y_binned,junk=instrument_tran_non_uniform(wlgrid,wno, F)

    chemarr=np.array([P,T, H2Oarr, CH4arr,COarr,CO2arr,NH3arr,Naarr,Karr,TiOarr,VOarr,C2H2arr,HCNarr,H2Sarr,FeHarr,H2arr,Hearr,Harr, earr, Hmarr,qc,r_eff,f_r])

    return y_binned,F,wno,chemarr


def fx_trans_free(x,wlgrid,gas_scale, xsects):
    """Transmission spectrscopy with free chemistry

    Parameters
    ----------
    x : list 
        See tutorials for description of list and order of list 
    wlgrid : ndarray
        Array to regrid final specturm on (micron)
    gas_scale : ndarray
        array to scale mixing ratio of gases 
    xsects : list 
        cross section array from `xsecs` function 

    Returns
    -------
    y_binned,F,wno,chemarr

    binned array of spectrum, high res spectrum, og wavenumber grid, chemistry array
    which includes: 
    chemarr = [P,T, H2Oarr, CH4arr,COarr,CO2arr,NH3arr,Naarr,Karr,TiOarr,VOarr,C2H2arr,HCNarr,H2Sarr,FeHarr,H2arr,Hearr,Harr, earr, Hmarr,qc,r_eff,f_r])
    """
    #UNPACKING PARAMETER VECTOR.......
    #Unpacking Guillot 2010 TP profile params (3 params)
    Tirr=x[0]
    logKir=x[1]
    logg1=x[2]
    Tint=x[3]
    #Unpacking Chemistry Parms
    Met=10.**x[4]  #metallicity
    CtoO=10.**x[5] #C/O
    logPQC=x[6]  #carbon quench pressure
    logPQN=x[7]  #nitrogen quench pressure
    #unpacking planet params
    Rp=x[8]  #planet radius (in jupiter)
    Rstar=x[9]   #stellar radius (in solar)
    M=x[10]   #planet mass (in jupiter)
    #unpacking and converting A&M cloud params
    Kzz=10**x[11]*1E-4  #Kzz for A&M cloud
    fsed=x[12]  #sedimentation factor for A&M cloud
    Pbase=10.**x[13]  #cloud top pressure
    Cld_VMR=10**x[14]  #Cloud Base Condensate Mixing ratio
    #unpacking and converting simple cloud params
    CldOpac=10**x[15]
    RayAmp=10**x[16]
    RaySlp=x[17]
    
    #Setting up atmosphere grid****************************************
    logP = np.arange(-6.8,1.5,0.1)+0.1
    P = 10.0**logP
    g0=6.67384E-11*M*1.898E27/(Rp*71492.*1.E3)**2
    kv=10.**(logg1+logKir)
    kth=10.**logKir
    tp=TP(Tirr, Tint,g0 , kv, kv, kth, 0.5)
    T = interp(logP,np.log10(tp[1]),tp[0])
    t1=time.time()
    Tavg=0.5*(T[1:]+T[:-1])
    Pavg=0.5*(P[1:]+P[:-1])
    
    #interpolation chem
    logCtoO, logMet, Tarr, logParr, loggas=xsects[9:]  #see xsects_HST/JWST routine...
    Ngas=loggas.shape[-2]
    gas=np.zeros((Ngas,len(P)))+1E-20

    #             H2O   CH4   CO    CO2    NH3   N2    HCN   H2S   PH3 C2H2   C2H6   Na     K    TiO    VO     FeH     H    H2   He   e  H-
    mu=np.array([18.02,16.04,28.01,44.01,17.03,28.01,27.02,34.08,  34.,26.04, 30.07,22.99, 39.1, 63.87, 66.94, 56.85, 1.01, 2.02, 4.0,0.,1.01,0 ])

    gas_scale*=1E0
    for i in range(Ngas): gas[i,:]=10**gas_scale[i]

    H2Oarr, CH4arr, COarr, CO2arr, NH3arr, N2arr, HCNarr, H2Sarr,PH3arr, C2H2arr, C2H6arr, Naarr, Karr, TiOarr, VOarr, FeHarr, Harr,H2arr, Hearr,earr, Hmarr,mmw=gas
    H2He=1.-np.sum(gas,axis=0)
    frac=0.176471
    H2arr=H2He/(1.+frac)
    Hearr=frac*H2arr
    gas[-5]=H2arr
    gas[-4]=Hearr
    #mmw
    mmw[:]=gas.T.dot(mu)
    
    
    #ackerman & Marley cloud model here
    mmw_cond=100.39#molecular weight of condensate (in AMU)  MgSiO3=100.39
    rho_cond=3250#density of condensate (in kg/m3)           MgSiO3=3250.
    rr=10**(np.arange(-2,2.6,0.1))  #Droplet radii to compute on: MUST BE SAME AS MIE COEFF ARRAYS!!!!!!!!! iF YOU CHANGE THIS IT WILL BREAK
    qc=cloud_profile(fsed,Cld_VMR, P,Pbase)
    r_sed, r_eff, r_g, f_r=particle_radius(fsed,Kzz,mmw,T, P,g0, rho_cond,mmw_cond,qc, rr*1E-6)
    
    Pref=1.1#10.1  #reference pressure bar-keep fixed
    #computing transmission spectrum-----------
    
    spec = tran(xsects, T,P,mmw, Pref,CldOpac, H2Oarr, CH4arr,COarr,CO2arr,NH3arr,Naarr,Karr,TiOarr,VOarr,C2H2arr,HCNarr,H2Sarr,FeHarr,Harr,earr,Hmarr,H2arr,Hearr,RayAmp,RaySlp,f_r, M, Rstar, Rp)
    wno = spec[0]
    F = spec[1]

    y_binned,junk=instrument_tran_non_uniform(wlgrid,wno, F)
    
    chemarr=np.array([P,T, H2Oarr, CH4arr,COarr,CO2arr,NH3arr,Naarr,Karr,TiOarr,VOarr,C2H2arr,HCNarr,H2Sarr,FeHarr,H2arr,Hearr,Harr, earr, Hmarr,qc,r_eff,f_r])
    
    return y_binned,F,wno,chemarr


def fx_emis(x,wlgrid,gas_scale, xsects):
    """Emission spectrscopy

    Parameters
    ----------
    x : list 
        See tutorials for description of list and order of list 
    wlgrid : ndarray
        Array to regrid final specturm on (micron)
    gas_scale : ndarray
        array to scale mixing ratio of gases 
    xsects : list 
        cross section array from `xsecs` function 

    Returns
    -------
    FpFstar_binned,FpFstar,wno,chemarr, Ftoa,Fstar,Fstar_TOA,Fup_therm[:,0],Fup_ref[:,0]   
    """   
    #Unpacking Guillot 2010 TP profile params (3 params)
    Tirr=x[0]
    logKir=x[1]
    logg1=x[2]
    Tint=x[3]

    #Unpacking Chemistry Parms
    Met=10.**x[4]  #metallicity
    CtoO=10.**x[5] #C/O
    logPQC=x[6]  #carbon quench pressure
    logPQN=x[7]  #nitrogen quench pressure

    #unpacking planet params
    Rp=x[8]  #planet radius (in jupiter)
    Rstar=x[9]   #stellar radius (in solar)
    M=x[10]   #planet mass (in jupiter)
    D=x[11]

    #unpacking and converting A&M cloud params
    Kzz=10**x[12]*1E-4  #Kzz for A&M cloud
    fsed=x[13]  #sedimentation factor for A&M cloud
    Pbase=10.**x[14]  #cloud top pressure
    Cld_VMR=10**x[15]  #Cloud Base Condensate Mixing ratio

    #unpacking and converting simple cloud params
    CldOpac=10**x[16]
    RayAmp=10**x[17]
    RaySlp=x[18]

    #Setting up atmosphere grid****************************************
    logP = np.arange(-6.8,1.5,0.1)+0.1
    P = 10.0**logP
    g0=6.67384E-11*M*1.898E27/(Rp*71492.*1.E3)**2
    kv=10.**(logg1+logKir)
    kth=10.**logKir

    tp=TP(Tirr, Tint,g0 , kv, kv, kth, 0.5)
    T = interp(logP,np.log10(tp[1]),tp[0])
    t1=time.time()
    Tavg=0.5*(T[1:]+T[:-1])
    Pavg=0.5*(P[1:]+P[:-1])


    #interpolation chem
    logCtoO, logMet, Tarr, logParr, loggas=xsects[9:]
    
    Ngas=loggas.shape[-2]
    gas=np.zeros((Ngas,len(Pavg)))+1E-20

    #capping T at bounds
    TTavg=np.zeros(len(Tavg))
    TTavg[:]=Tavg[:]
    TTavg[TTavg>3400]=3400
    TTavg[TTavg<500]=500

    for j in range(Ngas):
        gas_to_interp=loggas[:,:,:,j,:]
        IF=RegularGridInterpolator((logCtoO, logMet, np.log10(Tarr),logParr),gas_to_interp,bounds_error=False)
        for i in range(len(Pavg)):
            gas[j,i]=10**IF(np.array([np.log10(CtoO), np.log10(Met), np.log10(TTavg[i]), np.log10(Pavg[i])]))*gas_scale[j]

   
    H2Oarr, CH4arr, COarr, CO2arr, NH3arr, N2arr, HCNarr, H2Sarr,PH3arr, C2H2arr, C2H6arr, Naarr, Karr, TiOarr, VOarr, FeHarr, Harr,H2arr, Hearr,earr, Hmarr,mmw=gas

    #Super simplified non-self consistent quenching based on quench pressure

    #Carbon
    PQC=10.**logPQC
    loc=np.where(P <= PQC)
    CH4arr[loc]=CH4arr[loc][-1]
    COarr[loc]=COarr[loc][-1]
    H2Oarr[loc]=H2Oarr[loc][-1]
    CO2arr[loc]=CO2arr[loc][-1]

    #Nitrogen
    PQN=10.**logPQN
    loc=np.where(P <= PQN)
    NH3arr[loc]=NH3arr[loc][-1]
    N2arr[loc]=N2arr[loc][-1]
    t2=time.time()

    #hacked rainout (but all rainout is...)....if a mixing ratio profile hits '0' (1E-12) set it to 1E-20 at all layers above that layer
    rain_val=1E-8
    loc=np.where(TiOarr <= rain_val)[0]
    if len(loc>1): TiOarr[0:loc[-1]-1]=1E-20
    #loc=np.where(VOarr <= rain_val)[0]
    if len(loc>1):VOarr[0:loc[-1]-1]=1E-20 #VO and TiO rainout togather
    loc=np.where(Naarr <= rain_val)[0]
    if len(loc>1): Naarr[0:loc[-1]-1]=1E-20
    loc=np.where(Karr <= rain_val)[0]
    if len(loc>1):Karr[0:loc[-1]-1]=1E-20
    loc=np.where(FeHarr <= rain_val)[0]
    if len(loc>1):FeHarr[0:loc[-1]-1]=1E-20


    #ackerman & Marley cloud model here
    mmw_cond=100.39#molecular weight of condensate (in AMU)  MgSiO3=100.39
    rho_cond=3250#density of condensate (in kg/m3)           MgSiO3=3250.
    rr=10**(np.arange(-2,2.6,0.1))  #Droplet radii to compute on: MUST BE SAME AS MIE COEFF ARRAYS!!!!!!!!! iF YOU CHANGE THIS IT WILL BREAK
    qc=cloud_profile(fsed,Cld_VMR, Pavg,Pbase)
    r_sed, r_eff, r_g, f_r=particle_radius(fsed,Kzz,mmw,Tavg, Pavg,g0, rho_cond,mmw_cond,qc, rr*1E-6)

    Pref=1.1#10.1  #reference pressure bar-keep fixed--need this for gravity calc in emission

    #computing emission spectrum-----------
 
    wno, Fup, Fdn,dtau,ssa, asym,wts, Fstar,Fup_therm,Fup_ref=rad(xsects, T,P,mmw, Pref,CldOpac, H2Oarr, CH4arr,COarr,CO2arr,NH3arr,Naarr,Karr,TiOarr,VOarr,C2H2arr,HCNarr,H2Sarr,FeHarr,Harr,earr,Hmarr,H2arr,Hearr,RayAmp,RaySlp,f_r, M, Rstar, Rp, D)
    FpFstar=(Fup[:,0]/Fstar)*(Rp/Rstar*0.10279)**2  #"hi res Fp/Fstar"
    Ftoa=Fup[:,0]
    FpFstar_binned=instrument_emission_non_uniform(wlgrid,wno, Fup[:,0], Fstar)*(Rp/Rstar*0.10279)**2 #"binned version"
    p3=time.time()

    chemarr=np.array([P,T, H2Oarr, CH4arr,COarr,CO2arr,NH3arr,Naarr,Karr,TiOarr,VOarr,C2H2arr,HCNarr,H2Sarr,FeHarr,H2arr,Hearr,Harr,earr,Hmarr,qc,r_eff,f_r])
    Fstar_TOA=Fstar*(Rstar*6.95E8)**2/(D*1.496E11)**2

    #binned Fp/Fstar, CK resolution Fp/Fstar, wavenumber grid, abundance profiles/TP, Fplanet, Fstar, Fstar @ Planet, Fplanet Thermal, Fplanet reflected
    return FpFstar_binned,FpFstar,wno,chemarr, Ftoa,Fstar,Fstar_TOA,Fup_therm[:,0],Fup_ref[:,0]


