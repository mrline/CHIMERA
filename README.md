# CHIMERA Exo-Atmosphere Retrieval Code
Welcome to the CHIMERA atmospheric retrieval code github!  What is CHIMERA? It stands for CaltecH Inverse ModEling and Retrieval Algorithms (yes, totally a backronym) originally developed back in the day when I was a grad student (~2012, originally in IDL no less...). It has since morphed over the past ~7 years into something more functional....maybe. 

What is uploaded here is the latest and "greatest" version in Pure Python 3 and is accelerated using python's anaconda numba package (via the @jit decorator).  Long story short, with this code you should be able to "retrieve" on a typical HST WFC3 data set in both emission and transmission and/or create your own simulated JWST spectra (in emission/transmission) within the "chemically-consistent" (whereby equilibrium "quench" chemistry is assumed) or the classic "free retrieval" (retrieve a bunch of molecules) frameworks.   To get you started, there are numerous jupyter notebooks (/MASTER_CODE/.ipynb) for differing scenarios to get you started. I encourage you to work through those to gain familiarity.  

The code is meant to be flexible with numerous radiative transfer related modules in /MASTER_CODE/fm.py. Certainly this code does not take into account all infinite assumptions/possibilities for parameterizations.  I leave it to the user to modify at will (e.g., change up the TP profile parameterization, or clouds, etc.).  Have a look. Right now it's partially commented, I'll do more when/if I have time. 

Sorry. No fancy "read-the-docs" page. That's ok. Words are words regardless of the format.

"INSTALL" INSTRUCTIONS:
1. Clone/download this project 
2. Go into the ABSCOEFF_CK folder. You will find a readme. It tells you to download a lot of large files (opacities) here:
https://www.dropbox.com/sh/o4p3f8ukpfl0wg6/AADBeGuOfFLo38MGWZ8oFDX2a?dl=0
3.  Make sure you have python3/anaconda3 with the numba package. I'm running on a MAC OS 10.13.6 (High Sierra) with python 3.6.4.
4. Once everything is downloaded, go into the MASTER_ROUTINES folder go through the notebook tutorials (start with CHIMERA_TRANSMISSION_DEMO_WASP43b_WFC3.ipynb).  Further instructions await in the notebook.
5. Install the Nested Sampler packages (dynesty: https://github.com/joshspeagle/dynesty) and PyMultiNest (https://johannesbuchner.github.io/PyMultiNest/).  There are more pymultinest specific install instructions in the CHIMERA_TRANSMISSION_DEMO_WASP43b_WFC3.ipynb demo.  
6. Have fun!

Note: This code is very much **not a black box** (sorry?). The jupyter notebooks and the "call_pymultinest.../plot_PMN..." routines are the closest to black box, but with the intent of forcing the user to go through some "training".  Otherwise, use the notebooks as a template for modification if you want to add new parameters etc.  The "fx" routines are meant to be flexible enough to swap out different parameterizations for various custom fizzix (e.g., temperature profiles, cloud models, etc.).   In general, radiative transfer simply requires opacities and temperatures.  However you can ram those into the "tran" or "rad" routines, have at it.  

# Cliff Notes of Features/Methods:
Correlated-K opacity treatment (Lacis & Oinas 1991; Irwin et al. 2008) in both emission and transmission.  For emission the "resort-rebin" on-the-fly gas mixing procedure is used (Molliere et al. 2015; Amundsen et al. 2017).  For transmission, seperate gas transmittances within each ray-cell are multiplied together.  CK's generated from a variety of line-by-line cross-section databases, but most come from what is described in Freedman et al. 2014.  This is ever evolving... 

Multiple scattering "emission" radiative transfer for both "internal" (planckian) source functions and external stellar flux (for reflection component) computed with the Two-Stream Source Function Technique (Toon et al. 1989, Marley et al. 2000).

Ackerman & Marley 2001 "eddy-sed" cloud parameterization. This takes into account the change in particle size with altitude assuming said sizes are governed by a balance of sedimentation and uplift (through eddy diffusion).  Can specify a condensate within the currently available sources (in the /ABSCOEFF_CK/MIE_COEFFS/ folder downloaded in step (2)). 

Also has the "classic" boring gray cloud (through a single uniform opacity...not a cloud-top-pressure) and power law "haze" scattering.

Chemically-Consistent (CC) or Free Retrieval.  The CC is based on a pre-computed chemistry grid as a function of T, P, Metallicity, and C/O (Lodders 2009 abundances).  Hacked quenching through the nitrogen and carbon species quench free parameters. Certainly an atmosphere can have many many more composition dimensions (all 118 elements if you'd like).  Feel free to regenerate a chem grid with your favorite gibbs-free energy minimizer.  Whatever gases are included as opacities can be "free" retrieved.

Double-Gray (Guillot 2010; Parmentier & Guillot 2014; Line et al. 2013) Temperature-profile parameterization (yes yes, it goes isothermal at low pressures).  Feel free to add other TP profile functions (e.g., the Madhusudhan & Seager 2009 is a nice one to try).

Flexible for use with multiple Bayesian samplers. Right now it is (the notebooks) set up for both Dynesty and PyMultiNest.  Feel free to try it with EMCEE.  

Can run on a multi-core laptop (takes a few hours with the notebooks), but is really designed to run on clusters. PyMultiNest is the one to use for cluster computing.

Includes multiple "call_pymultinest..." routines for various scenarios as well as corresponding plotting routines (plot_PMN...).  To be honest, I didn't spend a lot of time making the plotting routines fancy.  They are self-explanatory...

# Things I didn't do b/c I am lazy (but are floating around somewhere on the cluster...)

Filter/filter profiles for Spitzer/TESS/Kepler. This is fairly easy to do.  Modify the instrument_emission/transmission_non_uniform routines in fm.py.  You would load in the filter profiles and putz around with the arrays in that function. For spitzer don't forget to do lambdaFlambda in the profile integrals (depending on how the filter profile is defined..).

Offset parameters for combined datasets (e.g., STIS+WFC3). Though this is also "easy" to do.

The spot contamination thing in transmission (e.g., Rackham et al. 2017).  This can be done (e.g., Iyer & Line 2019) by loading in a "pre-wavelength-interpolated" stellar model grid (as a function of Teff) into xsects_HST/JWST, a linear interpolating function in fx_trans/trans_free, and the simple contamination formula (along with the corresponding parameters).

This version doesn't have the "patchy clouds" (e.g., Line & Parmentier 2016), though it is straight forward to add by linearly combining a "clear" and "cloudy" atmosphere (the output of "tran"--make another output and but zero out f_r) in the fx_trans.. routines.

Make it "brown dwarf" friendly.  However, easily done, just get rid of "Fstar" in Fp/Fstar.  Probably best to make a new "fx" function (in fm.py).  Probably don't use the current TP profile parameterzation as that is not good for BDs (e.g., eddington approximation is not wise for this).  

Add support for multiple "TP" profiles (e.g., the Feng et al. 2016 thing).  This requires care with geometry. Linear combinations of patches/columns is easy enough by calling "rad" multiple times with different TPs and weighting the patches appropriately (which depends on your geometry method).


# Code History (from 2012...) tl;dr
The primordial version (Line et al. 2012) used the Reference Forward Model (http://eodg.atm.ox.ac.uk/RFM/#cant) for thermal emission combined with the "optimal estimation" approach (Rodgers 2000; Lee et al. 2012) (written in IDL!) to explore spectral "information content" with applications to the near infrared HST NICMOS (yeah...back in the day...) spectrum of HD 189733b. HITRAN/HITEMP opacities were used. This was before ExoMol really started to crank out molecules.  

In 2013 (Line et al. 2013a) I/we (fellow down-the-hallmate Xi Zhang) ditched RFM and wrote our own simple non-scattering emission RT (again in IDL!).  Since MCMC was becoming a "thing" (Madhusudhan et al. 2011; Benneke & Seager 2012 and too many conversations with fellow down-the-hall mate, Aaron Wolf), we decided to test the differences in various "parameter estimators". These were optimal estimation (the classic planetary tool of choice), boot-strap monte carlo, and markov chain monte carlo.  There weren't many "git-able" MCMC routines in IDL in 2013 (and I had not yet heard of the pythons), so we wrote our own Differential Evolution MCMC (ter Braak 2006;2008). We found, in general, that these approaches agreed when the data were "good"--even optimal estimation (with its gaussian posterior approximation, like any non-linear minimizer)--but disagreed when the data were sparse (low wavelength coverage, low SNR etc.)--unsurprisingly.  Shortly there-after we decided to do "useful" science by performing a "uniform" analysis on the currently available secondary eclipse spectra--around 9 planets (mind you, this is before HST WFC3 spatial scan was a thing, so all the data was...meh). Inspired by the notion of "high C/O" from the Madhushdhan et al. 2011 nature paper, we wanted to see if that held true for other planets....MORE tl;dr coming      




# Credits
Lots of folks have helped with/contributed to/influenced various flavors of the code and key relevant discussions over the years.  So thanks to (in no particular order...probably chronological) Xi Zhang, Aaron Wolf, Leigh Fletcher, Gautam Vasisht, Pin Chen, Mark Swain, Yuk Yung, Josh Kammer, Heather Knutson, Jonathan Fortney, Jacob Lustig-Yeager,  Kat Feng, Vivien Parmentier, Natasha Batalha, Mark Marley, Richard Freedman, Cesar Montero, Kyle Luther, Dylan Teal, Jacob Bean, Laura Kreidberg, Kevin Stevenson, Tom Greene, Jean-Michel Desert, Ingo Waldmann, Joe Barstow Eberhardt, Marco Rochetto, Ryan Garland, Chuhong Mai, Aisha Iyer...

