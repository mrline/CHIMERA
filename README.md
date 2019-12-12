# CHIMERA Exo-Atmosphere Retrieval Code
Welcome to the CHIMERA atmospheric retrieval code github!  What is CHIMERA? It stands for CaltecH Inverse ModEling and Retrieval Algorithms (yes, totally a backranym) originally developed back in the day when I was a grad student (~2012, originally in IDL no less...). It has since morphed over the past ~7 years into something more functional....maybe. 

What is uploaded here is the latest and "greatest" version in Pure Python 3 and is accelerated using python's anaconda numba package (via the @jit decorator).  Long story short, with this code you should be able to "retrieve" on a typical HST WFC3 data set in both emission and transmission and/or create your own simulated JWST spectra (in emission/transmission) within the "chemically-consistent" (whereby equilibrium "quench" chemistry is assumed) or the classic "free retrieval" (retrieve a bunch of molecules) frameworks.   To get you started, there are numerous jupyter notebooks (/MASTER_CODE/.ipynb) for differing scenarios to get you started. I encourage you to work through those to gain familiarity.  

The code is meant to be flexible with numerous radiative transfer related modules in /MASTER_CODE/fm.py. Certainly this code does not take into account all infinite assumptions/possibilities for parameterizations.  I leave it to the user to modify at will (e.g., change up the TP profile parameterization, or clouds, etc.).  Have a look. Right now it's partially commented, I'll do more when/if I have time. 


"INSTALL" INSTRUCTIONS:
1. Clone/download this project 
2. Go into the ABSCOEFF_CK folder. You will find a readme. It tells you to download a lot of large files (opacities) here:
https://www.dropbox.com/sh/o4p3f8ukpfl0wg6/AADBeGuOfFLo38MGWZ8oFDX2a?dl=0
3.  Make sure you have python3/anaconda3 with the numba package. I'm running on a MAC OS 10.13.6 (High Sierra) with python 3.6.4.
4. Once everhting is downloaded, go into the MASTER_CODE folder go through the notebook tutorials (start with CHIMERA_TRANSMISSION_DEMO_WASP43b_WFC3.ipynb).  Further instructions await in the notebook.
5. Install the Nested Sampler packages (dynesty: https://github.com/joshspeagle/dynesty) and PyMultiNest (https://johannesbuchner.github.io/PyMultiNest/).  There are more pymultinest specific install instructions in the CHIMERA_TRANSMISSION_DEMO_WASP43b_WFC3.ipynb demo.  

