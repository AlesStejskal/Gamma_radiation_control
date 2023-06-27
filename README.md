# Multitone_gamma_conrol.py is the script for numerical calculation of the gamma
# radiation intensity behind the vibrating Moessbauer absorber whose motion profile
# is defined by Fourier series.
#
# The code is optimized for speed. For that reason several preparatory calculations
# appear in the code.

# Cut out the first 340 nanoseconds of the numerical simulation which contain
# numerical artefacts
#
# The script was run by Python version 3.7.4
# with the libraries versions:
# numpy            1.21.6
# matplotlib       3.5.3
