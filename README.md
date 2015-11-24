# lag_spectra
Computes and plots frequency lags, energy lags, and covariance spectra for 
cross-spectral analysis of a time series. The cross spectra and power spectra 
are read in as a FITS file.

## Contents

### covariance_spectrum.py
Computes the covariance spectrum from the averaged cross spectrum. 
See Uttley et al 2014 section 2 for the relevant equations and physical 
explanation.

### get_lags.py
Computes the phase lag and time lag of bands of interest with a 
reference energy band from the averaged cross spectrum. Can average lags over 
frequency and over energy channel. 
See Uttley et al 2014 section 2 for the relevant equations and physical 
explanations.

### overplot_lag-energy.py
Plots multiple lag-energy spectra (from different observations, simulations, 
etc.) on the same plot.


[![astropy]
(http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/) 

## Authors
* Abigail Stevens (UvA API)

## Collaborators
* Phil Uttley (UvA API)
* Federico Vincentelli (INAF Roma, INAF Brera)

Pull requests are welcome!

## License

All code is Copyright 2015 The Authors, and is distributed under the MIT 
Licence. See LICENSE for details. If you are interested in the further 
development of lag_spectra, please [get in touch via the issues]
(https://github.com/abigailstev/lag_spectra/issues)!
