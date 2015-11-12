# lag_spectra
Makes lag-frequency and lag-energy spectra for cross-spectral analysis of a time
series. The cross spectra and power spectra are read in as a FITS file.

## Contents

### get_lags.py
Computes the phase lag and time lag of bands of interest with a 
reference energy band from the average cross spectrum. Can average over 
frequency and over energy.

Currently (09 Apr 2015) seems to be off by a factor of 2?
Saw this line in Nov 2015 -- don't think it is anymore.

### overplot_lag-energy.py
Plots multiple lag-energy spectra (from different observations, simulations, 
etc.) on the same plot.


## Authors and License
* Abigail Stevens (UvA API)

Pull requests are welcome!

All code is Copyright 2015 The Authors, and is distributed under the MIT 
Licence. See LICENSE for details. If you are interested in the further 
development of lag_spectra, please [get in touch via the issues]
(https://github.com/abigailstev/lag_spectra/issues)!

[![astropy]
(http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/) 