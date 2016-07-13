#!/usr/bin/env bash

################################################################################
##
## Bash script to run simple_cross_spectra.py and simple_plot_lag-freq.py.
##
## Don't give command line arguments. Change things in this script below.
##
## Change the directory names and specifiers before the double '#' row to best
## suit your setup.
##
## Notes: bash 3.*, and Python 2.7.* (with supporting libraries)
## 		  must be installed in order to run this script.
##
## Author: Abigail Stevens <A.L.Stevens at uva.nl>, 2016
##
################################################################################

home_dir=$(ls -d ~)

file_dir="${home_dir}/Downloads"
code_dir="${home_dir}/Dropbox/Research/lag_spectra"

interest_band="${file_dir}/event_125us3_4.lc"
reference_band="${file_dir}/event_125us4_64.lc"
out_base="${file_dir}/event_125us"
plot_ext="eps"

num_seconds=64
testing=0  ## 0 for no, 1 for yes

################################################################################
################################################################################

python "${code_dir}"/simple_cross_spectra.py "${interest_band}" \
        "${reference_band}" --out "${out_base}" --n_sec "${num_seconds}" \
        --test "${testing}"

python "${code_dir}"/simple_plot_lag-freq.py "${out_base}_lag-freq.fits" \
        --out "${out_base}" --ext "${plot_ext}"

open "${out_base}_lag-freq.${plot_ext}"

tput bel  ## Makes the little bash bell ring, so you know the script is done.