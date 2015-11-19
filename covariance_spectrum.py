#!/usr/bin/env python
"""
Computes the covariance spectrum for a data set. Equations come from the
Uttley et al. 2014 review paper, section 2.2.3. Assumes the averaged power
spectrum of the reference band, power spectra of the channels of interest, and
cross spectrum have already been computed.

"""
__author__ = 'Abigail Stevens <A.L.Stevens at uva.nl>'
__year__ = "2015"

import numpy as np
import argparse
import os
import subprocess
from astropy.io import fits
import ccf_lightcurves as ccf_lc
import get_lags


def bias_term(power_ci, power_ref, mean_rate_ci, mean_rate_ref, meta_dict):
    """
    Assumes power spectra are raw (not at all normalized, and not Poisson-noise-
    subtracted).

    :param power_ci:
    :param power_ref:
    :param n_bins:
    :param n_seg:
    :return:
    """
    Pnoise_ref = 0
    Pnoise_ci = 2.0 * mean_rate_ci  # absolute rms units

    abs_ci = power_ci * (2.0 * meta_dict['dt'] / np.float(meta_dict['n_bins']))
    abs_ref = power_ref * (2.0 * meta_dict['dt'] / np.float(meta_dict['n_bins']))
    abs_ref = np.resize(np.repeat(abs_ref, meta_dict['detchans']),
        np.shape(abs_ci))

    temp_a = (abs_ref - Pnoise_ref) * Pnoise_ci
    temp_b = (abs_ci - Pnoise_ci) * Pnoise_ref
    temp_c = Pnoise_ref * Pnoise_ci

    n_squared = (temp_a + temp_b + temp_c) / (meta_dict['n_bins'] * \
                                              meta_dict['n_seg'])

    return n_squared

################################################################################
def main(in_file, out_file, energies_file, plot_root="./covariance",
        prefix="--", plot_ext="eps", lo_freq=1.0, up_freq=10.0):
    """
    Computes the phase lag and time lag from the average cross spectrum. Note
    that power_ci, power_ref, and cs_avg should be unnormalized and without
    noise subtracted.

    """

    energies_tab = np.loadtxt(energies_file)

    ## Get necessary information and data from the input file
    freq, cs_avg, power_ci, power_ref, dt, n_bins, detchans, n_seconds, \
            n_seg, mean_rate_ci, mean_rate_ref, evt_list = get_inputs(in_file)

    #####################################
    ## Computing the covariance spectrum
    #####################################

    cs_bias = bias_term(power_ci, power_ref, mean_rate_ci, mean_rate_ref, \
            n_bins, n_seg)

    ##########
    ## Output
    ##########



    ############
    ## Plotting
    ############


################################################################################
if __name__ == "__main__":

    ##############################################
    ## Parsing input arguments and calling 'main'
    ##############################################

    parser = argparse.ArgumentParser(usage="python covariance_spectrum.py "\
            "infile outfile [OPTIONAL ARGUMENTS]", description=__doc__,
            epilog="For optional arguments, default values are given in "\
            "brackets at end of description.")

    parser.add_argument('infile', help="Name of the FITS file containing the "\
            "cross spectrum, power spectrum of the channels of interest, and "\
            "power spectrum of the reference band. Same as input file for get_"\
            "lags.py")

    parser.add_argument('outfile', help="Name of the FITS file to write the "\
            "covariance spectrum to.")

    parser.add_argument('energies_tab', help="Name of the txt file containing "\
            "a list of the keV energies that map to the detector energy "\
            "channels.")

    parser.add_argument('-o', dest='plot_root', default="./plot", help="Root "\
            "name for plots generated, to be appended with '_lag-freq.(ext"\
            "ension)' and '_lag-energy.(extension)'. [./plot]")

    parser.add_argument('--prefix', dest="prefix", default="--",
            help="The identifying prefix of the data (object nickname or "\
            "data ID). [--]")

    parser.add_argument('--ext', dest='plot_ext', default='eps',
            help="File extension for the plots. Do not include the dot. [eps]")

    parser.add_argument('--lf', dest='lo_freq', default=1.0,
            type=tools.type_positive_float, help="The lower limit of the "\
            "frequency range for the lag-energy spectrum to be computed for, "\
            "in Hz. [1.0]")

    parser.add_argument('--uf', dest='up_freq', default=10.0,
            type=tools.type_positive_float, help="The upper limit of the "\
            "frequency range for the lag-energy spectrum to be computed for, "\
            "in Hz. [10.0]")

    args = parser.parse_args()

    main(args.infile, args.outfile, args.energies_tab, args.plot_root,
            args.prefix, plot_ext=args.plot_ext, lo_freq=args.lo_freq,
            up_freq=args.up_freq)