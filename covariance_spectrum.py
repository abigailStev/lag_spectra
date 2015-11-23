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
from astropy.io import ascii
import astropy.table
import ccf_lightcurves as ccf_lc
import get_lags
import tools
from datetime import datetime

################################################################################
def fits_out(out_file, in_file, evt_list, meta_dict, lo_freq, up_freq,
        mean_rate_ci, mean_rate_ref, covariance_spectrum, cov_error,
        freq_range):
    """
    Writes the covariance spectrum to a FITS output file.
    Header info is in extension 0, covariance spectra data is in extension 1.

    Parameters
    ----------
    out_file : str
        The full path of the output file.

    in_file : str
        The full path of the cross-spectrum input file.

    evt_list : str
        The full path of the event list of the data.

    meta_dict : dict
        Dictionary of meta-parameters needed for analysis.

    lo_freq, up_freq : float
        The lower and upper frequency bounds to average over for computing the
        lag-energy spectrum. In Hz.

    mean_rate_ci : np.array of floats
        The mean photon count rate of each of the cross-spectral channels of
        interest.

    mean_rate_ref : float
        The mean photon count rate of the cross-spectral reference band.

    """

    chan = np.arange(0, meta_dict['detchans'])

    print "Output sent to: %s" % out_file

    ## Making FITS header (extension 0)
    prihdr = fits.Header()
    prihdr.set('TYPE', "Covariance spectrum")
    prihdr.set('DATE', str(datetime.now()), "YYYY-MM-DD localtime")
    prihdr.set('EVTLIST', evt_list)
    prihdr.set('CS_DATA', in_file)
    prihdr.set('DT', meta_dict['dt'], "seconds")
    prihdr.set('DF', meta_dict['df'], "Hz")
    prihdr.set('N_BINS', meta_dict['n_bins'], "time bins per segment")
    prihdr.set('SEGMENTS', meta_dict['n_seg'], "segments in the whole light curve")
    prihdr.set('SEC_SEG', meta_dict['n_seconds'], "seconds, per segment")
    prihdr.set('EXPOSURE', meta_dict['exposure'], "seconds, of light curve")
    prihdr.set('DETCHANS', meta_dict['detchans'], "Number of detector energy channels")
    prihdr.set('F_RANGE', freq_range, "Number of frequencies kept")
    prihdr.set('LAG_LF', lo_freq, "Hz")
    prihdr.set('LAG_UF', up_freq, "Hz")
    prihdr.set('RATE_CI', str(mean_rate_ci.tolist()), "counts/second")
    prihdr.set('RATE_REF', mean_rate_ref, "counts/second")
    prihdu = fits.PrimaryHDU(header=prihdr)

    ## Making FITS table for lag-frequency plot (extension 1)
    col1 = fits.Column(name='CHANNEL', format='D', array=chan)
    col2 = fits.Column(name='COVARIANCE', unit='counts', format='D',
                       array=covariance_spectrum)
    col3 = fits.Column(name='ERROR', unit='counts', format='D',
                       array=cov_error)
    cols = fits.ColDefs([col1, col2, col3])
    tbhdu = fits.BinTableHDU.from_columns(cols)

    ## If the file already exists, remove it
    assert out_file[-4:].lower() == "fits", \
        'ERROR: Output file must have extension ".fits".'
    if os.path.isfile(out_file):
        os.remove(out_file)

    ## Writing to a FITS file
    thdulist = fits.HDUList([prihdu, tbhdu])
    thdulist.writeto(out_file)

    ## Calling dat_out to save the covariance spectrum to a dat file, for
    ## the heasoft FTOOL ASCII2PHA
    dat_out(out_file, covariance_spectrum, cov_error,
            detchans=meta_dict['detchans'])


################################################################################
def dat_out(out_file, cov_spectrum, cov_error, detchans=64):
    """
    Saves the covariance spectrum to a ".dat" file for later use in ascii2pha,
    for XSPEC.

    Parameters
    ----------
    out_file : str
        The output file. Recommended that this is the same file name as the fits
        output file, but with .dat extension. If it has ".fits" on the end,
        replaces it with ".dat".

    cov_spectrum, cov_error : np.arrays of floats
        1-D arrays (size = detchans) of the covariance spectrum and error,
        averaged over the previously specified frequency range.

    detchans : int
        The number of detector energy channels for the data mode. [64]

    Returns
    -------
    nothing, but writes to a file

    """

    if ".fits" in out_file.lower():
        out_file = out_file.replace(".fits", ".dat")

    out_table = astropy.table.Table([np.arange(detchans), cov_spectrum,
            cov_error], names=("#CHAN", "COV", "ERR"))
    ascii.write(out_table, out_file, format='no_header', fast_writer=True)

    assert os.path.isfile(out_file), "ERROR: Saving covariance spectrum to "\
            "ASCII file did not work."


################################################################################
def plot_in_xspec(meta_dict, out_file, rsp_matrix="./PCU2.rsp"):

    cov_spec_dat = out_file.replace("_cov.fits", "_cov.dat")
    cov_spec_pha = out_file.replace("_cov.fits", "_cov.pha")

    # print cov_spec_dat
    # print cov_spec_pha
    # print os.path.dirname(out_file)
    out_dir = os.path.split(out_file)[0]
    # print "Our dir:", out_dir
    os.chdir(out_dir)
    # subprocess.call(['cd', out_dir])
    # print "Current dir:", os.getcwd()

    # print "Cov spec pha:", cov_spec_pha
    assert os.path.isfile(cov_spec_dat), "ERROR: .dat file of covariance "\
            "spectra does not exist."
    subprocess.call(["cp", cov_spec_dat, "temp_cov.dat"])

    assert os.path.isfile("temp_cov.dat"), "ERROR: temporary .dat file does "\
            "not exist."

    # print cov_spec_dat
    # print "./temp_cov.dat"

    ## Set up the shell command for ascii2pha
    ascii2pha = ["ascii2pha",
                 "infile=./temp_cov.dat",
                 "outfile=./temp_cov.pha",
                 "chanpres=yes",
                 "dtype=2",
                 "qerror=yes",
                 "rows=-",
                 "tlmin=0",
                 "detchans=%d" % meta_dict['detchans'],
                 "pois=no",
                 "telescope=XTE",
                 "instrume=PCA",
                 "detnam=PCU2",
                 "filter=NONE",
                 "exposure=%.6f" % meta_dict['exposure'],
                 "clobber=yes",
                 "respfile=%s" % rsp_matrix]

    ## Execute ascii2pha
    p = subprocess.Popen(ascii2pha, stdout=subprocess.PIPE,
            stdin=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    p.communicate()  ## Waits for the command to finish running.

    assert os.path.isfile("./temp_cov.pha"), "ERROR: ASCII2PHA did not work. "\
            "temp_cov.pha does not exist."

    os.rename("./temp_cov.pha", cov_spec_pha)
    # print "Cov spec pha:", cov_spec_pha
    assert os.path.isfile(cov_spec_pha), "ERROR: ASCII2PHA did not work. "\
            "cov_spec_pha does not exist: %s" % cov_spec_pha

    cov_spec_pha = os.path.relpath(cov_spec_pha)

    ## Writing the xspec script file
    xspec_cmd_file = out_file.replace("_cov.fits", "_cov_xspec.xcm")

    # print "Xspec cmd file:", xspec_cmd_file

    cov_plot_file = out_file.replace("_cov.fits", "_cov.eps")

    with open(xspec_cmd_file, 'w') as out:
        out.write("data %s\n" % cov_spec_pha)
        out.write("ignore **-3.0 20.0-**\n")
        out.write("notice 3.0-20.0\n")
        out.write("ignore 11\n")
        out.write("setplot energy\n")
        out.write("systematic 0.005\n")
        out.write("xsect vern\n")
        out.write("abund wilm\n")
        out.write("mod pow & 0\n")
        out.write("cpd /xw\n")
        out.write("setplot delete all\n")
        out.write("setplot command @cov_spec.pco %s\n" % \
                   os.path.relpath(cov_plot_file))
        out.write("plot eeufspec\n")

    assert os.path.isfile(xspec_cmd_file), "ERROR: Writing xspec script did "\
            "not work."

    xspec_cmd_file = os.path.relpath(xspec_cmd_file)
    # print xspec_cmd_file
    assert os.path.isfile(xspec_cmd_file), "ERROR: Writing xspec script did "\
            "not work."

    ## Set up the shell command for xspec
    xspec = 'xspec %s' % xspec_cmd_file  ## Use this one for debugging.
    # xspec = 'xspec %s > dump.txt' % xspec_cmd_file  ## Use this one to not print to the screen.

    ## Execute xspec script
    p = subprocess.Popen(xspec, shell=True)
            ## Setting shell=True allows you to run non- standard shell
            ## commands, and Popen lets us redirect the output
    p.communicate()  ## Waits for the command to finish running.

    assert os.path.isfile(cov_plot_file), "ERROR: Xspec didn't run correctly. "\
            "Covariance spectrum plot was not created."


################################################################################
def bias_term(power_ci, power_ref, mean_rate_ci, mean_rate_ref, meta_dict,
        n_freq):
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
    Pnoise_ci = mean_rate_ci * 2.0  # absolute rms units

    abs_ci = power_ci * (2.0 * meta_dict['dt'] / float(n_freq))
    abs_ref = power_ref * (2.0 * meta_dict['dt'] / float(n_freq))
    abs_ref = np.resize(np.repeat(abs_ref, meta_dict['detchans']),
        np.shape(abs_ci))

    temp_a = (abs_ref - Pnoise_ref) * Pnoise_ci
    temp_b = (abs_ci - Pnoise_ci) * Pnoise_ref
    temp_c = Pnoise_ref * Pnoise_ci

    n_squared = (temp_a + temp_b + temp_c) / (n_freq * meta_dict['n_seg'])
    return n_squared


################################################################################
def main(in_file, out_file, energies_file, plot_root="./covariance",
        prefix="--", plot_ext="eps", rsp_matrix="./PCU2.rsp", lo_freq=1.0,
        up_freq=10.0):
    """
    Compute the phase lag and time lag from the average cross spectrum.

    Note that power_ci, power_ref, and cs_avg should be unnormalized and without
    noise subtracted.

    """

    energies_tab = np.loadtxt(energies_file)

    ## Get necessary information and data from the input file
    freq, cs_avg, power_ci, power_ref, meta_dict, mean_rate_ci, mean_rate_ref, \
            evt_list = get_lags.get_inputs(in_file)

    low_freq_mask = freq >= lo_freq
    freq = freq[low_freq_mask]
    cs_avg = cs_avg[low_freq_mask, :]
    power_ci = power_ci[low_freq_mask, :]
    power_ref = power_ref[low_freq_mask]

    upper_freq_mask = freq <= up_freq
    freq = freq[upper_freq_mask]
    cs_avg = cs_avg[upper_freq_mask, :]
    power_ci = power_ci[upper_freq_mask, :]
    power_ref = power_ref[upper_freq_mask]

    freq_range = len(freq)

    # print np.shape(cs_avg)
    # print np.shape(power_ci)
    # print np.shape(power_ref)

    #####################################
    ## Computing the covariance spectrum
    #####################################

    # print meta_dict['df']

    cs_bias = bias_term(power_ci, power_ref, mean_rate_ci, mean_rate_ref, \
            meta_dict, freq_range)

    # print cs_bias
    # print np.shape(cs_bias)
    modsquare_min_bias = np.real(np.sum(cs_avg * np.conj(cs_avg) - cs_bias,
            axis=0))

    # print type(modsquare_min_bias[0])
    # print np.shape(modsquare_min_bias)

    abs_ref = power_ref * (2.0 * meta_dict['dt'] / float(freq_range)) - 0 ## Poisson errors on IR are weird.

    ref_variance = np.sum(abs_ref * meta_dict['df'])
    ref_rms = np.sqrt(ref_variance)

    covariance_spectrum = np.sqrt(modsquare_min_bias) * freq_range / ref_rms

    cov_errors = np.ones(meta_dict['detchans'])
    # print covariance_spectrum
    print np.shape(covariance_spectrum)
    print np.shape(cov_errors)

    ##########
    ## Output
    ##########

    fits_out(out_file, in_file, evt_list, meta_dict, lo_freq, up_freq,
            mean_rate_ci, mean_rate_ref, covariance_spectrum, cov_errors,
            freq_range)

    #######################
    ## Plotting (in XSPEC)
    #######################

    plot_in_xspec(meta_dict, out_file, rsp_matrix)


################################################################################
if __name__ == "__main__":

    #########################################
    ## Parse input arguments and call 'main'
    #########################################

    parser = argparse.ArgumentParser(usage="python covariance_spectrum.py "\
            "infile outfile [OPTIONAL ARGUMENTS]", description=__doc__,
            epilog="For optional arguments, default values are given in "\
            "brackets at end of description.")

    parser.add_argument('infile', help="Name of the FITS file containing the "\
            "cross spectrum, power spectrum of the channels of interest, and "\
            "power spectrum of the reference band. Same as input file for get_"\
            "lags.py. In format '*_cs.fits'.")

    parser.add_argument('outfile', help="Name of the FITS file to write the "\
            "covariance spectrum to, in format '*_cov.fits'.")

    parser.add_argument('energies_file', help="Name of the txt file containing "\
            "a list of the keV energies that map to the detector energy "\
            "channels. Generated in rxte_reduce/channel_to_energies.py.")

    parser.add_argument('-o', dest='plot_root', default="./plot", help="Root "\
            "name for plots generated, to be appended with '_lag-freq.(ext"\
            "ension)' and '_lag-energy.(extension)'. [./plot]")

    parser.add_argument('--prefix', dest="prefix", default="--",
            help="The identifying prefix of the data (object nickname or "\
            "data ID). [--]")

    parser.add_argument('--ext', dest='plot_ext', default='eps',
            help="File extension for the plots. Do not include the dot. [eps]")

    parser.add_argument('--rsp', dest='rsp_matrix', default='./PCU2.rsp',
            help="Local path name of the detector response matrix. "\
            "[./PCU2.rsp]")

    parser.add_argument('--lf', dest='lo_freq', default=1.0,
            type=tools.type_positive_float, help="The lower limit of the "\
            "frequency range for the lag-energy spectrum to be computed for, "\
            "in Hz. [1.0]")

    parser.add_argument('--uf', dest='up_freq', default=10.0,
            type=tools.type_positive_float, help="The upper limit of the "\
            "frequency range for the lag-energy spectrum to be computed for, "\
            "in Hz. [10.0]")

    args = parser.parse_args()

    main(args.infile, args.outfile, args.energies_file, plot_root=args.plot_root,
            prefix=args.prefix, plot_ext=args.plot_ext,
            rsp_matrix=args.rsp_matrix, lo_freq=args.lo_freq,
            up_freq=args.up_freq)