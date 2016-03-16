#!/usr/bin/env python
"""
Computes the covariance spectrum for a data set. Equations come from the
Uttley et al. 2014 review paper, section 2. Assumes the averaged power
spectrum of the reference band, power spectra of the channels of interest, and
cross spectrum have already been computed, and are not normalized or Poisson-
noise-subtracted.

Notes: HEASOFT 6.11.* and Python 2.7.* (with supporting libraries) must be
installed in order to run this script.

Files created
-------------
*_cov.fits :
    The covariance spectrum, in FITS format (more robust for storing floating
    point numbers).

*_cov.dat :
    The covariance spectrum, in ASCII format (for use in ASCII2PHA). Gets
    deleted after *_cov.pha is created (to save disk space).

*_cov.pha :
    The covariance spectrum, in .pha spectrum format, for use in XSPEC.

temp_cov.dat, temp_cov.pha :
    Temporary local files created because ASCII2PHA doesn't like how long the
    filenames are. Get deleted after use to save disk space.

*_cov_xspec.xcm :
    XSPEC script that plots the covariance spectrum unfolded through the
    detector response matrix and a power law of slope 1.

dump.txt
    Temporary local file that the XSPEC output is piped (dumped) into.

*_cov.eps :
    Plot of said covariance spectrum created by the XSPEC script.

"""
__author__ = 'Abigail Stevens <A.L.Stevens at uva.nl>'
__year__ = "2015-2016"

import numpy as np
import argparse
import os
import subprocess
from astropy.table import Table
from astropy.io import ascii
from datetime import datetime

## These are things I've written.
## Their directories are in my PYTHONPATH bash environment variable.
import ccf_lightcurves as ccf_lc
import get_lags
import tools

################################################################################
def fits_out(out_file, in_file, evt_list, meta_dict, lo_freq, up_freq,
        mean_rate_ci, mean_rate_ref, covariance_spectrum, cov_error,
        freq_range):
    """
    Writes the covariance spectrum to a FITS output file from an astropy table.
    Data and header are in extension 1.

    Parameters
    ----------
    out_file : str
        The full path of the output file, in format '*_cov.fits'.

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

    Returns
    -------
    Nothing, but writes to the file '*_cov.fits' (and calls dat_out which writes
    '*_cov.dat').

    """

    ## Check that filename has FITS file extension.
    assert out_file[-4:].lower() == "fits", "ERROR: Output file must have "\
            "extension '.fits'."

    print "Output sent to: %s" % out_file

    out_table = Table([np.arange(meta_dict['detchans']),
            covariance_spectrum, cov_error], names=("CHAN", "COVARIANCE",
            "ERROR"))

    out_table.meta['TYPE'] = "Covariance spectrum"
    out_table.meta['DATE'] = str(datetime.now())
    out_table.meta['EVTLIST'] = evt_list
    out_table.meta['CS_DATA'] = in_file
    out_table.meta['DT'] = meta_dict['dt']
    out_table.meta['DF'] = meta_dict['df']
    out_table.meta['N_BINS'] = meta_dict['n_bins']
    out_table.meta['SEGMENTS'] = meta_dict['n_seg']
    out_table.meta['SEC_SEG'] = meta_dict['n_seconds']
    out_table.meta['EXPOSURE'] = meta_dict['exposure']
    out_table.meta['DETCHANS'] = meta_dict['detchans']
    out_table.meta['F_RANGE'] = freq_range
    out_table.meta['LAG_LF'] = lo_freq
    out_table.meta['LAG_UF'] = up_freq
    out_table.meta['RATE_CI'] = str(mean_rate_ci.tolist())
    out_table.meta['RATE_REF'] = mean_rate_ref

    out_table.write(out_file, overwrite=True)

    ## Calling dat_out to save the covariance spectrum to a .dat file, for
    ## the heasoft FTOOL ASCII2PHA
    dat_out(out_file, out_table)


################################################################################
def dat_out(out_file, out_table):
    """
    Save the covariance spectrum to a ".dat" file for later use in ascii2pha,
    for XSPEC.

    Parameters
    ----------
    out_file : str
        The output file. Recommended that this is the same file name as the fits
        output file, but with .dat extension. If it has ".fits" on the end,
        replaces it with ".dat".

    out_table : astropy.table.Table
        Table containing 3 columns: energy channel (int), covariance (float),
        and error on covariance (float). All three columns are 1-dimension with
        size = (detchans).

    Returns
    -------
    Nothing, but writes to a file '*_cov.dat'.

    """

    if ".fits" in out_file.lower():
        out_file = out_file.replace(".fits", ".dat")

    ascii.write(out_table, out_file, format='no_header', fast_writer=True)

    assert os.path.isfile(out_file), "ERROR: Saving covariance spectrum to "\
            "ASCII file *_cov.dat did not work."


################################################################################
def var_and_rms(power, df):
    """
    Computes the variance and rms (root mean square) of a power spectrum.
    Assumes the negative-frequency powers have been removed. DOES NOT WORK ON
    2-D POWER ARRAYS! Not sure why.

    TODO: cite textbook or paper. Probably Michiel's big review paper from 2005?

    Parameters
    ----------
    power : np.array of floats
        1-D array (size = n_bins/2+1) of the raw power at each of the *positive*
        Fourier frequencies.

    df : float
        The step size between Fourier frequencies.

    Returns
    -------
    variance : float
        The variance of the power spectrum.

    rms : float
        The rms of the power spectrum.

    """

    # print "Shape power:", np.shape(power)
    # print "Nonzero power:", power[np.where(power<=0.0)]
    variance = np.sum(power * df)
    # print np.shape(variance)
    # print "Variance:", variance
    # if variance > 0:
    #     rms = np.sqrt(variance)
    # else:
    #     rms = np.nan
    rms = np.where(variance >= 0, np.sqrt(variance), np.nan)
    # print "rms:", rms
    return variance, rms


################################################################################
def plot_in_xspec(meta_dict, out_file, rsp_matrix="./PCU2.rsp", prefix="--"):
    """
    Save the covariance spectrum as a local .dat file, converts that to a .pha
    spectrum file using ASCII2PHA, writes an XSPEC script to plot the covariance
    spectrum, and executes that script.

    Parameters
    ----------
    meta_dict : dict
        Dictionary of meta-parameters needed for analysis.

    out_file : str
        Name of the output file for the covariance spectrum.
        In format '*_cov.fits'.

    rsp_matrix : str
        Local path name of the detector response matrix. [./PCU2.rsp]

    prefix : str
        Identifying prefix of the data (object nickname or data ID). [--]

    Files created
    -------------
    *_cov.dat
        The covariance spectrum, in ASCII format. Gets deleted after *_cov.pha
        is successfully created.

    *_cov.pha
        The covariance spectrum, in .pha spectrum format, for use in XSPEC.

    temp_cov.dat, temp_cov.pha
        Temporary local files created because ASCII2PHA doesn't like how long
        the filenames are. Get deleted after use.

    *_cov_xspec.xcm
        XSPEC script that plots the covariance spectrum unfolded through the
        detector response matrix and a power law of slope 1.

    dump.txt
        Temporary local file that the XSPEC output is piped (dumped) into.

    *_cov.eps
        Plot of said covariance spectrum created by the XSPEC script.

    """
    cov_spec_dat = out_file.replace("_cov.fits", "_cov.dat")
    cov_spec_pha = out_file.replace("_cov.fits", "_cov.pha")

    out_dir = os.path.split(out_file)[0]
    os.chdir(out_dir)

    assert os.path.isfile(cov_spec_dat), "ERROR: .dat file of covariance "\
            "spectra does not exist."
    subprocess.call(["cp", cov_spec_dat, "temp_cov.dat"])

    assert os.path.isfile("temp_cov.dat"), "ERROR: temporary .dat file does "\
            "not exist."

    assert os.path.isfile(rsp_matrix), "ERROR: Response matrix does not exist."

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
                 "respfile=%s" % os.path.relpath(rsp_matrix)]

    ## Execute ascii2pha
    p = subprocess.Popen(ascii2pha, stdout=subprocess.PIPE,
            stdin=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    p.communicate()  ## Waits for the command to finish running.

    # print os.getcwd()
    # print "./temp_cov.pha"

    assert os.path.isfile("./temp_cov.pha"), "ERROR: ASCII2PHA did not work. "\
            "temp_cov.pha does not exist."

    os.rename("./temp_cov.pha", cov_spec_pha)
    # print "Cov spec pha:", cov_spec_pha
    assert os.path.isfile(cov_spec_pha), "ERROR: ASCII2PHA did not work. "\
            "cov_spec_pha does not exist: %s" % cov_spec_pha

    ## Removing the .dat file, once the .pha is created, to save disk space
    ## Also removing the temporary file (.pha temp is already removed/renamed)
    os.remove(cov_spec_dat)
    os.remove("./temp_cov.dat")

    ## Writing the xspec script file
    xspec_cmd_file = out_file.replace("_cov.fits", "_cov_xspec.xcm")

    # print "Xspec cmd file:", xspec_cmd_file

    cov_plot_file = out_file.replace("_cov.fits", "_cov.eps")

    with open(xspec_cmd_file, 'w') as out:
        out.write("data %s\n" % os.path.relpath(cov_spec_pha))
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
    # xspec = 'xspec %s' % xspec_cmd_file  ## Use this one for debugging.
    xspec = 'xspec %s > dump.txt' % xspec_cmd_file  ## Use this one to not print to the screen.

    ## Execute xspec script
    p = subprocess.Popen(xspec, shell=True)
            ## Setting shell=True allows you to run non- standard shell
            ## commands, and Popen lets us redirect the output
    p.communicate()  ## Waits for the command to finish running.

    assert os.path.isfile(cov_plot_file), "ERROR: Xspec didn't run correctly. "\
            "Covariance spectrum plot was not created."

    subprocess.call(['open', cov_plot_file])


################################################################################
def bias_term(power_ci, power_ref, mean_rate_ci, mean_rate_ref, meta_dict,
        n_freq):
    """
    Compute the bias term to be subtracted off the cross spectrum to compute
    the covariance spectrum. Equation in Equation in footnote 4 (section 2.1.3,
    page 12) of Uttley et al. 2014.

    Assumes power spectra are absolute rms^2 normalized and NOT Poisson-noise-
    subtracted.

    Parameters
    ----------
    power_ci : np.array of floats
        2-D array of the power in the channels of interest, absolute rms^2 norm
        and not Poisson-noise-subtracted), of the frequencies to be averaged
        over. Size = (n_freq, detchans)

    power_ref : np.array of floats
        1-D array of the power in the reference band, absolute rms^2 norm and
        not Poisson-noise-subtracted), of the frequencies to be averaged over.
        Size = (n_freq).

    mean_rate_ci : np.array of floats
        1-D array of the mean count rate in the channels of interest, in cts/s.
        Size = (detchans).

    mean_rate_ref : float
        Mean count rate in the reference band, in cts/s.

    meta_dict : dict
        Dictionary of meta-parameters needed for analysis.

    n_freq : int
        Number of frequency bins that will be averaged together for the
        covariance spectrum.

    Returns
    -------
    n_squared : float
        The bias term to be subtracted off the cross spectrum for computing the
        covariance spectrum. Equation in footnote 4 (section 2.1.3, page 12) of
        Uttley et al. 2014.

    """
    ## Compute the Poisson noise level in absolute rms^2 units
    Pnoise_ref = 2.0 * mean_rate_ref
    Pnoise_ci = 2.0 * mean_rate_ci

    ## Reshaping (broadcasting) the ref to have same size as ci
    power_ref = np.resize(np.repeat(power_ref, meta_dict['detchans']),
        np.shape(power_ci))

    temp_a = (power_ref - Pnoise_ref) * Pnoise_ci
    temp_b = (power_ci - Pnoise_ci) * Pnoise_ref
    temp_c = Pnoise_ref * Pnoise_ci

    n_squared = (temp_a + temp_b + temp_c) / (n_freq * meta_dict['n_seg'])

    return n_squared

################################################################################
def compute_coherence(cross_spec, power_ci, power_ref, mean_rate_ci,
        mean_rate_ref, meta_dict, n_range):
    """
    Compute the coherence of the cross spectrum. Coherence equation from
    Uttley et al 2014 eqn 11, bias term equation from footnote 4 on same page.

    Parameters
    ----------
    cross_spec : np.array of complex numbers
        1-D array of the cross spectrum, averaged over the desired energy
        range or frequency range. Size = detchans (if avg over freq) or
        n_bins/2+1 (if avg over energy). Should be absolute rms^2 normalized and
        NOT noise-subtracted. Eqn 9 of Uttley et al 2014.

    power_ci : np.array of floats
        1-D array of the channel of interest power spectrum, averaged over the
        desired energy range or frequency range. Size = detchans (if avg over
        freq) or n_bins/2+1 (if avg over energy). Should be absolute rms^2
        normalized and NOT Poisson-noise-subtracted.

    power_ref : np.array of floats
        1-D array of the reference band power spectrum, possibly averaged over
        the desired frequency range. Size = n_bins/2+1 (if avg over energy) or
        detchans (if avg over freq; tiled down second axis to be same size as
        power_ci). Should be absolute rms^2 normalized and NOT Poisson-noise-
        subtracted.

    mean_rate_ci : np.array of floats
        1-D array of the mean count rates of the channels of interest, in cts/s.
        Size = detchans (if avg over freq), or 1 (if avg over energy).

    mean_rate_ref : float
        Mean count rate of the reference band, in cts/s.

    meta_dict : dict
        Dictionary of meta-paramters needed for analysis.

    n_range : int
        Number of bins averaged over for lags. Energy bins for frequency lags,
        frequency bins for energy lags. Same as K in equations in Section 2 of
        Uttley et al. 2014.

    Returns
    -------
    coherence : np.array of floats
        The coherence of the cross spectrum. (Uttley et al 2014, eqn 11)
        Size = n_bins/2+1 (if avg over energy) or detchans (if avg over freq).

    """

    cs_bias = bias_term(power_ci, power_ref, mean_rate_ci, mean_rate_ref,
            meta_dict, n_range)

    temp_1 = power_ci * power_ref
    temp_2 = cross_spec * np.conj(cross_spec) - cs_bias ** 2
    with np.errstate(all='ignore'):
        coherence = np.where(temp_1 != 0, temp_2 / temp_1, 0)
    # print "Coherence shape:", np.shape(coherence)
    # print coherence
    return np.real(coherence)


################################################################################
def main(in_file, out_file, prefix="--", plot_ext="eps",
        rsp_matrix="./PCU2.rsp", lo_freq=1.0, up_freq=10.0):
    """
    Compute the phase lag and time lag from the average cross spectrum.

    Note that power_ci, power_ref, and cs_avg should be unnormalized and without
    noise subtracted.

    Parameters
    ----------
    in_file : str
        Name of the FITS file containing the cross spectrum, power spectrum of
        the channels of interest, and power spectrum of the reference band.
        Same as input file for get_lags.py. In format '*_cs.fits'.

    out_file : str
        Name of the FITS file to write the covariance spectrum to, in format
        '*_cov.fits'.

    prefix : str
        The identifying prefix of the data (object nickname or data ID). [--]

    plot_ext : str
        File extension for the plots. Do not include the dot. [eps]

    rsp_matrix : str
        Local path name of the detector response matrix. [./PCU2.rsp]

    lo_freq : float
        The lower bound of the frequency range to average the covariance
        spectrum over, inclusive, in Hz. [1.0]

    up_freq : float
        The upper bound of the frequency range to average the covariance
        spectrum over, inclusive, in Hz. [1.0]

    Files created
    -------------
    *_cov.fits :
        The covariance spectrum, in FITS format (more robust for storing
        floating point numbers).

    *_cov.dat :
        The covariance spectrum, in ASCII format (for use in ASCII2PHA).

    *_cov.pha :
        The covariance spectrum, in .pha spectrum format, for use in XSPEC.

    temp_cov.dat, temp_cov.pha :
        Temporary local files created because ASCII2PHA doesn't like how long
        the filenames are.

    *_cov_xspec.xcm :
        XSPEC script that plots the covariance spectrum unfolded through the
        detector response matrix and a power law of slope 1.

    dump.txt
        Temporary local file that the XSPEC output is piped (dumped) into.

    *_cov.eps :
        Plot of said covariance spectrum created by the XSPEC script.

    """

    ## Get necessary information and data from the input file
    freq, cs_avg, power_ci, power_ref, meta_dict, mean_rate_ci, mean_rate_ref, \
            evt_list = get_lags.get_inputs(in_file)

    ## Make frequency mask so that we're only averaging over the desired
    ## frequency range
    freq_mask = (freq >= lo_freq) & (freq <= up_freq)
    freq = freq[freq_mask]
    n_freq_bins = len(freq)
    delta_freq = up_freq - lo_freq

    ## Apply frequency mask to cross spectrum and power spectra, and average
    ## over the kept frequencies.
    cs_avg = np.mean(cs_avg[freq_mask, :], axis=0)
    power_ci = np.mean(power_ci[freq_mask, :], axis=0)
    power_ref = np.repeat(np.mean(power_ref[freq_mask], axis=0),
            meta_dict['detchans'])

    ## Apply absolute rms^2 normalization to the cross spectrum and power
    ## spectra
    cs_norm = cs_avg * (2.0 * meta_dict['dt'] / float(n_freq_bins))
    norm_ci = power_ci * (2.0 * meta_dict['dt'] / float(n_freq_bins))
    norm_ref = power_ref * (2.0 * meta_dict['dt'] / float(n_freq_bins))

    ## Compute the raw coherence
    coherence = compute_coherence(cs_norm, norm_ci, norm_ref,
            mean_rate_ci, mean_rate_ref, meta_dict, n_freq_bins)

    ## Compute the covariance. Equation from Uttley et al 2014, footnote 8 on
    ## page 18
    ## It's possible that at the very high energy channels (that you won't want
    ## to use anyway) to over-subtract the noise level from the absolute-
    ## normalized power spectrum. We just tell it to assign a value of zero
    ## here.
    Pnoise_ci = 2.0 * mean_rate_ci
    Pnoise_ref = 2.0 * mean_rate_ref

    ## There should possibly be a *delta_freq in the next expression?
    under_the_sqrt = coherence * (norm_ci - (Pnoise_ci)) * \
                     meta_dict['df']
    with np.errstate(all='ignore'):
        covariance_spectrum = np.where(under_the_sqrt > 0,
                np.sqrt(under_the_sqrt), 0)

    ## 'Integrated' power in the reference band, absolute rms^2 normalization,
    ## not Poisson-noise-subtracted
    ref_var = norm_ref * delta_freq

    ## 'Integrated' Poisson noise levels for absolute rms normalization
    ## Equation in text just below eqn 14 in Uttley et al 2014, p 18
    integ_noise_rms_ci = Pnoise_ci * meta_dict['df']
    integ_noise_rms_ref = Pnoise_ref * meta_dict['df']
    temp1 = covariance_spectrum ** 2 * integ_noise_rms_ref
    temp2 = ref_var * integ_noise_rms_ci
    temp3 = integ_noise_rms_ci * integ_noise_rms_ref

    with np.errstate(all='ignore'):
        covariance_err = np.sqrt((temp1 + temp2 + temp3) /
                (2.0 * meta_dict['n_seg'] * n_freq_bins *
                (ref_var - integ_noise_rms_ref)))

    ##########
    ## Output
    ##########

    fits_out(out_file, in_file, evt_list, meta_dict, lo_freq, up_freq,
            mean_rate_ci, mean_rate_ref, covariance_spectrum, covariance_err,
            n_freq_bins)

    ###################
    ## Plot (in XSPEC)
    ###################

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

    parser.add_argument('--prefix', dest="prefix", default="--",
            help="The identifying prefix of the data (object nickname or "\
            "data ID). [--]")

    parser.add_argument('--ext', dest='plot_ext', default='eps',
            help="File extension for the plots. Do not include the dot. [eps]")

    parser.add_argument('--rsp', dest='rsp_matrix', default='./PCU2.rsp',
            help="Local path name of the detector response matrix. "\
            "[./PCU2.rsp]")

    parser.add_argument('--lf', dest='lo_freq', default=1.0,
            type=tools.type_positive_float, help="The lower bound of the "\
            "frequency range to average the covariance spectrum over, "\
            "inclusive, in Hz. [1.0]")

    parser.add_argument('--uf', dest='up_freq', default=10.0,
            type=tools.type_positive_float, help="The upper bound of the "\
            "frequency range to average the covariance spectrum over, "\
            "inclusive, in Hz. [10.0]")

    args = parser.parse_args()

    main(args.infile, args.outfile, prefix=args.prefix, plot_ext=args.plot_ext,
            rsp_matrix=args.rsp_matrix, lo_freq=args.lo_freq,
            up_freq=args.up_freq)
