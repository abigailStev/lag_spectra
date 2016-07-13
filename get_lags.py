#!/usr/bin/env
"""
Computes the phase lag and time lag of energy channels of interest with a
reference energy band from the average cross spectrum. Can average over
frequency and over energy.

Reads from a FITS file where frequency, raw cross spectrum, raw power spectra of
channels of interest, and raw power spectrum of reference band are saved in an
astropy table in FITS extension 1. Header info is also in extension 1.

Files created
-------------
*_lags.fits :
    Output file with header in FITS extension 0, lag-frequency in extension 1,
    lag-energy in extension 2.

*_lag-freq.(plot_ext) :
    Plot of the lags vs frequency.

*_lag-energy.(plot_ext) :
    Plot of the lag-energy spectrum.


Example call:
python get_lags.py ./cygx1_cs.fits ./cygx1_lags.fits ./cygx1_energies.txt

Enter   python get_lags.py -h   at the command line for help.

"""
import argparse
import subprocess
import numpy as np
from astropy.io import fits
from datetime import datetime
import os.path
from astropy.table import Table, Column
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import ScalarFormatter

import tools  ## in https://github.com/abigailStev/whizzy_scripts

__author__ = "Abigail Stevens <A.L.Stevens at uva.nl>"
__year__ = "2015-2016"


################################################################################
def freq_rebin_values(freq, values, rebin_const=1.02):
    """
    Re-bin a set of values in frequency space by some re-binning
    constant (rebin_const > 1).

    Parameters
    ----------
    freq : np.array of floats
        1-D array of the Fourier frequencies.

    values : np.array of floats
        1-D array of the lag at each Fourier frequency, with any/arbitrary
        normalization.

    rebin_const : float, optional
        The constant by which the data were geometrically re-binned.
        Default = 1.01

    Returns
    -------
    rb_freq : np.array of floats
        1-D array of the re-binned Fourier frequencies.

    rb_values : np.array of floats
        1-D array of the values at the re-binned Fourier frequencies, with the
        same normalization as the input values array.

    bin_array : np.array of ints
        1-D array of how many (old) frequency bins are in each new bin. Same
        size as rb_freq and rb_values.

    """

    ## Initializing variables
    rb_values = np.asarray([]) # List of re-binned values
    rb_freq = np.asarray([])   # List of re-binned frequencies
    bin_array = np.asarray([])
    real_index = 1.0		   # The unrounded next index in values
    current_m = 1			   # Current index in values
    prev_m = 0				   # Previous index m

    ## Looping through the length of the array values, new bin by new bin, to
    ## compute the average values and frequency of that new geometric bin.
    ## Equations for frequency, values, and error are from A. Ingram's PhD thesis
    while current_m < len(values):

        ## Determining the range of indices this specific geometric bin covers
        bin_range = np.absolute(current_m - prev_m)
        ## Want mean values of data points contained within one geometric bin
        bin_values = np.mean(values[prev_m:current_m])

        ## Computing the mean frequency of a geometric bin
        bin_freq = np.mean(freq[prev_m:current_m])

        ## Appending values to arrays
        rb_values = np.append(rb_values, bin_values)
        rb_freq = np.append(rb_freq, bin_freq)
        bin_array = np.append(bin_array, int(current_m - prev_m))

        ## Incrementing for the next iteration of the loop
        ## Since the for-loop goes from prev_m to current_m-1 (since that's how
        ## the range function and array slicing works) it's ok that we set
        ## prev_m = current_m here for the next round. This will not cause any
        ## double-counting bins or skipping bins.
        prev_m = current_m
        real_index *= rebin_const
        int_index = int(round(real_index))
        current_m += int_index
        bin_range = None
        bin_freq = None
        bin_values = None

    return rb_freq, rb_values, bin_array


################################################################################
def get_inputs(in_file):
    """
    Gets cross spectrum, interest bands power spectra, reference band power
    spectrum, and necessary constants from the input FITS file. Data is
    structured as an astropy table in FITS extension 1 with with "FREQUENCY",
    "CROSS", "POWER_REF", and "POWER_CI". The cross-spectrum and power spectra
    should be raw, i.e. un-normalized and not noise-subtracted.

    2-D arrays of cross spectrum and interest bands were flattened (and are
    reshaped here)'C-style'.

    Parameters
    ----------
    in_file : str
        The file "*_cs.fits" from cross_correlation/ccf.py. Data is structured
        as an astropy table in FITS extension 1 with "FREQUENCY", "CROSS",
        "POWER_REF", and "POWER_CI", and the header information is the Table
        metadata.

    Returns
    -------

    in_table : astropy Table

    freq : np.array of floats
        1-D array of the Fourier frequencies of the cross spectrum, in Hz.

    cs_avg : np.array of complex numbers
        2-D array of the cross spectrum. Should be un-normalized and not noise-
        subtracted. Size = (n_bins, detchans).

    power_ci : np.array of floats
        2-D array of the power in the channels of interest. Should be un-
        normalized and not noise-subtracted. Size = (n_bins, detchans).

    power_ref : np.array of floats
        1-D array of the power in the reference band. Should be un-normalized
        and not noise-subtracted. Size = (n_bins).

    meta_dict : dict
        Dictionary of meta-parameters needed for analysis.

    rate_ci : np.array of floats
        1-D array of the background-subtracted mean count rate in each energy
        channel, in cts/s.

    rate_ref : float
        Mean count rate in the reference band, in cts/s.

    evt_list : str
        The name of the event list(s) where the original data came from, to
        write to the FITS header of the output file.

    """

    try:
        in_table = Table.read(in_file, format='fits')
    except IOError:
        print("\tERROR: File does not exist: %s" % in_file)
        exit()
    rate_ci = np.asarray(in_table.meta['RATE_CI'].replace('[',\
            '').replace(']','').split(','), dtype=np.float64)
    in_table.meta['RATE_CI'] = rate_ci
    return in_table


################################################################################
def fits_out(out_file, freq_lags, energy_lags):
    """
    Write the lag-frequency and lag-energy spectra to a FITS output file.
    Header info is in extension 0, lag-frequency is in extension 1, and
    lag-energy is in extension 2.

    Parameters
    ----------
    out_file : str
        The full path of the output file, in format '*_lag.fits'.

    in_file : str
        The full path of the cross-spectrum input file.

    evt_list : str
        The full path of the event list of the data.

    meta_dict : dict
        Dictionary of meta-parameters needed for analysis.

    lo_freq, up_freq : float
        The lower and upper frequency bounds to average over for computing the
        lag-energy spectrum, inclusive, in Hz.

    lo_chan, up_chan : int
        The lower and upper energy bounds to average over for computing the
        lag-frequency spectrum, inclusive, in detector energy channel.

    mean_rate_ci : np.array of floats
        1-D array of the mean photon count rate of each of the cross-spectral
        channels of interest, in cts/s.

    mean_rate_ref : float
        The mean photon count rate of the cross-spectral reference band, in
        cts/s.

    freq : np.array of floats
        1-D array of the Fourier frequencies against which the lag-frequency
        spectrum is plotted.

    phase, err_phase : np.array of floats
        The phase and error in phase of the frequency lags, in radians.

    tlag, err_tlag : np.array of floats
        The time and error in time of the frequency lags, in seconds.

    e_phase, e_err_phase : np.array of floats
        The phase and error in phase of the energy lags, in radians.

    e_tlag, e_err_tlag : np.array of floats
        The time and error in time of the energy lags, in seconds.

    Returns
    -------
    Nothing, but writes to file '*_lag.fits'.

    """

    print "Output sent to: %s" % out_file
    freq_lags.meta['RATE_CI'] = str(freq_lags.meta['RATE_CI'].tolist())
    energy_lags.meta['RATE_CI'] = str(energy_lags.meta['RATE_CI'].tolist())

    # ## Make FITS header (extension 0)
    # prihdr = fits.Header()
    # prihdr.set('TYPE', "Lag-frequency and lag-energy spectra")
    # prihdr.set('DATE', str(datetime.now()), "YYYY-MM-DD localtime")
    # prihdr.set('EVTLIST', freq_lags.meta['EVTLIST'])
    # prihdr.set('CS_DATA', freq_lags.meta['CS_DATA'])
    # prihdr.set('DT', freq_lags.meta['DT'], "seconds")
    # prihdr.set('N_BINS', freq_lags.meta['N_BINS'], "time bins per segment")
    # prihdr.set('SEGMENTS', freq_lags.meta['SEGMENTS'],
    #            "segments in the whole light curve")
    # prihdr.set('EXPOSURE', freq_lags.meta['EXPOSURE'],
    #            "seconds, of light curve")
    # prihdr.set('DETCHANS', freq_lags.meta['DETCHANS'],
    #            "Number of detector energy channels")
    # prihdr.set('LAG_LF', freq_lags.meta['LO_FREQ'],
    #            "Hz; Lower frequency bound for energy lags")
    # prihdr.set('LAG_UF', freq_lags.meta['UP_FREQ'],
    #            "Hz; Upper frequency bound for energy lags")
    # prihdr.set('LAG_LE', freq_lags.meta['LO_CHAN'],
    #            "Lower energy channel bound for frequency lags")
    # prihdr.set('LAG_UE', freq_lags.meta['UP_CHAN'],
    #            "Upper energy channel bound for frequency lags")
    # prihdr.set('RATE_CI', freq_lags.meta['RATE_CI'], "cts/s")
    # prihdr.set('RATE_REF', freq_lags.meta['RATE_REF'], "cts/s")
    # prihdu = fits.PrimaryHDU(prihdr)

    # ## Make FITS table for lag-frequency plot (extension 1)
    # col1 = fits.Column(name='FREQUENCY', format='D', array=f_bins)
    # col2 = fits.Column(name='PHASE', unit='radians', format='D',
    #                    array=phase.flatten('C'))
    # col3 = fits.Column(name='PHASE_ERR', unit='radians', format='D',
    #                    array=err_phase.flatten('C'))
    # col4 = fits.Column(name='TIME_LAG', unit='s', format='D',
    #                    array=tlag.flatten('C'))
    # col5 = fits.Column(name='TIME_LAG_ERR', unit='s', format='D',
    #                    array=err_tlag.flatten('C'))
    # cols = fits.ColDefs([col1, col2, col3, col4, col5])
    # tbhdu1 = fits.BinTableHDU.from_columns(cols)
    #
    # ## Make FITS table for lag-energy plot (extension 2)
    # col1 = fits.Column(name='PHASE', unit='radians', format='D', array=e_phase)
    # col2 = fits.Column(name='PHASE_ERR', unit='radians', format='D', \
    #                    array=e_err_phase)
    # col3 = fits.Column(name='TIME_LAG', unit='s', format='D', array=e_tlag)
    # col4 = fits.Column(name='TIME_LAG_ERR', unit='s', format='D', \
    #                    array=e_err_tlag)
    # col5 = fits.Column(name='CHANNEL', unit='', format='I', \
    #                    array=chan)
    # cols = fits.ColDefs([col1, col2, col3, col4, col5])
    # tbhdu2 = fits.BinTableHDU.from_columns(cols)
    #
    # ## Check that the filename has FITS file extension
    # assert out_file[-4:].lower() == "fits", \
    #     'ERROR: Output file must have extension ".fits".'
    #
    # ## Write to a FITS file
    # prihdr = fits.Header(freq_lags.meta)
    # thdulist = fits.HDUList([prihdu,
    #                          fits.table_to_hdu(freq_lags)]) #,
    #                          # fits.table_to_hdu(energy_lags)])
    # thdulist.writeto(out_file, clobber=True)


    ## I know this his hack-y, but it's good enough for now. I kept getting a
    ## string64 or string32 error when trying to make a primary HDU from the
    ## Table meta information.
    ##      self._bitpix = DTYPE2BITPIX[data.dtype.name]
    ##      KeyError: 'string32'

    freq_lags.write(out_file, overwrite=True, format='fits')
    hdulist = fits.open(out_file, mode='update')
    hdulist.append(fits.table_to_hdu(energy_lags))
    hdulist.flush()


################################################################################
def plot_lag_freq(out_root, plot_ext, prefix, freq_lags):
    """
    Plots the lag-frequency spectrum.

    Parameters
    ----------
    out_root : str
        Dir+base name for plots generated, to be appended with '_lag-freq.(plot
        _ext)'.

    plot_ext : str
        File extension for the plots. Do not include the dot.

    prefix : str
        Identifying prefix of the data (object nickname or data ID).

    freq : np.array of floats
        1-D array of the Fourier frequency of the cross-spectrum, in Hz.
        Size = (f_range), range of frequencies to plot.

    phase, err_phase : np.arrays of floats
        1-D arrays of the phase and error on phase of the frequency lags, in
        radians. Shape = (f_range), range of frequencies to plot.

    tlag, err_tlag : np.arrays of floats
        1-D arrays of the time lag and error on time lag for frequency lags, in
        seconds. Size = (f_range), range of frequencies to plot.

    lo_freq, up_freq : floats
        Lower and upper bound of frequency range for averaging, inclusive, in
        Hz.

    lo_chan, up_chan : ints
        Lower and upper bound of energy channel range for averaging, inclusive,
        in detector energy channels. Written to plot title.

    Returns
    -------
    Nothing, but saves a plot to '*_lag-freq.[plot_ext]'.
    """

    font_prop = font_manager.FontProperties(size=20)
    # xLocator = MultipleLocator(0.2)  ## loc of minor ticks on x-axis

    plot_file = out_root + "_lag-freq." + plot_ext
    print "Lag-frequency spectrum: %s" % plot_file

    fig, ax = plt.subplots(1, 1, figsize=(10,7.5), dpi=300, tight_layout=True)
    ax.plot([freq_lags['FREQUENCY'][0], freq_lags['FREQUENCY'][-1]], [0, 0],
            lw=1.5, ls='dashed', c='black')

    # ax.plot([freq_lags['FREQUENCY'][0], freq_lags['FREQUENCY'][-1]],
    #         [np.pi,np.pi], lw=1.5, ls='dashed', c='black')
    # ax.plot([freq_lags['FREQUENCY'][0], freq_lags['FREQUENCY'][-1]],
    #         [-np.pi,-np.pi], lw=1.5, ls='dashed', c='black')
    # ax.errorbar(freq_lags['FREQUENCY'], freq_lags['PHASE_LAG'],
    #             yerr=freq_lags['PHASE_ERR'], lw=3, c='blue', ls='steps-mid',
    #             elinewidth=2, capsize=2)

    ax.errorbar(freq_lags['FREQUENCY'], freq_lags['TIME_LAG'],
                yerr=freq_lags['TIME_ERR'], lw=2, c='blue', ls='steps-mid',
                capsize=2, elinewidth=2)

    ax.set_xlabel('Frequency (Hz)', fontproperties=font_prop)
    ax.set_ylabel('Time lag (s)', fontproperties=font_prop)
    ax.set_xscale('log')
    # ax.set_yscale('log')
    # 	ax.set_ylabel('Phase lag (radians)', fontproperties=font_prop)
    ax.set_xlim(freq_lags.meta['LO_FREQ'], freq_lags.meta['UP_FREQ'])
    ax.set_ylim(-0.04, 0.04)
    # ax.set_ylim(1.3*np.min(freq_lags['TIME_LAG']), 1.30*np.max(freq_lags['TIME_LAG']))
    # 	print np.min(freq_lags['TIME_LAG'])
    # 	print np.max(freq_lags['TIME_LAG'])
    # 	ax.set_ylim(-0.3, 0.3)
    # 	ax.set_ylim(-6, 6)
    # ax.xaxis.set_minor_locator(xLocator)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(which='major', width=1.5, length=7)
    ax.tick_params(which='minor', width=1.5, length=4)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
    title = "Lag-frequency spectrum, %s, channels %d - %d" % (prefix,
                                                              freq_lags.meta[
                                                                  'LO_CHAN'],
                                                              freq_lags.meta[
                                                                  'UP_CHAN'])
    ax.set_title(title, fontproperties=font_prop)

    plt.savefig(plot_file)
    # plt.show()
    plt.close()
    # subprocess.call(['open', plot_file])


################################################################################
def plot_lag_energy(out_root, energies_tab, plot_ext, prefix, energy_lags,
                    detchans=64):
    """
    Plots the lag-energy spectrum.

    Parameters
    ----------
    out_root : str
        Dir+base name for plots generated, to be appended with '_lag-energy.
        (plot_ext)'.

    plot_ext : str
        File extension for the plots. Do not include the dot.

    prefix : str
        Identifying prefix of the data (object nickname or data ID).

    energy_lags : astropy Table

    detchans : int


    Returns
    -------
    Nothing, but saves a plot to '*_lag-energy.[plot_ext]'.

    """
    font_prop = font_manager.FontProperties(size=20)

    energy_list = []
    energy_err = []
    if energies_tab is not None:
        energy_list = [np.mean([x, y]) for x,y in tools.pairwise(energies_tab)]
        energy_err = [np.abs(a-b) for (a,b) in zip(energy_list,
                                                   energies_tab[0:-1])]

    plot_file = out_root + "_lag-energy." + plot_ext
    print "Lag-energy spectrum: %s" % plot_file

    ## Deleting the values at energy channel 10 for RXTE PCA event-mode data
    ## No counts detected in channel 10 in this data mode
    if detchans == 64 and len(energy_lags['PHASE_LAG']) == 64:
        energy_lags.remove_row(10)
        energy_list = np.delete(energy_list, 10)
        energy_err = np.delete(energy_err, 10)

    #############
    ## Plotting!
    #############

    fig, ax = plt.subplots(1, 1, figsize=(10,7.5), dpi=300, tight_layout=True)
    # ax.plot([0,detchans],[0,0], lw=1.5, ls='dashed', c='black')
    # 	ax.plot([0,detchans],[np.pi,np.pi], lw=1.5, ls='dashed', c='black')
    # 	ax.plot([0,detchans],[-np.pi,-np.pi], lw=1.5, ls='dashed', c='black')
    # 	ax.errorbar(e_chans, phase, yerr=err_phase, lw=3, c='red', \
    # 		ls="steps-mid", elinewidth=2, capsize=2)
    # ax.errorbar(e_chans, tlag, yerr=err_tlag, ls='none', marker='x', \
    #             ms=10, mew=2, mec='red', ecolor='red', \
    #             elinewidth=2, capsize=2)
    # ax.set_xlabel('Energy channel (0-%d)' % (detchans - 1), \
    #               fontproperties=font_prop)
    # ax.set_xlim(1.5, 25.5)


    # print energy_list
    if len(energy_list) > 0:
        ax.hlines(0.0, 3, 20.5, linestyle='dashed', lw=2, color='black')

        ax.errorbar(energy_list[2:26], energy_lags['TIME_LAG'][2:26],
                    xerr=energy_err[2:26], yerr=energy_lags['TIME_ERR'][2:26],
                    ls='none', marker='o', ms=10, mew=2, mec='black',
                    mfc='black', ecolor='black', elinewidth=2, capsize=0)
        ax.set_xlim(3,21)
        ax.set_xlabel('Energy (keV)', fontproperties=font_prop)
        ax.set_xscale('log')
        x_maj_loc = [5,10,20]
        ax.set_xticks(x_maj_loc)
        ax.xaxis.set_major_formatter(ScalarFormatter())

    else:
        ax.hlines(0.0, energy_lags['CHANNEL'][0], energy_lags['CHANNEL'][-1],
                  linestyle='dashed', lw=2, color='black')

        ax.errorbar(energy_lags['CHANNEL'], energy_lags['TIME_LAG'],
                    yerr=energy_lags['TIME_ERR'], ls='none', marker='o', ms=10,
                    mew=2, mec='black', mfc='black', ecolor='black',
                    elinewidth=2, capsize=0)
        ax.set_xlabel('Energy channel', fontproperties=font_prop)

    # ax.errorbar(energy_list[5:200], tlag[5:200], xerr=energy_err[5:200],
    #       yerr=err_tlag[5:200], ls='none',
    #         marker='o', ms=5, mew=2, mec='black', mfc='black', ecolor='black',
    #         elinewidth=2, capsize=0)
    # ax.errorbar(energy_list, phase, xerr=energy_err, yerr=err_phase, ls='none',
    #         marker='+', ms=8, mew=2, mec='black', ecolor='black', elinewidth=2,
    #         capsize=0)

    ax.set_ylim(-0.01, 0.017)
    # ax.set_ylim(1.3 * np.min(tlag[2:25]), 1.30 * np.max(tlag[2:25]))

    # y_maj_loc = [-0.005, 0, 0.005, 0.01, 0.015]
    # ax.set_yticks(y_maj_loc)
    xLocator = MultipleLocator(1)  ## loc of minor ticks on x-axis
    yLocator = MultipleLocator(0.001)  ## loc of minor ticks on y-axis
    ax.xaxis.set_minor_locator(xLocator)
    ax.yaxis.set_minor_locator(yLocator)
    ax.tick_params(which='major', width=1.5, length=7)
    ax.tick_params(which='minor', width=1.5, length=4)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)

    ax.set_ylabel('Time lag (s)', fontproperties=font_prop)
    # ax.set_ylabel('Phase lag (radians)', fontproperties=font_prop)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    title = "Lag-energy spectrum, %s, %.2f - %.2f Hz" % (prefix,
                                                         energy_lags.meta[
                                                             'LO_FREQ'],
                                                         energy_lags.meta[
                                                             'UP_FREQ'])
    # ax.set_title(title, fontproperties=font_prop)

    plt.savefig(plot_file)
    # 	plt.show()
    plt.close()
    # subprocess.call(['open', plot_file])


################################################################################
def bias_term(in_table, n_range):
    """
    Compute the bias term to be subtracted off the cross spectrum to compute
    the covariance spectrum. Equation in Equation in footnote 4 (section 2.1.3,
    page 12) of Uttley et al. 2014.

    Assumes power spectra are raw (not at all normalized, and not Poisson-noise-
    subtracted).

    Parameters
    ----------
    power_ci : np.array of floats
        2-D array of the power in the channels of interest, raw (not normalized
        and not Poisson-noise-subtracted), of the frequencies to be averaged
        over. Size = detchans (if avg over freq) or n_bins/2+1 (if avg over energy).

    power_ref : np.array of floats
        1-D array of the power in the reference band, raw (not normalized and
        not Poisson-noise-subtracted), of the frequencies to be averaged over.
        Size = 1 (if avg over freq) or n_bins/2+1 (if avg over energy).

    mean_rate_ci : np.array of floats
        1-D array of the mean count rate in the channels of interest, in cts/s.
        Size = 1 (if avg over energy) or detchans (if avg over freq).

    mean_rate_ref : float
        Mean count rate in the reference band, in cts/s.

    meta_dict : dict
        Dictionary of meta-parameters needed for analysis.

    n_range : int
        Number of bins that will be averaged together for the lags. Energy bins
        for frequency lags, frequency bins for energy lags.

    Returns
    -------
    n_squared : float
        The bias term to be subtracted off the cross spectrum for computing the
        covariance spectrum. Equation in footnote 4 (section 2.1.3, page 12) of
        Uttley et al. 2014.

    """
    ## Compute the Poisson noise level in absolute rms units
    Pnoise_ref = in_table.meta['RATE_REF'] * 2.0
    Pnoise_ci = in_table.meta['RATE_CI'] * 2.0

    ## Normalizing power spectra to absolute rms normalization
    ## Not subtracting the noise (yet)!
    abs_ci = in_table['POWER_CI'] * (2.0 * in_table.meta['DT'] / n_range)
    abs_ref = in_table['POWER_REF'] * (2.0 * in_table.meta['DT'] / n_range)

    temp_a = (abs_ref - Pnoise_ref) * Pnoise_ci
    temp_b = (abs_ci - Pnoise_ci) * Pnoise_ref
    temp_c = Pnoise_ref * Pnoise_ci

    n_squared = (temp_a + temp_b + temp_c) / (n_range * in_table.meta['SEGMENTS'])
    return n_squared


################################################################################
def compute_coherence(in_table, n_range=1):
    """
    Compute the raw coherence of the cross spectrum. Coherence equation from
    Uttley et al 2014 eqn 11, bias term equation from footnote 4 on same page.

    Parameters
    ----------
    in_table : astropy Table

    n_range : int
        Number of frequency bins averaged over per new frequency bin for lags.
        For energy lags, this is the number of frequency bins averaged over. For
        frequency lags not re-binned in frequency, this is 1. For frequency lags
        that have been re-binned, this is a 1-D array with ints of the number of
        old bins in each new bin. Same as K in equations in Section 2 of
        Uttley et al. 2014. Default=1

    Returns
    -------
    coherence : np.array of floats
        The raw coherence of the cross spectrum. (Uttley et al 2014, eqn 11)
        Size = n_bins/2+1 (if avg over energy) or detchans (if avg over freq).

    """

    ## Reshaping (broadcasting) the ref to have same size as ci
    # if np.shape(power_ref) != np.shape(power_ci):
    #     power_ref = np.resize(np.repeat(power_ref, np.shape(power_ci)[1]),
    #             np.shape(power_ci))

    cs_bias = bias_term(in_table, n_range)

    powers = in_table['POWER_CI'] * in_table['POWER_REF']
    temp_2 = in_table['CROSS'] * np.conj(in_table['CROSS']) - cs_bias
    with np.errstate(all='ignore'):
        coherence = np.where(powers != 0, temp_2 / powers, 0)
    # print "Coherence shape:", np.shape(coherence)
    # print coherence
    return np.real(coherence)


################################################################################
def get_phase_err(in_table, n_range=1):
    """
    Compute the error on the complex phase (in radians) via the coherence.
    Power should NOT be Poisson-noise-subtracted or normalized.

    Parameters
    ----------
    in_table : Astropy table


    n_range : int or np.array of ints
        Number of frequency bins averaged over per new frequency bin for lags.
        For energy lags, this is the number of frequency bins averaged over. For
        frequency lags not re-binned in frequency, this is 1. Same as K in
        equations in Section 2 of Uttley et al. 2014.

    Returns
    -------
    phase_err : np.array of floats
        1-D array of the error on the phase of the lag.

    """

    # coherence = np.where(a != 0, cs_avg * np.conj(cs_avg) / a, 0)
    coherence = compute_coherence(in_table, n_range)

    with np.errstate(all='ignore'):
        phase_err = np.sqrt(np.where(coherence != 0, (1 - coherence) /
                (2 * coherence * n_range * in_table.meta['SEGMENTS']), 0))

    return phase_err


################################################################################
def phase_to_tlags(phase, f):
    """
    Convert a complex-plane cross-spectrum phase (in radians) to a time lag
    (in seconds).

    Parameters
    ----------
    phase : float or np.array of floats
        1-D array of the phase of the lag, in radians.

    f : float or np.array of floats
        1-D array of the Fourier frequency of the cross-spectrum, in Hz.

    Returns
    -------
    tlags : float or np.array of floats
        1-D array of the time of the lag, in seconds.

    """

    if np.shape(phase) != np.shape(f):
        ## Reshaping (broadcasting) f to have same size as phase
        f = np.resize(np.repeat(f, np.shape(phase)[1]), np.shape(phase))

    assert np.shape(phase) == np.shape(f), "ERROR: Phase array must have same "\
            "dimensions as frequency array."

    with np.errstate(all='ignore'):
        tlags = np.where(f != 0, phase / (2.0 * np.pi * f), 0)

    return tlags


################################################################################
def calc_freq_lags(in_table):
    """
    Calculating the frequency lags over an energy range.
    See steps in Uttley et al 2014 section 2.2.1.

    Parameters
    ----------
    cs_avg : np.array of complex numbers
        2-D array of the cross-spectrum averaged together over light curve
        segments. Only the positive Fourier frequencies.
        Size = (n_bins/2, detchans).

    :param power_ci:
    :param power_ref:
    :param meta_dict:
    :param lo_chan:
    :param up_chan:
    :return:
    """
    print np.shape(in_table['CROSS'])
    ###########################
    ## Averaging over energies
    ###########################

    erange_cs = np.mean(in_table['CROSS'][:,
                        in_table.meta['LO_CHAN']:in_table.meta['UP_CHAN']+1],
                        axis=1)
    erange_pow_ci = np.mean(in_table['POWER_CI'][:,
                            in_table.meta['LO_CHAN']:in_table.meta['UP_CHAN']
                                                     +1], axis=1)
    erange_pow_ref = in_table['POWER_REF']

    ## Re-bin the PSDs and cross-spectrum in frequency
    ## Uttley et al 2014 section 2.2.1 step 4
    rb_freq_cs, rb_cs, bin_cs = freq_rebin_values(in_table['FREQUENCY'],
                                                  erange_cs)
    rb_freq_powci, rb_pow_ci, bin_powci = \
            freq_rebin_values(in_table['FREQUENCY'], erange_pow_ci)
    rb_freq_powref, rb_pow_ref, bin_powref = \
            freq_rebin_values(in_table['FREQUENCY'], erange_pow_ref)

    assert rb_freq_cs.all() == rb_freq_powci.all(), "Something went wrong in re-binning " \
                                        "for frequency lags."
    # assert rb_freq_cs == rb_freq_powref, "Something went wrong in re-binning " \
    #                                      "for frequency lags."
    # assert bin_cs == bin_powci, "Something went wrong in re-binning for " \
    #                             "frequency lags."
    # assert bin_cs == bin_powref, "Something went wrong in re-binning for " \
    #                              "frequency lags."

    rb_table = Table()
    rb_table.meta = in_table.meta
    rb_table['FREQUENCY'] = Column(rb_freq_cs)
    rb_table['CROSS'] = Column(rb_cs)
    rb_table['POWER_CI'] = Column(rb_pow_ci)
    rb_table['POWER_REF'] = Column(rb_pow_ref)

    rb_table.meta['RATE_CI'] = np.mean(rb_table.meta[ \
                                           'RATE_CI'][rb_table.meta[ \
                                           'LO_CHAN']:rb_table.meta[
                                                          'UP_CHAN'] + 1])
    ############################################
    ## Getting lag and error for frequency lags
    ############################################

    ## Negative sign is so that a positive lag is a hard energy lag
    # f_tlag = phase_to_tlags(f_phase, rb_freq)
    # f_err_tlag = phase_to_tlags(f_err_phase, rb_freq)


    freq_lags = Table()
    freq_lags.meta = in_table.meta
    freq_lags['FREQUENCY'] = Column(rb_table['FREQUENCY'], unit='Hz',
                                    description='Fourier frequency')
    freq_lags['PHASE_LAG'] = Column(-np.arctan2(rb_table['CROSS'].imag,
                                                rb_table['CROSS'].real),
                                    unit='rad', description='Phase lag')
    freq_lags['PHASE_ERR'] = Column(get_phase_err(rb_table, bin_cs),
                                    unit='rad',description='Error on phase lag')
    freq_lags['TIME_LAG'] = Column(phase_to_tlags(freq_lags['PHASE_LAG'],
                                                  rb_table['FREQUENCY']),
                                   unit='s', description='Time lag')
    freq_lags['TIME_ERR'] = Column(phase_to_tlags(freq_lags['PHASE_ERR'],
                                                  rb_table['FREQUENCY']),
                                   unit='s', description='Error on time lag')

    freq_lags.meta['LO_FREQ'] = in_table.meta['LO_FREQ']
    freq_lags.meta['UP_FREQ'] = in_table.meta['UP_FREQ']
    freq_lags.meta['LO_CHAN'] = in_table.meta['LO_CHAN']
    freq_lags.meta['UP_CHAN'] = in_table.meta['UP_CHAN']

    return freq_lags


################################################################################
def calc_energy_lags(in_table):
    """
    Calculating the energy lags over a frequency range.
    See steps in Uttley et al 2014 section 2.2.2.

    :param cs_avg:
    :param power_ci:
    :param power_ref:
    :param meta_dict:
    :param lo_freq:
    :param up_freq:
    :return:
    """

    ##############################
    ## Averaging over frequencies
    ##############################

    if in_table.meta['LO_FREQ'] in in_table['FREQUENCY']:
        f_span_low = np.argmax(in_table['FREQUENCY'] == in_table.meta['LO_FREQ'])
    else:
        f_span_low = np.argmax(
            np.where(in_table['FREQUENCY'] <= in_table.meta['LO_FREQ']))  ## The last True value
    if in_table.meta['UP_FREQ'] in in_table['FREQUENCY']:
        f_span_hi = np.argmax(in_table['FREQUENCY'] == in_table.meta['UP_FREQ'])
    else:
        f_span_hi = np.argmax(np.where(in_table['FREQUENCY'] < in_table.meta['UP_FREQ'])) + 1  ## The first False value

    f_span = f_span_hi - f_span_low + 1  ## including both ends

    new_table = Table()
    new_table.meta = in_table.meta
    temp_f = np.mean(in_table['FREQUENCY'][f_span_low:f_span_hi + 1], )
    new_table['FREQUENCY'] = Column(np.repeat(temp_f,
            in_table.meta['DETCHANS']), dtype=np.float32)
    new_table['CROSS'] = Column(np.mean(in_table['CROSS'][f_span_low:\
            f_span_hi + 1, ], axis=0), dtype=np.complex128)
    new_table['POWER_CI'] = Column(np.mean(in_table['POWER_CI'][f_span_low:\
            f_span_hi + 1, ], axis=0), dtype=np.float64)
    new_table['POWER_REF'] = Column(np.repeat(np.mean(in_table[
            'POWER_REF'][f_span_low:f_span_hi + 1]),
            in_table.meta['DETCHANS']), dtype=np.float64)

    #########################################
    ## Getting lag and error for energy lags
    #########################################


    energy_lags = Table()
    energy_lags.meta = in_table.meta
    energy_lags['CHANNEL'] = Column(np.arange(in_table.meta['DETCHANS']),
                                    description='Detector energy channel',
                                    dtype=np.int, unit='chan')
    energy_lags['PHASE_LAG'] = Column(-np.arctan2(new_table['CROSS'].imag,
                                                  new_table['CROSS'].real),
                                      unit='rad', description='Phase lag',
                                      dtype=np.float64)
    energy_lags['PHASE_ERR'] = Column(get_phase_err(new_table, f_span),
                                      unit='rad', dtype=np.float64,
                                      description='Error on phase lag')
    energy_lags['TIME_LAG'] = Column(phase_to_tlags(energy_lags['PHASE_LAG'],
                                                    new_table['FREQUENCY']),
                                     unit='s', dtype=np.float64,
                                     description='Time lag')
    energy_lags['TIME_ERR'] = Column(phase_to_tlags(energy_lags['PHASE_ERR'],
                                                    new_table['FREQUENCY']),
                                     unit='s', dtype=np.float64,
                                     description='Error on time lag')

    energy_lags.meta['LO_FREQ'] = in_table.meta['LO_FREQ']
    energy_lags.meta['UP_FREQ'] = in_table.meta['UP_FREQ']
    energy_lags.meta['LO_CHAN'] = in_table.meta['LO_CHAN']
    energy_lags.meta['UP_CHAN'] = in_table.meta['UP_CHAN']
    #
    # e_phase = -np.arctan2(frange_cs.imag, frange_cs.real)
    # e_err_phase = get_phase_err(frange_cs, frange_pow_ci, frange_pow_ref,
    #         mean_rate_ci, mean_rate_ref, meta_dict, f_span)
    # e_tlag = phase_to_tlags(e_phase, f)
    # e_err_tlag = phase_to_tlags(e_err_phase, f)

    return energy_lags


################################################################################
def compute_lags(in_table):
    """
    Computing frequency lags (averaged over specified energies) and energy lags
    (averaged over specified frequencies).

    Parameters
    ----------
    freq : np.array of floats
        1-D array of the Fourier frequencies of the cross spectrum, in Hz.
        Size = (n_bins), or (n_bins/2+1) for only positive Fourier frequencies.

    cs_avg : np.array of complex numbers
        2-D array of the cross spectrum, not normalized.
        Eqn 9 of Uttley et al 2014, with out summing over or dividing by K
        (i.e., just averaged over the segments of data).
        Size = (n_bins, detchans) or (n_bins/2+1, detchans).

    power_ci : np.array of floats
        2-D array of the power spectra per channel of interest.
        Size = (n_bins, detchans) or (n_bins/2+1, detchans).
        Eqn 2 of Uttley et al 2014, averaged over Fourier segments of data, per
        channel of interest.

    power_ref : np.array of floats
        1-D array of the power in the reference band. Size = (n_bins) or
        (n_bins/2+1). Eqn 2 of Uttley et al 2014, averaged over Fourier segments
        of data.

    meta_dict : dict
        Dictionary of meta-parameters needed for analysis.

    mean_rate_ci : np.array of floats
        1-D array of the mean count rate per channel of interest, in cts/s.

    mean_rate_ref : float
        Mean count rate of the reference band, in cts/s.

    lo_freq, up_freq : floats
        Lower and upper bound of frequency range for averaging the energy lags,
        inclusive, in Hz.

    lo_chan, up_chan : floats
        Lower and upper bound of energy channel range for averaging the
        frequency lags, inclusive, in detector energy channels.

    Returns
    -------
    f_phase, f_err_phase, f_tlag, f_err_tlag, e_phase, e_err_phase,
            e_tlag, e_err_tlag
    """

    ## If cross spectrum contains positive and negative frequencies, only keep
    ## the positive ones
    if np.shape(in_table['CROSS']) == (in_table.meta['N_BINS'], in_table.meta['DETCHANS']):
        nyq_ind = np.argmax(in_table['FREQUENCY']) + 1  ## because in python, the scipy fft makes
                ## the nyquist frequency negative, and we want it to be
                ## positive! (it is actually both pos and neg)
        in_table['FREQUENCY'] = np.abs(in_table['FREQUENCY'][0:nyq_ind + 1])  ## because it slices at end-1, and
                ## we want to include 'nyq_ind'; abs is because the nyquist freq
                ## is both pos and neg, and we want it pos here.
        in_table['CROSS'] = in_table['CROSS'][0:nyq_ind + 1, ]
        in_table['POWER_CI'] = in_table['POWER_CI'][0:nyq_ind + 1, ]
        in_table['POWER_REF'] = in_table['POWER_REF'][0:nyq_ind + 1]

    assert np.shape(in_table['POWER_CI']) == (in_table.meta['N_BINS'] / 2 + 1,
            in_table.meta['DETCHANS'])
    assert np.shape(in_table['POWER_REF']) == (in_table.meta['N_BINS'] / 2 + 1,)
    assert np.shape(in_table['CROSS']) == (in_table.meta['N_BINS'] / 2 + 1,
            in_table.meta['DETCHANS'])
    assert np.shape(in_table['FREQUENCY']) == (in_table.meta['N_BINS'] / 2 + 1,)

    ##############################
    ## Calculating frequency lags
    ##############################

    freq_lags = calc_freq_lags(in_table)

    energy_lags = calc_energy_lags(in_table)

    return freq_lags, energy_lags


################################################################################
def main(in_file, out_file, energies_file, plot_root, prefix="--",
        plot_ext="eps", lo_freq=1.0, up_freq=10.0, lo_chan=2, up_chan=26):
    """
    Compute the phase lag and time lag from the average cross spectrum.

    Note that power_ci, power_ref, and cs_avg should be unnormalized and without
    noise subtracted.

    Parameters
    ----------
    in_file : str
        The full path of the cross-spectrum input file.

    out_file : str
        The full path of the output file, in format '*_lag.fits'.

    energies_file : str
        Name of the txt file containing a list of the keV energy boundaries of
        the detector energy channels (so for 4 channels, would have 5 energies
        listed). Created in rxte_reduce/channel_to_energy.py.

    plot_root : str
        Dir+base name for plots generated, to be appended with '_lag-freq.(plot
        _ext)' and '_lag-energy.(plot_ext)'.

    prefix : str
        Identifying prefix of the data (object nickname or data ID). [--]

    plot_ext : str
        File extension for the plots. Do not include the dot. [eps]

    lo_freq : float
        The lower bound of the frequency range to average the energy lags over
        and plot the frequency lags in, inclusive, in Hz.

    up_freq : float
        The upper bound of the frequency range to average the energy lags over
        and plot the frequency lags in, inclusive, in Hz.

    lo_chan : int
        The lower bound of the energy channel range to average the frequency
        lags over and plot the energy lags in, inclusive, in detector energy
        channels.

    up_chan : int
        The upper bound of the energy channel range to average the frequency
        lags over and plot the energy lags in, inclusive, in detector energy
        channels.

    Files created
    -------------
    *_lags.fits :
        Output file with header in FITS extension 0, lag-frequency in extension
        1, lag-energy in extension 2.

    *_lag-freq.(plot_ext) :
        Plot of the lags vs frequency.

    *_lag-energy.(plot_ext) :
        Plot of the lag-energy spectrum.

    """

    energies_tab = np.loadtxt(energies_file)
    # energies_tab = []

    #######################################################
    ## Get analysis constants and data from the input file
    #######################################################

    in_table = get_inputs(in_file)
    in_table.meta['LO_CHAN'] = lo_chan
    in_table.meta['UP_CHAN'] = up_chan
    in_table.meta['LO_FREQ'] = lo_freq
    in_table.meta['UP_FREQ'] = up_freq
    in_table.meta['CS_DATA'] = in_file


    ######################
    ## Compute the lags
    ######################

    freq_lags, energy_lags = compute_lags(in_table)

    ##########
    ## Output
    ##########

    fits_out(out_file, freq_lags, energy_lags)

    ############
    ## Plotting
    ############

    plot_lag_freq(plot_root, plot_ext, prefix, freq_lags)

    plot_lag_energy(plot_root, energies_tab, plot_ext, prefix, energy_lags,
                    in_table.meta['DETCHANS'])


################################################################################
if __name__ == "__main__":

    #########################################
    ## Parse input arguments and call 'main'
    #########################################

    parser = argparse.ArgumentParser(usage="python get_lags.py infile outfile "\
            "[OPTIONAL ARGUMENTS]", description=__doc__,
            epilog="For optional arguments, default values are given in "\
            "brackets at end of description.")

    parser.add_argument('infile', help="Name of the FITS file containing the "\
            "cross spectrum, power spectrum of the channels of interest, and "\
            "power spectrum of the reference band.")

    parser.add_argument('outfile', help="Name of the FITS file to write the "\
            "lag spectra to.")

    parser.add_argument('energies_tab', help="Name of the txt file containing "\
            "a list of the keV energy boundaries of the detector energy "\
            "channels (so for 4 channels, would have 5 energies listed). "\
            "Created in rxte_reduce/channel_to_energy.py.")

    parser.add_argument('-o', dest='plot_root', default="./plot", help="Root "\
            "name for plots generated, to be appended with '_lag-freq.(ext"\
            "ension)' and '_lag-energy.(extension)'. [./plot]")

    parser.add_argument('--prefix', dest="prefix", default="--",
            help="The identifying prefix of the data (object nickname or "\
            "data ID). [--]")

    parser.add_argument('--ext', dest='plot_ext', default='eps',
            help="File extension for the plots. Do not include the dot. [eps]")

    parser.add_argument('--lf', dest='lo_freq', type=tools.type_positive_float,
            default=1.0, help="The lower bound of the frequency range to "\
            "average the energy lags over and plot the frequency lags in, "\
            "inclusive, in Hz. [1.0]")

    parser.add_argument('--uf', dest='up_freq', type=tools.type_positive_float,
            default=10.0, help="The upper bound of the frequency range to "\
            "average the energy lags over and plot the frequency lags in, "\
            "inclusive, in Hz. [10.0]")

    parser.add_argument('--le', dest='lo_chan', type=tools.type_positive_int,
            default=2, help="The lower bound of the energy channel range to "\
            "average the frequency lags over and plot the energy lags in, "\
            "inclusive, in detector energy channels. [2]")

    parser.add_argument('--ue', dest='up_chan', type=tools.type_positive_int,
            default=26, help="The upper bound of the energy channel range to "\
            "average the frequency lags over and plot the energy lags in, "\
            "inclusive, in detector energy channels. [25]")

    args = parser.parse_args()

    main(args.infile, args.outfile, args.energies_tab, args.plot_root,
            args.prefix, plot_ext=args.plot_ext, lo_freq=args.lo_freq,
            up_freq=args.up_freq, lo_chan=args.lo_chan,
            up_chan=args.up_chan)


################################################################################
