#!/usr/bin/env
"""
Read in two extracted light curves (interest band and reference band), split
into segments, compute the power spectra per band and cross spectrum of each
segment, averages cross spectrum of all the segments, and computes frequency
lags between the two bands.

Example call:
python simple_cross_spectra.py ./cygx1_i.lc ./cygx1_ref.lc -o "./cygx1"

Enter   python simple_cross_spectra.py -h   at the command line for help.
"""
from __future__ import print_function
from astropy.table import Table, Column
from astropy.io import fits
import numpy as np
from scipy import fftpack
import argparse
import subprocess
from datetime import datetime

__author__ = "Abigail Stevens <A.L.Stevens at uva.nl>"
__year__ = "2016"


class Band(object):
    def __init__(self, n_bins=8192, dt=0.0078125):
        self.power = np.zeros(n_bins, dtype=np.float64)
        self.mean_rate = 0.0
        self.rms = 0.0
        self.freq = fftpack.fftfreq(n_bins, d=dt)


################################################################################
def type_power_of_two(num):
    """
    Check if an input is a power of 2 (1 <= num < 2147483648), as an argparse
    type.

    Parameters
    ----------
    num : int
        The number in question.

    Returns
    -------
    n : int
        The number in question, if it's a power of two

    Raises
    ------
    ArgumentTypeError if n isn't a power of two.

    """
    n = int(num)
    x = 2
    assert n > 0

    if n == 1:
        return n
    else:
        while x <= n and x < 2147483648:
            if n == x:
                return n
            x *= 2

    message = "%d is not a power of two." % n
    raise argparse.ArgumentTypeError(message)


################################################################################
def get_key_val(fits_file, ext, keyword):
    """
    Get the value of a keyword from a FITS header. Keyword does not seem to be
    case-sensitive.

    Parameters
    ----------
    fits_file : str
        File name of the FITS file.

    ext : int
        The FITS extension in which to search for the given keyword.

    keyword : str
        The keyword for which you want the associated value.

    Returns
    -------
    any type
        Value of the given keyword.

    Raises
    ------
    IOError if the input file isn't actually a FITS file.

    """

    ext = np.int8(ext)
    assert (ext >= 0 and ext <= 2)
    keyword = str(keyword)

    try:
        hdulist = fits.open(fits_file)
    except IOError:
        print("\tERROR: File does not exist: %s" % fits_file)
        exit()

    key_value = hdulist[ext].header[keyword]
    hdulist.close()

    return key_value


################################################################################
def raw_to_absrms(power, mean_rate, n_bins, dt, noisy=True):
    """
    Normalize the power spectrum to absolute rms^2 normalization.

    TODO: cite paper.

    Parameters
    ----------
    power : np.array of floats
        The raw power at each Fourier frequency, as a 1-D or 2-D array.
        Size = (n_bins) or (n_bins, detchans).

    mean_rate : float
        The mean count rate for the light curve, in cts/s.

    n_bins : int
        Number of bins per segment of light curve.

    dt : float
        Timestep between bins in n_bins, in seconds.

    noisy : boolean
        True if there is Poisson noise in the power spectrum (i.e., from real
        data), False if there is no noise in the power spectrum (i.e.,
        simulations without Poisson noise). Default is True.

    Returns
    -------
    np.array of floats
        The noise-subtracted power spectrum in absolute rms^2 units, in the
        same size array as the input power.

    """
    if noisy:
        noise = 2.0 * mean_rate
    else:
        noise = 0.0

    return power * (2.0 * dt / np.float(n_bins)) - noise


################################################################################
def var_and_rms(power, df):
    """
    Computes the variance and rms (root mean square) of a power spectrum.
    Assumes the negative-frequency powers have been removed. DOES NOT WORK ON
    2-D POWER ARRAYS! Not sure why.

    TODO: cite textbook or paper.

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
    variance = np.sum(power * df, axis=0)
    rms = np.where(variance >= 0, np.sqrt(variance), np.nan)

    return variance, rms


################################################################################
def cs_out(out_base, meta_dict, cs_avg, ci, ref):
    """
    Saving header data, the cross spectrum, CoI power spectrum, and reference
    band power spectrum to a FITS file to use in the program make_lags.py to get
    cross-spectral lags. Cross spectra and power spectra are raw, as in un-
    normalized.

    Parameters
    ----------
    out_base : str
        The name the FITS file to write the cross spectrum and power spectra to,
        for computing the lags.

    meta_dict : dict
        Dictionary of necessary meta-parameters for data analysis.

    cs_avg : np.array of complex numbers
        2-D array of the averaged cross spectrum. Size = (n_bins, detchans).

    ci : ccf_lc.Lightcurve object
        The channel of interest light curve. Must already have freq, mean_rate,
        and power assigned.

    ref : ccf_lc.Lightcurve object
        The reference band light curve. Must already have mean_rate, rms, and
        power assigned.

    Returns
    -------
    nothing, but writes to the file "*_cs.fits"

    """

    out_file = out_base + "_cs.fits"
    out_dir = out_file[0:out_file.rfind("/")+1]
    if len(out_dir) >= 2:
        subprocess.call(['mkdir', '-p', out_dir])
    print("Output sent to: %s" % out_file)

    out_table = Table()
    out_table.add_column(Column(data=ci.freq, name='FREQUENCY', unit='Hz'))
    out_table.add_column(Column(data=cs_avg, name='CROSS'))
    out_table.add_column(Column(data=ci.power, name='POWER_CI'))
    out_table.add_column(Column(data=ref.power, name='POWER_REF'))

    out_table.meta['TYPE'] = "Cross spectrum and power spectra, saved for lags."
    out_table.meta['DATE'] = str(datetime.now())
    out_table.meta['EVTLIST'] = " "
    out_table.meta['DT'] = meta_dict['dt']
    out_table.meta['DF'] = meta_dict['df']
    out_table.meta['N_BINS'] = meta_dict['n_bins']
    out_table.meta['SEGMENTS'] = meta_dict['n_seg']
    out_table.meta['SEC_SEG'] = meta_dict['n_seconds']
    out_table.meta['EXPOSURE'] = meta_dict['exposure']
    out_table.meta['DETCHANS'] = meta_dict['detchans']
    out_table.meta['RATE_CI'] = ci.mean_rate
    out_table.meta['RATE_REF'] = ref.mean_rate
    out_table.meta['RMS_REF'] = float(ref.rms)
    out_table.meta['NYQUIST'] = meta_dict['nyquist']
    out_table.write(out_file, overwrite=True)


################################################################################
def make_cs(rate_ci, rate_ref, meta_dict):
    """
    Generate the power spectra for each band and the cross spectrum for one
    segment of the light curve.

    Parameters
    ----------
    rate_ci : np.array of floats
        2-D array of the channel of interest light curve, Size = (n_bins).

    rate_ref : np.array of floats
        1-D array of the reference band lightcurve, Size = (n_bins).

    meta_dict : dict
        Dictionary of necessary meta-parameters for data analysis.

    Returns
    -------
    cs_seg : np.array of complex numbers
        2-D array of the cross spectrum of each channel of interest with the
        reference band.

    ci_seg : Band object
        The channel of interest light curve.

    ref_seg : Band object
        The reference band light curve.

    """
    assert np.shape(rate_ci) == (meta_dict['n_bins'], ),\
        "ERROR: CoI light curve has wrong dimensions. Must have size (n_bins, "\
        ")."
    assert np.shape(rate_ref) == (meta_dict['n_bins'], ), "ERROR: Reference "\
        "light curve has wrong dimensions. Must have size (n_bins, )."

    ci_seg = Band(n_bins=meta_dict['n_bins'], dt=meta_dict['dt'])
    ref_seg = Band(n_bins=meta_dict['n_bins'], dt=meta_dict['dt'])

    ## Computing the mean count rate of the segment
    ci_seg.mean_rate = np.mean(rate_ci)
    ref_seg.mean_rate = np.mean(rate_ref)

    ## Subtracting the mean off each value of 'rate'
    rate_sub_mean_ci = np.subtract(rate_ci, ci_seg.mean_rate)
    rate_sub_mean_ref = np.subtract(rate_ref, ref_seg.mean_rate)

    ## Taking the FFT of the time-domain photon count rate
    ## SciPy is faster than NumPy or pyFFTW for my array sizes
    fft_data_ci = fftpack.fft(rate_sub_mean_ci)
    fft_data_ref = fftpack.fft(rate_sub_mean_ref)

    ## Computing the power from the fourier transform
    ci_seg.power = np.absolute(fft_data_ci) ** 2
    ref_seg.power = np.absolute(fft_data_ref) ** 2

    ## Computing the cross spectrum from the fourier transform
    cs_seg = np.multiply(fft_data_ci, np.conj(fft_data_ref))

    return cs_seg, ci_seg, ref_seg


################################################################################
def lc_in(interest_file, ref_file, meta_dict):

    n_seg = 0
    interest_band = Band(n_bins=meta_dict['n_bins'], dt=meta_dict['dt'])
    ref_band = Band(n_bins=meta_dict['n_bins'], dt=meta_dict['dt'])
    cross_spec = np.zeros(meta_dict['n_bins'], dtype=np.complex128)

    ## Open the light curve files and load the data as astropy tables
    try:
        interest_table = Table.read(interest_file)
    except IOError:
        print("\tERROR: File does not exist: %s" % interest_file)
        exit()
    try:
        ref_table = Table.read(ref_file)
    except IOError:
        print("\tERROR: File does not exist: %s" % ref_file)
        exit()

    start_time_i = interest_table['TIME'][0]
    end_time_i = interest_table['TIME'][-1]
    start_time_r = ref_table['TIME'][0]
    end_time_r = ref_table['TIME'][-1]

    len_i = len(interest_table['TIME'])
    len_r = len(ref_table['TIME'])
    # print("i: %.15f \t %.15f" % (start_time_i, end_time_i))
    # print("r: %.15f \t %.15f" % (start_time_r, end_time_r))
    # print("len i:", len_i)
    # print("len r:", len_r)

    # assert len_i == len_r
    # assert start_time_i == start_time_r
    # assert end_time_i == end_time_r

    ## The following is in case the two files aren't the exact same length.
    a = 0  ## start of bin index to make segment of data
    c = 0
    b = meta_dict['n_bins']  ## end of bin index to make segment of data for
                             ## inner for-loop
    d = meta_dict['n_bins']

    if start_time_i > start_time_r:
        bin_diff = int((start_time_i - start_time_r) / meta_dict['dt'])
        assert bin_diff < len_r
        c += bin_diff
        d += bin_diff
    elif start_time_r > start_time_i:
        bin_diff = int((start_time_r - start_time_i) / meta_dict['dt'])
        assert bin_diff < len_i
        a += bin_diff
        b += bin_diff

    ## Loop through segments of the light curves
    while b <= len_i and d <= len_r:
        n_seg += 1

        ## Extract the count rates for each segment
        rate_ci = interest_table["RATE"][a:b]
        rate_ref = ref_table["RATE"][c:d]

        ## Compute the power spectra and cross spectrum for that segment
        cs_seg, ci_seg, ref_seg = make_cs(rate_ci, rate_ref, meta_dict)

        assert int(len(cs_seg)) == meta_dict['n_bins'], "ERROR: "\
                    "Something went wrong in make_cs. Length of cross spectrum"\
                    " segment  != n_bins."

        ## Keep running total (to be made into averages)
        cross_spec += cs_seg
        interest_band.power += ci_seg.power
        ref_band.power += ref_seg.power
        interest_band.mean_rate += ci_seg.mean_rate
        ref_band.mean_rate += ref_seg.mean_rate

        if (test is True) and (n_seg == 1):  ## For testing
            break

        ## Clear loop variables for the next round
        rate_ci = None
        rate_ref = None
        cs_seg = None
        ci_seg = None
        ref_seg = None

        ## Increment the counters and indices
        a = b
        c = d
        b += meta_dict['n_bins']
        d += meta_dict['n_bins']
        ## Since the for-loop goes from i to j-1 (since that's how the range
        ## function works) it's ok that we set i=j here for the next round.
        ## This will not cause double-counting rows or skipping rows.

    return cross_spec, interest_band, ref_band, n_seg


################################################################################
def freq_lag_out(out_base, meta_dict, freq, phase, err_phase, tlag, err_tlag,
                 ci_mean_rate, ref_mean_rate):
    """
    Saving header data, the cross spectrum, CoI power spectrum, and reference
    band power spectrum to a FITS file to use in the program make_lags.py to get
    cross-spectral lags. Cross spectra and power spectra are raw, as in un-
    normalized.

    Parameters
    ----------
    out_base : str
        The name the FITS file to write the cross spectrum and power spectra to,
        for computing the lags.

    meta_dict : dict
        Dictionary of necessary meta-parameters for data analysis.

    freq : np.array of floats
        1-D array of the Fourier frequencies against which the lag-frequency
        spectrum is plotted.

    phase, err_phase : np.array of floats
        The phase and error in phase of the frequency lags, in radians.

    tlag, err_tlag : np.array of floats
        The time and error in time of the frequency lags, in seconds.

    ci_mean_rate : floats
        The mean photon count rate of each of the interest band, in cts/s.

    ref_mean_rate : float
        The mean photon count rate of the reference band, in cts/s.

    Returns
    -------
    nothing, but writes to the file "*_lag-freq.fits"

    """

    out_file = out_base + "_lag-freq.fits"
    out_dir = out_file[0:out_file.rfind("/")+1]
    if len(out_dir) >= 2:
        subprocess.call(['mkdir', '-p', out_dir])
    print("Output sent to: %s" % out_file)

    out_table = Table()
    out_table.add_column(Column(data=freq, name='FREQUENCY', unit='Hz'))
    out_table.add_column(Column(data=phase, name='PHASE_LAG', unit='radian'))
    out_table.add_column(Column(data=err_phase, name='PHASE_LAG_ERR',
                                unit='radian'))
    out_table.add_column(Column(data=tlag, name='TIME_LAG', unit='s'))
    out_table.add_column(Column(data=err_tlag, name='TIME_LAG_ERR', unit='s'))

    out_table.meta['TYPE'] = "Lag-frequency spectrum"
    out_table.meta['DATE'] = str(datetime.now())
    out_table.meta['CS_DATA'] = out_base + "_cs.fits"
    out_table.meta['DT'] = meta_dict['dt']
    out_table.meta['DF'] = meta_dict['df']
    out_table.meta['N_BINS'] = meta_dict['n_bins']
    out_table.meta['SEGMENTS'] = meta_dict['n_seg']
    out_table.meta['SEC_SEG'] = meta_dict['n_seconds']
    out_table.meta['EXPOSURE'] = meta_dict['exposure']
    out_table.meta['DETCHANS'] = meta_dict['detchans']
    out_table.meta['RATE_CI'] = ci_mean_rate
    out_table.meta['RATE_REF'] = ref_mean_rate
    out_table.meta['NYQUIST'] = meta_dict['nyquist']
    out_table.write(out_file, overwrite=True)


################################################################################
def bias_term(ci, ref, meta_dict, n_range):
    """
    Compute the bias term to be subtracted off the cross spectrum to compute
    the covariance spectrum. Equation in Equation in footnote 4 (section 2.1.3,
    page 12) of Uttley et al. 2014.

    Assumes power spectra are raw (not at all normalized, and not Poisson-noise-
    subtracted).

    Parameters
    ----------
    ci : Band object
        The channel of interest or interest band. ci.power is raw (not
        normalized and not Poisson-noise-subtracted), of the frequencies to be
        averaged over, with size = 1 (if avg over freq) or n_bins/2+1 (if avg
        over energy). ci.mean_rate is in units of cts/s (size = 1 if avg over
        energy, size = detchans if avg over freq). Power and frequency
        have only the positive (tot en met Nyquist) frequencies.

    ref : Band object
        The reference band. ref.power is raw (not normalized and not Poisson-
        noise-subtracted), of the frequencies to be averaged over, with
        size = 1 (if avg over freq) or n_bins/2+1 (if avg over energy).
        ref.mean_rate is in units of cts/s. Power and frequency have only the
        positive (tot en met Nyquist) frequencies.

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
        Uttley et al. 2014. If you get an undefined error for the bias term,
        just set it equal to zero.

    """
    ## Compute the Poisson noise level in absolute rms units
    Pnoise_ref = ref.mean_rate * 2.0
    Pnoise_ci = ci.mean_rate * 2.0

    ## Normalizing power spectra to absolute rms normalization
    ## Not subtracting the noise (yet)!
    abs_ci = ci.power * (2.0 * meta_dict['dt'] / float(n_range))
    abs_ref = ref.power * (2.0 * meta_dict['dt'] / float(n_range))

    temp_a = (abs_ref - Pnoise_ref) * Pnoise_ci
    temp_b = (abs_ci - Pnoise_ci) * Pnoise_ref
    temp_c = Pnoise_ref * Pnoise_ci

    n_squared = (temp_a + temp_b + temp_c) / (n_range * meta_dict['n_seg'])
    return n_squared


################################################################################
def compute_coherence(cross_spec, ci, ref, meta_dict, n_range):
    """
    Compute the raw coherence of the cross spectrum. Coherence equation from
    Uttley et al 2014 eqn 11, bias term equation from footnote 4 on same page.
    Note that if the bias term gets way too wonky or undefined, it's usually
    tiny so you can just set it to zero.

    Parameters
    ----------
    cross_spec : np.array of complex numbers
        1-D array of the cross spectrum, averaged over the desired energy
        range or frequency range. Size = detchans (if avg over freq) or
        n_bins/2+1 (if avg over energy). Should be raw, not normalized or
        noise-subtracted. Eqn 9 of Uttley et al 2014.

    ci : Band object
        The channel of interest or interest band. ci.power is raw (not
        normalized and not Poisson-noise-subtracted), of the frequencies to be
        averaged over, with size = 1 (if avg over freq) or n_bins/2+1 (if avg
        over energy). ci.mean_rate is in units of cts/s (size = 1 if avg over
        energy, size = detchans if avg over freq). Power and frequency
        have only the positive (tot en met Nyquist) frequencies.

    ref : Band object
        The reference band. ref.power is raw (not normalized and not Poisson-
        noise-subtracted), of the frequencies to be averaged over, with
        size = 1 (if avg over freq) or n_bins/2+1 (if avg over energy).
        ref.mean_rate is in units of cts/s. Power and frequency have only the
        positive (tot en met Nyquist) frequencies.

    meta_dict : dict
        Dictionary of meta-parameters needed for analysis.

    n_range : int
        Number of bins averaged over for lags. Energy bins for frequency lags,
        frequency bins for energy lags. Same as K in equations in Section 2 of
        Uttley et al. 2014.

    Returns
    -------
    coherence : np.array of floats
        The raw coherence of the cross spectrum. (Uttley et al 2014, eqn 11)
        Size = n_bins/2+1 (if avg over energy) or detchans (if avg over freq).

    """

    cs_bias = bias_term(ci, ref, meta_dict, n_range)

    powers = ci.power * ref.power
    crosses = cross_spec * np.conj(cross_spec) - cs_bias
    with np.errstate(all='ignore'):
        coherence = np.where(powers != 0, crosses / powers, 0)
    # print("Coherence shape:", np.shape(coherence))
    # print(coherence)
    return np.real(coherence)


################################################################################
def get_phase_err(cs_avg, ci, ref, meta_dict, n_range):
    """
    Compute the error on the complex phase (in radians) via the coherence.
    Power should NOT be Poisson-noise-subtracted or normalized.

    Parameters
    ----------
    cs_avg : np.array of complex numbers
        1-D array of the raw cross spectrum, averaged over Fourier segments and
        energy channels or frequency bins.
        Size = detchans (if avg over freq) or n_bins/2+1 (if avg over energy).
        Eqn 9 of Uttley et al 2014.

    ci : Band object
        The channel of interest or interest band. ci.power is raw (not
        normalized and not Poisson-noise-subtracted), of the frequencies to be
        averaged over, with size = 1 (if avg over freq) or n_bins/2+1 (if avg
        over energy). ci.mean_rate is in units of cts/s (size = 1 if avg over
        energy, size = detchans if avg over freq). Power and frequency
        have only the positive (tot en met Nyquist) frequencies.

    ref : Band object
        The reference band. ref.power is raw (not normalized and not Poisson-
        noise-subtracted), of the frequencies to be averaged over, with
        size = 1 (if avg over freq) or n_bins/2+1 (if avg over energy).
        ref.mean_rate is in units of cts/s. Power and frequency have only the
        positive (tot en met Nyquist) frequencies.

    meta_dict :
        Dictionary of meta-paramters needed for analysis.

    n_range : int
        Number of bins averaged over for lags. Energy bins for frequency lags,
        frequency bins for energy lags. Same as K in equations in Section 2 of
        Uttley et al. 2014.

    Returns
    -------
    phase_err : np.array of floats
        1-D array of the error on the phase of the lag.

    """

    # print("Pow ci:", np.shape(power_ci))
    # print("Pow ref:", np.shape(power_ref))
    # print("Pow cs:", np.shape(cs_avg))

    coherence = compute_coherence(cs_avg, ci, ref, meta_dict, n_range)

    with np.errstate(all='ignore'):
        phase_err = np.sqrt(np.where(coherence != 0, (1 - coherence) /
                (2 * coherence * n_range * meta_dict['n_seg']), 0))

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
def main(interest_file, ref_file, out_base="./out", n_seconds=64, test=False):
    """
    Read in two extracted light curves (interest band and reference band),
    split into segments, compute the power spectra per band and cross spectrum
    of each segment, averages cross spectrum of all the segments, and computes
    frequency lags between the two bands.

    Parameters
    ----------
    interest_file : str
        The name of the .lc extracted light curve for the interest band.

    ref_file : str
        The name of the .lc extracted light curve for the reference band.

    out_base : str, default = "out"
        The base name to save output to. The extension will be appended to the
        end.

    n_seconds : int, default = 64
        Number of seconds in each Fourier segment. Must be a power of 2,
        positive.

    test : bool, default = False
        If true, only computes one segment of data. If false, runs like normal.
    """

    assert n_seconds > 0, "ERROR: Number of seconds per segment must be a "\
            "positive integer."

    try:
        t_res = float(get_key_val(interest_file, 0, 'TIMEDEL'))
    except KeyError:
        t_res = float(get_key_val(interest_file, 1, 'TIMEDEL'))

    try:
        t_res_ref = float(get_key_val(ref_file, 0, 'TIMEDEL'))
    except KeyError:
        t_res_ref = float(get_key_val(ref_file, 1, 'TIMEDEL'))

    assert t_res == t_res_ref, "ERROR: Interest band and reference band have "\
            "different time binnings. Code cannot currently cope with that."

    meta_dict = {'dt': t_res,  ## assumes light curves are binned to desired
                               ## resolution already
                 't_res': t_res,
                 'n_seconds': n_seconds,
                 'df': 1.0 / np.float(n_seconds),
                 'nyquist': 1.0 / (2.0 * t_res),
                 'n_bins': n_seconds * int(1.0 / t_res),
                 'detchans': 1,  ## only using 1 interest band
                 'exposure': 0,  ## will be computed later
                 'n_seg': 0}  ## will be computed later

    ## Read in from the light curve files, compute power spectra and cross
    ## spectrum
    total_cross, total_ci, total_ref, total_seg = lc_in(interest_file, ref_file,
                                                     meta_dict)

    ## Assign n_seg and exposure in meta_dict
    meta_dict['n_seg'] = total_seg
    meta_dict['exposure'] = meta_dict['dt'] * meta_dict['n_bins'] * \
                            meta_dict['n_seg']

    ## Turn running totals into averages
    total_cross /= np.float(meta_dict['n_seg'])
    total_ci.power /= np.float(meta_dict['n_seg'])
    total_ci.mean_rate /= np.float(meta_dict['n_seg'])
    total_ref.power /= np.float(meta_dict['n_seg'])
    total_ref.mean_rate /= np.float(meta_dict['n_seg'])

    ## Only keeping the parts associated with positive Fourier frequencies
    ## numpy arrays slice at end-1, and we want to include 'nyq_index';
    ## for frequency, abs is because the nyquist freq is both pos and neg, and
    ## we want it pos here.
    nyq_index = meta_dict['n_bins'] / 2
    total_cross = total_cross[0:nyq_index + 1]
    total_ci.power = total_ci.power[0:nyq_index + 1]
    total_ci.freq = np.abs(total_ci.freq[0:nyq_index + 1])
    total_ref.power = total_ref.power[0:nyq_index + 1]
    total_ref.freq = np.abs(total_ref.freq[0:nyq_index + 1])

    ## Compute the variance and rms of the absolute-rms-normalized reference
    ## band power spectrum
    absrms_ref_pow = raw_to_absrms(total_ref.power, total_ref.mean_rate,
                                   meta_dict['n_bins'], meta_dict['dt'],
                                   noisy=True)

    total_ref.var, total_ref.rms = var_and_rms(absrms_ref_pow, meta_dict['df'])

    ## Save cross spectrum and power spectra to "*_cs.fits"
    cs_out(out_base, meta_dict, total_cross, total_ci, total_ref)

    ## Computing frequency lags
    ## Negative sign is so that a positive lag is a hard energy lag
    phase = -np.arctan2(total_cross.imag, total_cross.real)
    # print(np.shape(phase))
    err_phase = get_phase_err(total_cross, total_ci, total_ref, meta_dict, 1)
    # print(np.shape(err_phase))
    tlag = phase_to_tlags(phase, total_ci.freq)
    err_tlag = phase_to_tlags(err_phase, total_ci.freq)

    ## Save lag-frequency spectrum to "*_lag-freq.fits"
    freq_lag_out(out_base, meta_dict, total_ci.freq, phase, err_phase, tlag,
                 err_tlag, total_ci.mean_rate, total_ref.mean_rate)


################################################################################
if __name__ == "__main__":

    #########################################
    ## Parse input arguments and call 'main'
    #########################################

    parser = argparse.ArgumentParser(usage="python simple_cross_spectra.py "
            "interest_band_file reference_band_file [OPTIONAL ARGUMENTS]",
            description=__doc__,
            epilog="For optional arguments, default values are given in "\
            "brackets at end of description.")

    parser.add_argument('interest_band_file', help="The .lc background-"\
            "subtracted extracted light curve for the interest band.")

    parser.add_argument('reference_band_file', help="The .lc background-"\
            "subtracted extracted light curve for the reference band. Assumes "\
            "it has the same time binning as the interest band.")

    parser.add_argument('-o', '--out', default="./out", dest='outbase',
                        help="The base name for output files. Extension will be"
                             " appended. [./out]")

    parser.add_argument('-n', '--n_sec', type=type_power_of_two, default=64,
                        dest='n_seconds', help="Number of seconds in each "
                                               "Fourier segment. Must be a "
                                               "power of 2, positive, integer. "
                                               "[64]")

    parser.add_argument('--test', type=int, default=0, choices={0,1},
                        dest='test', help="Int flag: 0 if computing all "
                                          "segments, 1 if only computing one "
                                          "segment for testing. [0]")

    args = parser.parse_args()

    test = False
    if args.test == 1:
        test = True

    main(args.interest_band_file, args.reference_band_file,
            out_base=args.outbase, n_seconds=args.n_seconds, test=test)
