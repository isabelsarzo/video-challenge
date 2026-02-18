import numpy as np
from scipy.signal import welch

def RMS(data: np.ndarray) -> np.ndarray:
    """
    Computes the root mean square (RMS) across each muscle channel in 'data'.

    Args:
        data (np.ndarray of shape (n_samples, n_muscles)): 
                The input sEMG or ACC data.

    Returns:
        np.ndarray of shape (n_muscles,): 
                The root mean square values of each muscle channel.

    """
    rms = np.sqrt(np.mean(np.square(data), axis=0))
    return rms

def ZCR(data:np.ndarray, threshold: int) -> list:
    """
    Computes the zero crossing rate (ZCR) across each muscle channel in 'data'.

    Args:
        data (np.ndarray of shape (n_samples, n_muscles)): 
                The input sEMG or ACC data.
        threshold (int): 
                The threshold value to be used when counting crossings. If 'threshold' 
                is crossed, the count is increased.

    Returns:
        list of length n_muscles: 
                The zero crossing rate of each muscle channel.
    """
    zcr = []
    for m in range(data.shape[1]):
        # Absolute values lower than threshold are set to zero:
        level = np.where(np.abs(data[:, m]) < threshold, 0, data[:, m])

        # Calculate differences in sign changes to get the crossings:
        crossings = np.diff(np.sign(level))

        # Count crossings and determine the rate:
        zcr.append(np.sum(crossings != 0) / len(data[:, m]))
    return zcr

def medFreq(data: np.ndarray, fs: int | float) -> list:
    """
    Computes the median frequency of each muscle channel using Welch's method to
    estimate the power spectral density. 

    The median frequency is the frequency that divides the power specturm into two
    halves of equal power. 

    Args:
        data (np.ndarray of shape (n_samples, n_muscles)): 
                The input sEMG or ACC data.
        fs (int | float): 
                The sampling frequency (Hz) of the input data.

    Returns:
        list of length n_muscles: 
                The median frequency of each muscle channel.

    """
    mf = []
    for m in range(data.shape[1]):
        freqs, psd = welch(data[:, m], fs, nperseg = 64)
        cpsd = np.cumsum(psd)
        tpower = cpsd[-1]
        mf.append(freqs[np.where(cpsd >= (tpower/2))[0][0]])
    return mf

def peak_freq(data: np.ndarray, fs: int | float) -> list:
    """
    Computes the peak frequency of each muscle channel using Welch's method to
    estimate the power spectral density. 

    Args:
        data (np.ndarray of shape (n_samples, n_muscles)): 
                The input sEMG or ACC data.
        fs (int | float): 
                The sampling frequency (Hz) of the input data.

    Returns:
        list of length n_muscles: 
                The peak frequency of each muscle channel.

    """
    pf = []
    for m in range(data.shape[1]):
        freqs, psd = welch(data[:, m], fs, nperseg = 64)
        pf.append(freqs[np.argmax(psd)])
    return pf

def variance(data: np.ndarray) -> np.ndarray:
    """
    Computes the variance across each muscle channel in 'data'.

    Args:
        data (np.ndarray of shape (n_smaples, n_muscles)): 
                The input sEMG or ACC data.

    Returns:
        np.ndarray of shape (n_mucles,): 
                The variance of each muscle channel.

    """
    var = np.var(data, axis=0)
    return var

def relative_power(data: np.ndarray, fs: int | float, freqband: tuple[int, int]) -> list:
    """
    Computes the relative power of each muscle channel in 'data' using Welch's method
    to estimate the power spectral density.

    The relative power measures the power in a specific frequency band in relation to
    the total power of the signal.

    NOTE: If total power is zero, then relative power is also returned as zero.

    Args:
        data (np.ndarray of shape (n_samples, n_muscles)): 
                The input sEMG or ACC data
        fs (int | float): 
                The sampling frequency (Hz) of the input data
        freqband (tuple[int, int]): 
                The frequency band (range) of interest in Hz. Example: [100, 500]

    Returns:
        list of length n_muscles: 
                The relative power of each muscle channel.

    """
    rp = []
    for m in range(data.shape[1]):
        freqs, psd = welch(data[:, m], fs=fs, nperseg= 64)
        total_power = np.sum(psd)
        band_power = np.sum(psd[(freqs >= freqband[0]) & (freqs <= freqband[1])])
        if total_power == 0: # signal is absent
            rp.append(0)
        else:
            rp.append(band_power / total_power)
    return rp

def jerk(data: np.ndarray, fs: int | float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes jerk across the three ACC axes (x, y, z) and ACC norm of each muscle channel in 'data'.

    Jerk is measured as the rate of change of acceleration (the derivate of the acceleration).

    Args:
        data (ndarray of shape (n_samples, n_muscles*4)): 
                The input ACC data, where each column represents an axis (x, y, z) or the norm of the
                accelerometer data for a specific muscle.
        fs (int | float): 
                The sampling frequency (Hz) of the input data.

    Returns:
        mean_jerk (ndarray of shape (n_muscles*4,)): 
                Mean jerk values of each column in the input data.
        max_jerk (ndarray of shape (n_muscles*4)): 
                Maximum jerk values of each column in the input data.
        std_jerk (ndarray of shape (n_muscles*4)): 
                Standard deviation of jerk for each column in the input data.

    """
    dt = 1 / fs
    jk = np.diff(data, axis=0) / dt
    mean_jerk = np.mean(jk, axis=0)
    max_jerk = np.max(jk, axis=0)
    std_jerk = np.std(jk, axis=0)
    return mean_jerk, max_jerk, std_jerk

def IQR(data: np.ndarray) -> np.ndarray:
    """
    Computes the interquartile range (IQR) of each of the three ACC axes (x, y, z) and
    ACC norm for each muscle channel in 'data'.

    The IQR measures variability in acceleration: 
        IQR = Q3 - Q1, 
        where Q3 and Q1 are the 75th and 25th percentiles, respectively.

    Args:
        data (np.ndarray of shape (n_samples, n_muscles*4)): 
                The input ACC data, where each column represents an axis (x, y, z) or the norm of the
                accelerometer data for a specific muscle.

    Returns:
        np.ndarray of shape (n_muscles*4,): 
                IQR values of each column in the input data.

    """
    q3 = np.percentile(data, 75, axis=0)
    q1 = np.percentile(data, 25, axis=0)
    iqr = q3 - q1
    return iqr

# TODO: correlation between axes
# TODO: Tilt angle change
# TODO: burst duration, burst amplitude, rise time / fall time
# TODO: sample entropy!!!
# TODO: kurtosis, skewness
# TODO: spectral centroid, spectral entropy!!!, band power ratios, spectral roll-off