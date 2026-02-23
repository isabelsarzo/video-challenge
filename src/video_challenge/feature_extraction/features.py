import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis as kurt
from scipy.stats import skew

def compute_psd(data: np.ndarray, fs: float):
    """
    Computes Welch PSD for all channels simultaneously.

    Returns:
        freqs: (n_freqs,)
        psd: (n_freqs, n_channels)
    """
    freqs, psd = welch(data, fs=fs, axis=0, nperseg=64)
    return freqs, psd

def adaptive_threshold(data, k=4):
    median = np.median(data, axis=0)
    mad = np.median(np.abs(data - median), axis=0)
    return median + k * mad

def compute_magnitude(data):
    n_samples, n_channels = data.shape
    n_landmarks = n_channels // 3
    reshaped = data.reshape(n_samples, n_landmarks, 3)
    mag = np.sqrt(np.sum(reshaped**2, axis=2))
    return mag

def RMS(data: np.ndarray) -> np.ndarray:
    """
    Computes the root mean square (RMS) of all channels (i.e., columns) in data.

    Args:
        data (np.ndarray of shape (n_samples, n_channels)): 
            Input data.

    Returns:
        np.ndarray of shape "data.shape[1]": Root mean square values of all channels.

    """
    rms = np.sqrt(np.mean(np.square(data), axis=0))
    return rms

def ZCR(data:np.ndarray, threshold: int) -> list:
    """
    Computes the zero crossing rate (ZCR) of all channels (i.e., columns) in data.
    Crossings are computed based on the number of times that "threshold" is crossed.

    Args:
        data (np.ndarray of shape (n_samples, n_channels)): 
            Input data.
        threshold (int): 
            The threshold value to be used when counting crossings. If 'threshold' 
            is crossed, the count is increased.

    Returns:
        list of length "data.shape[1]": 
                The zero crossing rate of all channels.
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

def medfreq(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """
    Computes median frequency per channel.

    Returns:
        np.ndarray (n_channels,)
    """
    cumulative = np.cumsum(psd, axis=0)
    total_power = cumulative[-1, :]
    half_power = total_power / 2

    # First index where cumulative >= half_power for each channel
    idx = np.argmax(cumulative >= half_power, axis=0)

    return freqs[idx]

def peak_freq(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """
    Computes peak frequency per channel.

    Returns:
        np.ndarray (n_channels,)
    """
    idx = np.argmax(psd, axis=0)
    return freqs[idx]

def variance(data: np.ndarray) -> np.ndarray:
    """
    Computes the variance across each channel in 'data'.

    Args:
        data (np.ndarray of shape (n_smaples, n_channels)): 
                The input data.

    Returns:
        np.ndarray of shape (n_channels,): 
                The variance of each channel.

    """
    var = np.var(data, axis=0)
    return var

def relative_power(freqs, psd, freqband: tuple) -> np.ndarray:
    """
    Computes the relative power of each channel in 'data' using Welch's method
    to estimate the power spectral density.

    The relative power measures the power in a specific frequency band in relation to
    the total power of the signal.

    NOTE: If total power is zero, then relative power is also returned as zero.

    Args:
        freqs: (n_freqs,)
        psd: (n_freqs, n_channels)
        freqband: (low, high)

    Returns:
        np.ndarray (n_channels,)

    """
    mask = (freqs >= freqband[0]) & (freqs <= freqband[1])

    band_power = np.sum(psd[mask, :], axis=0)
    total_power = np.sum(psd, axis=0)

    return np.divide(
        band_power,
        total_power,
        out=np.zeros_like(band_power),
        where=total_power != 0
    )

def jerk(data: np.ndarray, fs: int | float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the jerk of each channel in 'data'. Jerk is measured as the rate of change
    of acceleration (the derivate of the acceleration).

    Args:
        data (ndarray of shape (n_samples, n_channels)): 
                The input ACC data.
        fs (int | float): 
                The sampling frequency (Hz) of the input data.

    Returns:
        mean_jerk (ndarray of shape (n_channels,)): 
                Mean jerk values of each column in the input data.
        max_jerk (ndarray of shape (n_channels)): 
                Maximum jerk values of each column in the input data.
        std_jerk (ndarray of shape (n_channels)): 
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
    Computes the interquartile range (IQR) of each channel in 'data'.

    The IQR measures variability in acceleration: 
        IQR = Q3 - Q1, 
        where Q3 and Q1 are the 75th and 25th percentiles, respectively.

    Args:
        data (np.ndarray of shape (n_samples, n_channels)): 
                The input data.

    Returns:
        np.ndarray of shape (n_channels): 
                IQR values of each column in the input data.

    """
    q3 = np.percentile(data, 75, axis=0)
    q1 = np.percentile(data, 25, axis=0)
    iqr = q3 - q1
    return iqr

def sample_entropy(data: np.ndarray, m: int = 2, r: float | None = None) -> np.ndarray:
    """
    Computes sample entropy (SampEn) per channel.

    Args:
        data (ndarray of shape (n_samples, n_channels)): 
            Input data.
        m (int, optional): 
            Embedding dimension
        r (float | None, optional): 
            Tolerance (if None, 0.2 * std of channel)

    Returns:
        np.ndarray (n_channels,)
    """

    n_samples, n_channels = data.shape
    se = np.zeros(n_channels)

    for ch in range(n_channels):
        x = data[:, ch]
        if r is None:
            r_ch = 0.2 * np.std(x)
        else:
            r_ch = r

        def _phi(m):
            x_m = np.array([x[i:i+m] for i in range(n_samples - m)])
            C = 0
            for i in range(len(x_m)):
                dist = np.max(np.abs(x_m - x_m[i]), axis=1)
                C += np.sum(dist <= r_ch) - 1
            return C

        B = _phi(m)
        A = _phi(m+1)

        if B == 0 or A == 0:
            se[ch] = 0
        else:
            se[ch] = -np.log(A / B)
    return se

def spectral_entropy(psd: np.ndarray) -> np.ndarray:
    """
    psd shape: (n_freqs, n_channels)
    """
    psd_norm = psd / (np.sum(psd, axis=0, keepdims=True) + 1e-12)
    psd_norm += 1e-12
    return -np.sum(psd_norm * np.log2(psd_norm), axis=0)

def kurtosis(data: np.ndarray) -> np.ndarray:
    return kurt(data, axis=0, fisher=True)

def skewness(data: np.ndarray) -> np.ndarray:
    return skew(data, axis=0)

def band_power_ratio(freqs: np.ndarray, psd: np.ndarray,
                          band1: tuple, band2: tuple):
    
    mask1 = (freqs >= band1[0]) & (freqs <= band1[1])
    mask2 = (freqs >= band2[0]) & (freqs <= band2[1])

    p1 = np.sum(psd[mask1, :], axis=0)
    p2 = np.sum(psd[mask2, :], axis=0)

    return np.divide(p1, p2, out=np.zeros_like(p1), where=p2!=0)

def axis_correlation(data: np.ndarray):
    """
    Returns:
        shape (n_landmarks, 3)
        [xy, xz, yz]
    """
    n_landmarks = data.shape[1] // 3
    reshaped = data.reshape(data.shape[0], n_landmarks, 3)

    x = reshaped[:, :, 0]
    y = reshaped[:, :, 1]
    z = reshaped[:, :, 2]

    def corr(a, b):
        return np.sum((a - a.mean(0)) * (b - b.mean(0)), axis=0) / (
            np.sqrt(np.sum((a - a.mean(0))**2, axis=0) *
                    np.sum((b - b.mean(0))**2, axis=0)) + 1e-12)

    xy = corr(x, y)
    xz = corr(x, z)
    yz = corr(y, z)

    return np.vstack((xy, xz, yz)).T

def burst_duration(data: np.ndarray) -> np.ndarray:
    """
    Mean burst duration per channel.
    """
    threshold = adaptive_threshold(np.abs(data), k=4)

    n_samples, n_channels = data.shape
    durations = np.zeros(n_channels)

    for ch in range(n_channels):
        above = np.abs(data[:, ch]) > threshold[ch]
        burst_lengths = []
        count = 0
        for val in above:
            if val:
                count += 1
            elif count > 0:
                burst_lengths.append(count)
                count = 0
        if count > 0:
            burst_lengths.append(count)

        durations[ch] = np.mean(burst_lengths) if burst_lengths else 0

    return durations

def burst_amplitude(data: np.ndarray) -> np.ndarray:
    """
    Mean burst peak amplitude per channel.
    """
    threshold = adaptive_threshold(np.abs(data), k=4)

    n_channels = data.shape[1]
    amp = np.zeros(n_channels)

    for ch in range(n_channels):
        x = data[:, ch]
        bursts = np.abs(x) > threshold[ch]
        peak_vals = []

        start = None
        for i in range(len(x)):
            if bursts[i] and start is None:
                start = i
            elif not bursts[i] and start is not None:
                peak_vals.append(np.max(np.abs(x[start:i])))
                start = None

        amp[ch] = np.mean(peak_vals) if peak_vals else 0

    return amp

def rise_fall_ratio(data: np.ndarray) -> np.ndarray:
    """
    Computes log(rise/fall) per channel.
    
    Answers thw question: Is acceleration building up faster than it decays?

    Args:
        data: shape (n_samples, n_features)

    Returns:
        np.ndarray (n_features,)
    """
    ddata = np.diff(data, axis=0)

    rise = np.nanmean(np.where(ddata > 0, ddata, np.nan), axis=0)
    fall = np.nanmean(np.where(ddata < 0, -ddata, np.nan), axis=0)

    rise = np.nan_to_num(rise)
    fall = np.nan_to_num(fall)

    ratio = np.divide(rise, fall, out=np.ones_like(rise), where=fall != 0)

    return np.log(ratio + 1e-12)

def spectral_centroid(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    return np.sum(freqs[:, None] * psd, axis=0) / (np.sum(psd, axis=0) + 1e-12)

def tilt_angle_change(data: np.ndarray):
    n_landmarks = data.shape[1] // 3
    reshaped = data.reshape(data.shape[0], n_landmarks, 3)

    x = reshaped[:, :, 0]
    y = reshaped[:, :, 1]
    z = reshaped[:, :, 2]

    magnitude = np.sqrt(x**2 + y**2 + z**2) + 1e-12
    tilt = np.arccos(z / magnitude)

    return np.std(np.diff(tilt, axis=0), axis=0)

def spectral_rolloff(freqs: np.ndarray, psd: np.ndarray, roll_percent=0.85):
    cumulative = np.cumsum(psd, axis=0)
    threshold = roll_percent * cumulative[-1, :]
    idx = np.argmax(cumulative >= threshold, axis=0)
    
    return freqs[idx]

def generate_channel_names(n_channels: int, axes=("X", "Y", "Z", "MAG")):
    """
    Generates channel labels like:
    X_0, Y_0, Z_0, MAG_0, X_1, Y_1, ...
    """

    if n_channels % len(axes) != 0:
        raise ValueError("n_channels must be divisible by number of axes.")

    n_landmarks = n_channels // len(axes)
    names = []

    for i in range(n_landmarks):
        for axis in axes:
            names.append(f"{axis}_{i}")

    return names

def features_to_dataframe(features: dict, n_channels: int) -> pd.DataFrame:
    """
    Converts feature dictionary into (1, n_features) DataFrame
    with column names FEATURE_AXIS_LANDMARK
    """

    row = {}

    axes = ("X", "Y", "Z", "MAG") # ACC axes and magnitude
    n_axes = len(axes)

    if n_channels % n_axes != 0:
        raise ValueError("n_channels must be divisible by 4 (X,Y,Z,MAG).")

    n_landmarks = n_channels // n_axes
    channel_labels = generate_channel_names(n_channels, axes=axes)

    for feat_name, values in features.items():
        values = np.asarray(values)

        # ---- Per channel (X,Y,Z,MAG)
        if values.shape == (n_channels,):
            for label, val in zip(channel_labels, values):
                row[f"{feat_name}_{label}"] = val

        # ---- Per landmark (e.g., TAC)
        elif values.shape == (n_landmarks,):
            for i, val in enumerate(values):
                row[f"{feat_name}_LMK_{i}"] = val

        # ---- Correlation (landmark-level axis relationships)
        elif values.shape == (n_landmarks, 3):
            corr_labels = ["XY", "XZ", "YZ"]
            for i in range(n_landmarks):
                for j, corr_label in enumerate(corr_labels):
                    row[f"{feat_name}_{corr_label}_{i}"] = values[i, j]

        else:
            raise ValueError(
                f"Unexpected shape for feature {feat_name}: {values.shape}"
            )

    return pd.DataFrame([row])

def interleave(d1: np.ndarray, d2: np.ndarray, n: int = 3) -> np.ndarray:
    """
    Interleave one column from d2 after every n columns of d1.

    Parameters
    ----------
    d1 : np.ndarray (n_samples, n_features)
        Main data (e.g., X,Y,Z per sensor).
    d2 : np.ndarray (n_samples, n_insert)
        Data to insert (e.g., magnitude per sensor).
    n : int
        Number of consecutive columns from d1 before inserting one from d2.

    Returns
    -------
    np.ndarray
        Interleaved array.
    """

    if d1.shape[0] != d2.shape[0]:
        raise ValueError("d1 and d2 must have equal number of rows.")

    if d1.shape[1] % n != 0:
        raise ValueError("Number of columns in d1 must be divisible by n.")

    n_landmarks = d1.shape[1] // n

    if d2.shape[1] != n_landmarks:
        raise ValueError(
            "d2 must have exactly one column per n columns of d1."
        )

    n_samples = d1.shape[0]

    # Reshape d1 to (samples, sensors, n)
    d1_reshaped = d1.reshape(n_samples, n_landmarks, n)

    # Expand d2 to (samples, sensors, 1)
    d2_expanded = d2[:, :, np.newaxis]

    # Concatenate along last axis → (samples, sensors, n+1)
    combined = np.concatenate([d1_reshaped, d2_expanded], axis=2)

    # Reshape back to 2D
    return combined.reshape(n_samples, n_landmarks * (n + 1))