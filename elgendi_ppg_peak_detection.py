import matplotlib.pyplot as plt
import scipy.signal
import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.signal


def elegendi_ppg_findpeaks(ppg_cleaned, sampling_rate=100, show=False):
    peaks = _ppg_findpeaks_elgendi(ppg_cleaned, sampling_rate, show=show)
    # Prepare output.
    output = {"PPG_Peaks": peaks}

    return output


def signal_smooth(signal, size=10):
    if isinstance(signal, pd.Series):
        signal = signal.values

    length = len(signal)

    # Check length.
    size = int(size)
    if size > length or size < 1:
        raise TypeError("'size' should be between 1 and length of the signal.")

    smoothed = scipy.ndimage.uniform_filter1d(signal, size, mode="nearest")
    return smoothed


def _ppg_findpeaks_elgendi(
    signal, sampling_rate=1000, peakwindow=0.111, beatwindow=0.667, beatoffset=0.02, mindelay=0.3, show=False):
    if show:
        __, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax0.plot(signal, label="filtered")

    # Ignore the samples with negative amplitudes and square the samples with
    # values larger than zero.
    signal_abs = signal.copy()
    signal_abs[signal_abs < 0] = 0
    sqrd = signal_abs ** 2

    # Compute the thresholds for peak detection. Call with show=True in order
    # to visualize thresholds.
    ma_peak_kernel = int(np.rint(peakwindow * sampling_rate))
    ma_peak = signal_smooth(sqrd, size=ma_peak_kernel)

    ma_beat_kernel = int(np.rint(beatwindow * sampling_rate))
    ma_beat = signal_smooth(sqrd, size=ma_beat_kernel)

    thr1 = ma_beat + beatoffset * np.mean(sqrd)  # threshold 1

    if show:
        ax1.plot(sqrd, label="squared")
        ax1.plot(thr1, label="threshold")
        ax1.legend(loc="upper right")

    # Identify start and end of PPG waves.
    waves = ma_peak > thr1
    beg_waves = np.where(np.logical_and(np.logical_not(waves[0:-1]), waves[1:]))[0]
    end_waves = np.where(np.logical_and(waves[0:-1], np.logical_not(waves[1:])))[0]
    # Throw out wave-ends that precede first wave-start.
    end_waves = end_waves[end_waves > beg_waves[0]]

    # Identify systolic peaks within waves (ignore waves that are too short).
    num_waves = min(beg_waves.size, end_waves.size)
    min_len = int(np.rint(peakwindow * sampling_rate))  # this is threshold 2 in the paper
    min_delay = int(np.rint(mindelay * sampling_rate))
    peaks = [0]

    for i in range(num_waves):

        beg = beg_waves[i]
        end = end_waves[i]
        len_wave = end - beg

        if len_wave < min_len:
            continue

        # Visualize wave span.
        if show:
            ax1.axvspan(beg, end, facecolor="m", alpha=0.5)

        # Find local maxima and their prominence within wave span.
        data = signal[beg:end]
        locmax, props = scipy.signal.find_peaks(data, prominence=(None, None))

        if locmax.size > 0:
            # Identify most prominent local maximum.
            peak = beg + locmax[np.argmax(props["prominences"])]
            # Enforce minimum delay between peaks.
            if peak - peaks[-1] > min_delay:
                peaks.append(peak)

    peaks.pop(0)

    if show:
        ax0.scatter(peaks, signal_abs[peaks], c="r")

    peaks = np.asarray(peaks).astype(int)
    return peaks