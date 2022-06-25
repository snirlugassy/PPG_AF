import heartpy as hp
import config


def bandpass_filter(x, freq, filter_order=2):
    return hp.filter_signal(x, [config.LOW_FREQ_FILTER, config.HI_FREQ_FILTER], sample_rate=freq, order=filter_order, filtertype='bandpass')


def heartpy_peak_detection():
    pass


def elgendi_peak_detection():
    pass


def split_windows():
    pass


def get_window(sig, size, pos):
    start_idx = pos * size
    end_idx = start_idx + size
    return sig[start_idx:end_idx]
