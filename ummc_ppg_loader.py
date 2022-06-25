import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import math

import torch
from torch.utils.data import Dataset

import heartpy as hp
from heartpy.exceptions import BadSignalWarning
from heartpy.datautils import rolling_mean
from heartpy.peakdetection import detect_peaks, fit_peaks

from numpy.lib.stride_tricks import as_strided

RR_WINDOW_SIZE = 60

FS = 128
DFS = 50

AF = 1
NON_AF = 0

LOW_FREQ_FILTER = (30 / 60)
HI_FREQ_FILTER = (240 / 60)


class PPGDataset(Dataset):
    def __init__(self, data_path, freq):
        self.freq = freq
        self.data_path = data_path
        data_files = os.listdir(data_path)

        # File discovery
        self.subjects = {f[:4] for f in data_files if f[:4].isdigit()}
        self.signal_files = [os.path.join(data_path, s + '.mat') for s in self.subjects]
        self.ground_truth_files = [os.path.join(data_path, s + '_ground_truth.mat') for s in self.subjects]

        # self.ground_truth_files = [f for f in data_files if 'ground_truth' in f]
        # self.peak_ref_files = [f for f in data_files if 'Peak_Ref_ECG' in f]
        # self.ecg_30_sec_files = [f for f in data_files if 'ECG_30sec' in f]
        # self.ref_ecg_files = [f for f in data_files if 'RefECG' in f]

        self.info_file = None
        if 'UMass_SimbandInfo.mat' in data_files:
            self.info_file = os.path.join(data_path, 'UMass_SimbandInfo.mat')

        # self.signal_files = set(data_files) - set(self.ground_truth_files + self.peak_ref_files + self.ecg_30_sec_files + self.ref_ecg_files + [self.info_file])
        # self.signal_files = list(self.signal_files)
        # print('signal_files:', self.signal_files)
        # self.subjects = [int(os.path.splitext(x)[0]) for x in self.signal_files]
        # self.subjects.sort()

        self.cache = {}
        self.data = []
        self.rr_intervals = []

    # def get_raw_ppg_signal(self, pid, channel=0):
    #     signal_file = os.path.join(self.data_path, f'{pid}.mat')
    #     _data = scipy.io.loadmat(signal_file, matlab_compatible=True, simplify_cells=True)
    #     if channel not in range(8):
    #         channel = 0
    #     return _data['data']['physiosignal']['ppg'][chr(ord('a') + channel)]['signal']

    def bandpass_filter(self, x, filter_order=2):
        return hp.filter_signal(x, [LOW_FREQ_FILTER, HI_FREQ_FILTER], sample_rate=self.freq, order=filter_order, filtertype='bandpass')

    def get_ppg_signal(self, pid, filter=True):
        signal_file = os.path.join(self.data_path, f'{pid}.mat')
        _data = scipy.io.loadmat(signal_file, matlab_compatible=True, simplify_cells=True)
        if filter:
            return self.bandpass_filter(_data['PPG_FULL'])
        return _data['PPG_FULL']

    def get_ground_truth(self, pid):
        gt_file = os.path.join(self.data_path, f'{pid}_ground_truth.mat')
        _data = scipy.io.loadmat(gt_file, matlab_compatible=True, simplify_cells=True)
        _gt = _data['disease_label']
        return (_gt[:,1:] == AF).flatten().astype(float)

    def process_signals(self):
        window_size = 30 * self.freq
        sub_window_size = window_size // 3

        for subject in self.subjects:
            sig = self.get_ppg_signal(subject)
            gt = self.get_ground_truth(subject)
            rrs = []
            gts = []
            for i in range(len(gt)):
                window = sig[window_size*i:window_size*(i+1)]
                rol_mean = rolling_mean(window, windowsize=1, sample_rate=DFS)
                wd = detect_peaks(window, rol_mean, ma_perc=20, sample_rate=DFS)
                rr = wd['RR_list']
                rrs.append(rr)
                gts.append(np.ones_like(rr) * gt[i])
            
            rrs = np.hstack(rrs)
            gts = np.hstack(gts)

            rrs = as_strided(rrs,shape=(len(rrs)//RR_WINDOW_SIZE, RR_WINDOW_SIZE))
            gts = as_strided(gts,shape=(len(gts)//RR_WINDOW_SIZE, RR_WINDOW_SIZE))

            prec = 0
            for i in range(rrs.shape[0]):
                # Filtering / Masking ignorance should go here
                self.rr_intervals.append((rrs[i], gts[i], prec, subject))
                prec += 1

                # for j in range(3):
                #     start_idx = window_size*i + sub_window_size*j
                #     end_idx = start_idx + sub_window_size
                #     self.data.append((sig[start_idx:end_idx], gt[i]))

    def window(self, x, freq, start=0, sec=10):
        start_idx = math.floor(start * freq)
        end_idx = math.floor(start_idx + sec * freq)
        return x[start_idx:end_idx]

    def __len__(self):
        return len(self.rr_intervals)

    def __getitem__(self, idx):
        return self.rr_intervals[idx]