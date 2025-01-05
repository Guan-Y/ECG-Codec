import argparse
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
from wfdb import rdrecord
from scipy import signal

import matplotlib.pyplot as plt

# filter parameters (not used)
fs = 360
bp_freq_high = 150.
bp_freq_low = 0.5
wl = 2.*bp_freq_low/fs
wh = 2.*bp_freq_high/fs
b, a = signal.butter(4, [wl, wh], 'bandpass')

# generate_full_data = True # use this for debug

def Slice_And_Save(ecg, output_dir, name, lead):
    Ns = len(ecg) // segment_length
    for i in range(0, Ns):
        sig_name = f"{name}_{lead}_seg{i}"
        filename = osp.join(output_dir, "{}.npy".format(sig_name))

        np.save(filename, ecg[i * segment_length: (i + 1) * segment_length])

def test():
    test_sig = np.load("../data/full_data/mit-bih/100/MLII/100_MLII_seg0.npy")
    plt.plot(test_sig)
    plt.show()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=False, default="../datasets/mit-bih\\200")
    parser.add_argument("--full", required=False, default="False")
    parser.add_argument("--augment", required=False, default="False")
    args = parser.parse_args()

    ecg = args.file
    generate_full_data = eval(args.full)
    augment = eval(args.augment)

    if generate_full_data:
        output_dir = "../data/full_data"
        segment_length = 650000
    else:
        # split ecg into segment
        segment = 30
        segment_length = fs * segment
        output_dir = "../data/train_data"

    name = osp.basename(ecg)
    record = rdrecord(ecg)

    for sig_name, sig in zip(record.sig_name, record.p_signal.T):
        if not np.all(np.isfinite(sig)):
            continue

        if generate_full_data:
            sig_scaled = sig
        else:
            offset = np.min(sig)
            sig -= offset
            scale = np.max(np.abs(sig))
            sig_scaled = sig / scale

        one_dim_data_dir = osp.join(output_dir, "mit-bih", name, sig_name)
        os.makedirs(one_dim_data_dir, exist_ok=True)
        Slice_And_Save(sig_scaled, one_dim_data_dir, name, sig_name)

        '''use data augmentation'''
        if augment and ~generate_full_data:
            signal_time_reverse = np.flip(sig_scaled)
            signal_amp_reverse = -sig_scaled

            time_reverse_data_dir = osp.join(output_dir, "mit-bih-time-reverse", name, sig_name)
            os.makedirs(time_reverse_data_dir, exist_ok=True)
            Slice_And_Save(signal_time_reverse, time_reverse_data_dir, name, sig_name)

            amp_reverse_data_dir = osp.join(output_dir, "mit-bih-amp-reverse", name, sig_name)
            os.makedirs(amp_reverse_data_dir, exist_ok=True)
            Slice_And_Save(signal_amp_reverse, amp_reverse_data_dir, name, sig_name)

