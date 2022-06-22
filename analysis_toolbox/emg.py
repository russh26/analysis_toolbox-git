# -*- coding: utf-8 -*-
"""
Created on Wed, Jun 22 13:50:15 2022

@author: Russell Hardesty, PhD,
"""
import numpy as np
from scipy.signal import find_peaks

# load packages
import matplotlib as mpl
import matplotlib.pyplot as plt

# set global parameters
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "Arial"


def moving_window_rms(data, window_size):

    smoothed_data = []

    for ind in range(len(data) - window_size):
        y = np.sqrt(np.mean(data[ind : ind + window_size] ** 2))
        smoothed_data.append(y)

    return smoothed_data


def emg_snr(data, fs, plot_flag):
    """

    **DESCRIPTION**

    Calculate signal-to-noise ratio for cyclic EMG as described in Agostini and Knaflitz, 2012 

    Agostini, V. , & Knaflitz, M. (2012) An algorithm for hte estimation of signal-to-noise ratio in surface myoelectric signals generated during cyclic movements. IEEE Transactions on Biomedical Engineering, 59(1), 219-225. doi: 10.11.1109/TBME.2001.2170687

    Link: http://ieeexplore.ieee.org/abstract/document/6035761/

    
    ------------------------------------------------------------------------------
    **INPUTS**
    
    data
        <array> 1D array of EMG values [nx1]

    fs
        <int> sampling frequency in Hz

    plot_flag
        <boolean> flag to plot original signal and histogram
    
    ------------------------------------------------------------------------------
    **RETURNS**
    
    E_noise
        <float> root-mean-square of the noise content

    SNR
        <float> signal-to-noise ration (in dB)
    
    DC
        <float> double, duty-cycle (relative duration of muscle activity throughout the signal (%))
    
    ------------------------------------------------------------------------------
    **EXAMPLE:**
    
    """

    N = len(data)

    # r = 10 suggested by Agostini and Knaflitz for a sample frequency of 2000Hz
    r = int(10 / (2000 / fs))
    M = N / r

    Cr = []
    for k in range(int(M) - 1):
        Cr_k = np.sum((data[k * r : k * r + r] ** 2) / r)
        Cr.append(Cr_k)

    num_bins = 60
    bins = []

    for m in range(1, 2 * num_bins - 1, 2):
        bin_m = m * (np.max(np.log10(Cr) - min(np.log10(Cr)))) / (
            2 * num_bins
        ) + np.min(np.log10(Cr))
        bins.append(bin_m)

    counts, _, _ = plt.hist(np.log10(Cr), bins)
    plt.show()

    # smoothing (not sure if this is needed)
    smoothed_counts = moving_window_rms(counts, 7)

    pks = find_peaks(smoothed_counts, distance=3)

    if len(pks[0]) != 2:
        print("Two peaks not found.")

    inx_noise = pks[0][0]
    inx_sig = pks[0][1]

    bins = np.array(bins)

    P_noise = (
        sum(
            10 ** bins[inx_noise - 2 : inx_noise + 2]
            * counts[inx_noise - 2 : inx_noise + 2]
        )
    ) / sum(counts[inx_noise - 2 : inx_noise + 2])

    P_sig = (
        sum(10 ** bins[inx_sig - 2 : inx_sig + 2] * counts[inx_sig - 2 : inx_sig + 2])
    ) / sum(counts[inx_sig - 2 : inx_sig + 2])

    E_noise = np.sqrt(P_noise)
    SNR = 10 * np.log10((P_sig - P_noise) / P_noise)
    DC = (
        100
        * sum(counts[inx_sig - 2 : inx_sig + 2])
        / (
            sum(counts[inx_sig - 2 : inx_sig + 2])
            + sum(counts[inx_noise - 2 : inx_noise + 2])
        )
    )

    return E_noise, SNR, DC


# testing
filename = r"C:\SynologyDrive\SynologyDrive\projects\emg_learning\database\EMGL003\77677976_EMGL003_BL001_MVC_trial2_EMG.csv"

data = np.genfromtxt(filename, delimiter=",")
fs = 1000

# plt.plot(data[:, 1])
# plt.show()

E_noise, SNR, DC = emg_snr(data[:, 1], fs, False)

print(E_noise)
print(SNR)
print(DC)

