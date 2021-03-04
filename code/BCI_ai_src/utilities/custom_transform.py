# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 00:26:15 2020

@author: Prophet X
"""
import numpy as np
from pywt import wavedec
from scipy.fft import fft, rfft, rfftfreq, fftfreq

# Transformation files.
def perform_RFFT(channel_mat, N, pass_high, pass_low, zero_filler=False):
    timeStep = 1.0 / 250
    n = N  # channel 1
    freq_bins = rfftfreq(n, d=timeStep)  # frequency bins
    freqSpectrum = rfft(channel_mat)  # spectram
    extracted_freq = np.zeros(freq_bins.shape)

    start_index = 0  # bin start index
    end_index = freq_bins.shape[0] - 1  # bin end index

    for i in range(extracted_freq.shape[0]):
        if freq_bins[i] < pass_low:
            start_index = start_index + 1
        if freq_bins[i] > pass_high:
            end_index = end_index - 1

    # if zero_filler:
    #     extracted_freq[start_index:end_index] = freqSpectrum[start_index:end_index]
    #     return extracted_freq
    # else:

    return freqSpectrum[end_index:start_index]


# Transformation files.
def perform_FFT(channel_mat, N, pass_high, pass_low, zero_filler=False):
    timeStep = 1.0 / 250
    n = N  # channel 1
    freq_bins = fftfreq(n, d=timeStep)[:N // 2]  # frequency bins
    freqSpectrum = 2.0 / N * np.abs(fft(channel_mat)[0:N // 2])  # spectram

    end_index = 0  # bin start index
    start_index = freq_bins.shape[0] - 1  # bin end index

    for i in range(freq_bins.shape[0]):
        if freq_bins[i] < pass_low:
            end_index = end_index + 1
        if freq_bins[i] > pass_high:
            start_index = start_index - 1

    if zero_filler:
        extracted_freq = np.zeros(freq_bins.shape)
        extracted_freq[start_index:end_index] = freqSpectrum[start_index:end_index]
        return extracted_freq
    else:
        return freqSpectrum[start_index:end_index]


# Transformation files.
def perform_wavelet(channel_mat):
    # Daubechies wavelet i.e. db5 (level=5).
    w_coefficients = wavedec(data=channel_mat,  # data
                             wavelet='db4',  # wavelet type
                             level=5)  # number of decomposition level
    cA5, cD5, cD4, cD3, cD2, cD1 = w_coefficients.copy()

    result = np.hstack((cD4, cD3))

    return result
