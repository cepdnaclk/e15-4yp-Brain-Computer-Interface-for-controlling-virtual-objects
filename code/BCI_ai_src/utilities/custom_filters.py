# -*- coding: utf-8 -*-
from scipy.signal import butter, lfilter
from scipy.signal import iirfilter


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def Notch_Filter(Fs, band, freq, ripple, order, filter_type, data):
    fs = Fs
    nyq = fs / 2.0
    low = freq - band / 2.0
    high = freq + band / 2.0
    low = low / nyq
    high = high / nyq
    b, a = iirfilter(order,
                     [low, high],
                     rp=ripple,
                     btype='bandstop',
                     analog=False,
                     ftype=filter_type)
    filtered_data = lfilter(b, a, data)
    return filtered_data
