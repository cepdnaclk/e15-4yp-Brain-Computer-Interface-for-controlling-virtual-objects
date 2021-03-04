# -*- coding: utf-8 -*-
from scipy.io import loadmat
import mne

mat = loadmat("../dataset/sample_data/A01T.mat")
eeg = mat["data"][0, 3]["X"][0, 0] * (10e-6)

ch_names = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz",
            "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz",
            "P2", "POz", "EOG1", "EOG2", "EOG3"]

info = mne.create_info(ch_names, 250, ch_types=["eeg"] * 22 + ["eog"] * 3)
raw = mne.io.RawArray(eeg.T, info)
raw.set_montage("standard_1020")

raw_tmp = raw.copy()
raw_tmp.filter(1, None)

ica = mne.preprocessing.ICA(method="infomax",
                            fit_params={"extended": True},
                            random_state=1)

ica.fit(raw_tmp)

ica.plot_components(inst=raw_tmp, picks=range(22))

ica.exclude = [1]  # component for removing
raw_corrected = raw.copy()
ica.apply(raw_corrected)

raw.plot(n_channels=25, start=54, duration=4,
         scalings=dict(eeg=250e-6, eog=750e-6))

raw_corrected.plot(n_channels=25, start=54, duration=4,
                   scalings=dict(eeg=250e-6, eog=750e-6))
