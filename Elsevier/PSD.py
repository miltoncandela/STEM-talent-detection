import os
import pandas as pd
from brainflow.data_filter import DataFilter
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from scipy.signal import coherence
from statistics import mean
from math import factorial
from copy import deepcopy

# Limiting the data to 10 minutes and only using Programming
# df = pd.read_csv('data/EEG_Frontiers/DT01.csv', index_col=0)


used_children = ['DJ01', 'DJ02', 'DJ03',
                 'DT01',
                 'ES01', 'ES02',
                 'MJ01',
                 'MT01',
                 'ST01']

def norm_metric(df_name, df_name_calib):
    # name = corr_coef_nolast5min

    df = pd.read_csv(df_name + '.csv', index_col=0)
    df_calib = pd.read_csv(df_name_calib + '.csv', index_col=0)
    df_calib = df_calib[df_calib.Band.isin(df.Band.unique())]

    nch = len(df.Ch.unique())
    nband = len(df.Band.unique())

    a = np.zeros(shape=(df.shape[0],))
    c = np.zeros(shape=(df.shape[0],))
    l_missing = []

    for subject in df.Subject.unique():
        l = []
        l.append(subject)
        if set(l) <= set(df_calib.Subject.unique()):
            df_calib_temp = df_calib[df_calib.Subject == subject]
            nlect = len(df_calib_temp.Lecture.unique())
            if nlect > 1:
                b = np.zeros(shape=(nch * nband,))
                for i in range(nlect):
                    b = b + np.array(df_calib_temp.iloc[(i*b.shape[0]):((i+1)*b.shape[0]), 4])
                b /= nlect
            else:
                b = np.array(df_calib_temp.Power)

            for lecture in df.Lecture.unique():
                df_temp = df[(df.Subject == subject) & (df.Lecture == lecture)]
                c[df_temp.index[0]:(df_temp.index[-1] + 1)] = b
                a[df_temp.index[0]:(df_temp.index[-1] + 1)] = (df_temp.Power - b)/abs(b)
        else:
            print(subject, 'not found in Calib!')
            l_missing.append(subject)
            continue

    df['Power'] = a
    df_calib = deepcopy(df)
    df_calib['Power'] = c

    print(df)

    df = df[~df.Subject.isin(l_missing)].reset_index(drop=True)
    df_calib = df_calib[~df_calib.Subject.isin(l_missing)].reset_index(drop=True)

    df_calib.to_csv(df_name_calib + '_pross.csv', index=False)
    df.to_csv(df_name + '_norm.csv')
    exit()

norm_metric('PSD_TAB', 'PSD_TAB_calib')

def calc_bands(s, band):
    eta = 6
    sr = 250
    ft = 0

    # Assuming 6 s windows
    a = np.zeros(shape=(s.shape[0], ))
    f1, f2 = freq_bands[band]
    for w in range(s.shape[0] // sr):
        b = np.array(s.iloc[(w * sr):((w + 6) * sr)])

        # f1–f2 Hz 6th order Butterworth bandpass filter
        DataFilter.perform_lowpass(data=b, sampling_rate=sr, cutoff=f1, order=eta, filter_type=ft, ripple=0)
        DataFilter.perform_highpass(data=b, sampling_rate=sr, cutoff=f2, order=eta, filter_type=ft, ripple=0)

        # s[(w * sr):((w + 6) * sr)] = a
        a[(w * sr):((w + 6) * sr)] = b
        w += 6
    return a


def calc_PSD(s, band):
    """

    :param s: Signal of a given channel
    :return:
    """

    # EEG
    sr = 250  # Sampling frequency
    wf = 3  # Windows function (0: No Window, 1: Hanning, 2: Hamming, 3: Blackman Harris)
    nfft = DataFilter.get_nearest_power_of_two(sr)
    over = DataFilter.get_nearest_power_of_two(sr) // 2

    # Calculate the PSD using the Welch method with specified window parameters
    psd = DataFilter.get_psd_welch(data=np.array(s).astype(float), nfft=nfft,
                                   overlap=over, sampling_rate=sr, window=wf)

    # Calculate the average alpha power (e.g., for alpha frequency range of 8-13 Hz)
    power = DataFilter.get_band_power(psd, freq_bands[band][0], freq_bands[band][1])

    return power


# Calculate correlation for each band, for both df_Prog and df_Desi, export each correlation to CSV
# freq_bands = {'Theta': [4, 8], 'Alpha': [8, 13], 'Beta': [13, 30]}
freq_bands = {'Theta': [4, 8], 'AlphaL': [8, 10], 'AlphaH': [10, 12], 'Alpha': [8, 12],
                                'BetaL': [12, 20], 'BetaH': [20, 30], 'Beta': [12, 30]}
channels = ['FP1', 'F3', 'C3', 'PZ', 'C4', 'F4', 'FP2']
samples = 250*60*4  # Removing the last 1 minute of the take

df_PSD = pd.DataFrame(columns=['Subject', 'Lecture', 'Ch', 'Band', 'Power'])
for file in os.listdir('data/Frontiers/EEG_ICA_Csv_Calib'): # used_children: #   #   # os.listdir('data/EEG_Frontiers'):
    subject = file[:4]
    lecture = file[5:9]
    print(subject)
    # for lecture in ['Desi', 'Prog', 'Robo']:
    print(lecture)
    df = pd.read_csv('data/Frontiers/EEG_ICA_Csv_Calib/{}_{}.csv'.format(subject, lecture)).drop(0, axis=0)
    df = df.iloc[:250*60*1, :]  # Only gathering the first minute of data (calibration)
    # df = df.iloc[(250*60*4):(250*60*19), :] # Removing the first 4 minutes until 19 minutes (15 min)
    # df = df.iloc[samples:(df.shape[0] - samples + 250 * 60 * 1), :]  # Dropping the last 1 minute
    df.columns = channels
    for band in freq_bands.keys():
        combinations = []
        for channel_i in channels:
            val = calc_PSD(df[channel_i], band)

            data_dict = {'Subject': subject[:4], 'Lecture': lecture, 'Ch': channel_i, 'Band': band, 'Power': val}
            df_PSD = pd.concat([df_PSD, pd.DataFrame(data_dict, index=[0])], axis=0)
    print()

df_PSD.reset_index(drop=True).to_csv('PSD_TAB_calib.csv')
