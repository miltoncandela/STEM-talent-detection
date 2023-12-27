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

    nch = len(df.Ch1.unique()) + 1
    r = 2
    ncomb = int((factorial(nch))/(factorial(nch-r) * factorial(r)))
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
                b = np.zeros(shape=(ncomb * nband,))
                for i in range(nlect):
                    b = b + np.array(df_calib_temp.iloc[(i*b.shape[0]):((i+1)*b.shape[0]), 4])
                b /= nlect
            else:
                b = np.array(df_calib_temp.Cor)

            for lecture in df.Lecture.unique():
                df_temp = df[(df.Subject == subject) & (df.Lecture == lecture)]
                c[df_temp.index[0]:(df_temp.index[-1] + 1)] = b
                a[df_temp.index[0]:(df_temp.index[-1] + 1)] = (df_temp.Cor - b)/abs(b)
        else:
            print(subject, 'not found in Calib!')
            l_missing.append(subject)
            continue

    df['Cor'] = a
    df_calib = deepcopy(df)
    df_calib['Cor'] = c

    print(df)

    df = df[~df.Subject.isin(l_missing)].reset_index(drop=True)
    df_calib = df_calib[~df_calib.Subject.isin(l_missing)].reset_index(drop=True)

    df_calib.to_csv(df_name_calib + '_pross.csv', index=False)
    df.to_csv(df_name + '_norm.csv')
    exit()

norm_metric('cohe_coef_15min_TABS', 'cohe_coef_TABS_calib')

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


def calc_coh(s1, s2, band):
    fs = 250

    f, coh = coherence(s1, s2, fs=fs)
    val = np.mean(coh[(f >= freq_bands[band][0]) & (f <= freq_bands[band][1])])

    return val

    '''
    l = []
    # Coherence tomando en cuenta datos de 30 segundos por 15 minutos, entonces 30 samples
    for min in range(0, 30):  # min([s1.shape[0], s2.shape[0]])
        x = s1[(min * 30 * fs):((min + 1) * 30 * fs)]
        y = s2[(min * 30 * fs):((min + 1) * 30 * fs)]

        f, coh = coherence(x, y, fs=fs)
        val = np.mean(coh[(f >= freq_bands[band][0]) & (f <= freq_bands[band][1])])
        l.append(val)
    return l
    '''



def correlation(s1, s2):
    cross_cov = lambda x, y: np.correlate(x, y)
    return cross_cov(s1, s2)/sqrt(cross_cov(s1, s1)*cross_cov(s2, s2))


# Calculate correlation for each band, for both df_Prog and df_Desi, export each correlation to CSV
# Correlation between C1_Band, C2_Band
# freq_bands = {'Delta': [1, 4], 'Theta': [4, 8], 'Alpha': [8, 13], 'Beta': [13, 30], 'Gamma': [30, 49]}
freq_bands = {'Theta': [4, 8], 'Alpha': [8, 13], 'Beta': [13, 30]}
# freq_bands = {'Theta': [4, 8], 'AlphaL': [8, 10], 'AlphaH': [10, 12], 'Alpha': [8, 12],
#                                 'BetaL': [12, 20], 'BetaH': [20, 30], 'Beta': [12, 30]}
# freq_bands = {'Delta': [1, 4], 'Theta': [4, 8], 'Alpha': [8, 13], 'LowBeta': [13, 21], 'Gamma': [30, 49]}
# Cambiar Beta por LowBeta? Aumentaria los de Design ~ Robotics + 1 de Programming ~ Robotics

# freq_bands = {'Theta': [4, 8], 'Alpha': [8, 13], 'LowBeta': [13, 21], 'HighBeta': [21, 30], 'Gamma': [30, 49]}
channels = ['FP1', 'F3', 'C3', 'PZ', 'C4', 'F4', 'FP2']
samples = 250*60*4  # Removing the last 1 minute of the take

df_cor = pd.DataFrame(columns=['Subject', 'Lecture', 'Ch1', 'Ch2', 'Cor'])
for file in os.listdir('data/Frontiers/EEG_ICA_Csv_Calib'):  # used_children  # os.listdir('data/EEG_Frontiers'):
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
            for channel_j in channels:
                if (channel_i != channel_j) and ((channel_i, channel_j) not in combinations):
                    combinations.append((channel_j, channel_i))
                    combinations.append((channel_i, channel_j))

                    # val = correlation(calc_bands(df[channel_i], band), calc_bands(df[channel_j], band))
                    val = calc_coh(calc_bands(df[channel_i], band), calc_bands(df[channel_j], band), band)

                    data_dict = {'Subject': subject[:4], 'Lecture': lecture, 'Ch1': channel_i,
                                 'Ch2': channel_j, 'Band': band, 'Cor': val}
                    df_cor = pd.concat([df_cor, pd.DataFrame(data_dict, index=[0])], axis=0)
    print()

df_cor.reset_index(drop=True).to_csv('cohe_coef_TABS_calib.csv')
