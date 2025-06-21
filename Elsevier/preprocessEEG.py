# Author: Milton Candela (https://github.com/milkbacon)
# Date: Septembrer 2023

import pandas as pd
import numpy as np

# Set a DF for each subject at EEG_Frontiers folder (Considering Primera, Tercera y Cuarta)
# Only taken into account "pre" values (A)

children_encoding = {'Alejandro Contreras': 'DJ04', 'Arturo Sanchez': 'ST01', 'Emiliano Ruiz': 'DJ01',
                     'Ernesto Daniel': 'MT01', 'Evelyn Rosas': 'DJ03', 'Israel Torres': 'DT02',
                     'Jezael Montano': 'DT01', 'Joaquin Orrante': 'DJ02', 'Jorge Ortega': 'MJ01',
                     'Luca Rocha': 'EJ01', 'Marcelo Contreras': 'ES01', 'Mateo Rodriguez': 'ES02',
                     'Patricio Sadot': 'DT03', 'Sofia Galvan': 'MJ02', 'Dante Javier': 'ST02'}

MCE_categories = {1: 'Malo', 2: 'Insuficiente', 3: 'Regular', 4: 'Bueno', 5: 'Excelente'}
MCE_scores = {'DJ04':  [2.25, 2.5, 2.5], 'ST01': [2.5, 1.75, 2.5], 'DJ01': [2.75, 4.25, 4.5],
              'MT01': [3.75, 4, 4.5], 'DJ03': [3, 2.75, 4.75], 'DT02': [3.75, 3.5, 5],
              'DT01': [3, 3.5, 4.75], 'DJ02': [2.5, 2.25, 3.5], 'MJ01': [3, 3.5, 4],
              'EJ01': [np.nan, np.nan, np.nan], 'ES01': [3.5, 3, 4.5], 'ES02': [3, 3, 3.75],
              'DT03': [4, 3.75, 4.25], 'MJ02': [3, 3.75, 4]}

PSI_scores = {'DJ04': [0.6, 0, -0.3], 'ST01': [0.1, 0.6, -0.1], 'DJ01': [0.7, 0.2, -0.3],
              'MT01': [0.3, -0.3, -0.4], 'DJ03': [0, 0.3, -0.1], 'DT02': [0.8, 0.1, -0.2],
              'DT01': [-0.9, -0.3, -0.1], 'DJ02': [-0.1, -0.5, -0.6], 'MJ01': [0.6, 1.3, 0.4],
              'EJ01': [np.nan, np.nan, np.nan], 'ES01': [0.5, 0.4, 0.6], 'ES02': [-0.1, 0.5, -0.2],
              'DT03': [-0.1, 0.8, 0.1], 'MJ02': [0.5, 0.5, -0.3]}


def get_df(take, sub, letter):
    sr = 250
    eta = 4
    ft = 0
    det = 2

    df_2 = pd.DataFrame(columns=['C_' + str(x) for x in range(1, 9)])

    for j, channels in enumerate(['E4', 'E8']):
        df = pd.read_csv('data/EEG_Raw_Pruned_Used/{}Toma/{}-{}0{}-{}.csv'.format(take, sub, channels, takes_to_number[take], letter))
        rows_with_nan = list(df[df.isnull().any(axis=1)].index) + [df.shape[0]]

        for i in range(len(rows_with_nan) - 1):
            df_2['C_' + str(i + 1 + j*4)] = df.iloc[(rows_with_nan[i] + 1):(rows_with_nan[i + 1]), 1].reset_index(drop=True)

    df_2 = df_2.drop('C_8', axis=1)
    df_2.columns = electrode_names

    return df_2


def process_take(take, sub, letter):
    df = get_df(take, sub, letter)
    t = 1  # Removing the first minute
    df = df.iloc[(250 * 60 * t):, :]
    return df.reset_index(drop=True)


used_children = ['DJ01', 'DJ02', 'DJ03',
                 'DT01',
                 'ES01', 'ES02',
                 'MJ01',
                 'MT01',
                 'ST01']

# Niveles MCE #

# SJ: Starter Junior
# ST: Starter Staff     1
# SJ: Starter Senior

# EJ: Explorer Junior
# ET: Explorer Staff
# ES: Explorer Senior   2

# DJ: Deployer Junior   3
# DT: Deployer Staff    1
# DJ: Deployer Senior

# MJ: Master Junior     1
# MT: Master Staff      1
# MJ: Master Senior

# Tomando datos de antes del descanso
df_prepos = pd.read_csv('Frontiers/old/preposchildren2.csv')

takes_to_number = {'Primera': 1, 'Tercera': 3, 'Cuarta': 4}
takes_to_class = {'Primera': 'Programming', 'Tercera': 'Robotics', 'Cuarta': 'Design'}
electrode_names = ['FP1', 'F3', 'C3', 'PZ', 'C4', 'F4', 'FP2']

# Concatenate both lectures and having a dataset without the last EOG channel
for subject in used_children:
    print(subject)
    df_sub = pd.DataFrame(columns=electrode_names + ['Datetime', 'Take'])
    for take in takes_to_number.keys():
        try:
            letters = df_prepos[(df_prepos.Toma == takes_to_number[take]) & (df_prepos.ID == subject)].Pre.values[0]
            print('{}: {}'.format(take, letters))
        except IndexError:
            continue

        if letters == '-':
            continue
        if len(letters) == 1:
            df = process_take(take, subject, letters[0])
        else:
            df = pd.DataFrame()
            for letter in letters.split(', '):
                df_temp = process_take(take, subject, letters[0])
                df = pd.concat([df, df_temp], axis=0)
        df.to_csv('data/Frontiers/EEG_Txt/{}_{}.txt'.format(subject, takes_to_class[take][:4]), index=False, header=False)

    print()
