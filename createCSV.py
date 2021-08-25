# Author: Milton Candela (https://github.com/milkbacon)
# Date: July 2021

# The following code creates a CSV file from processed data, this data corresponds to data from three different
# biometrics, which are electroencephalogram (EEG), wristband with electrocardiogram (ECG) features, and a Computer
# Vision (CV) algorithm which extracts emotions of participants. The current code takes all the data and forms a
# unique DataFrame which has values with domain between 0 and 1, the data was extracted via multiple devices with
# different sampling rates. In order to establish a granularity the following considerations were taken:

# EEG and ECG (PPG): Median of each features on 5 minute windows.
# CV: Probability distribution for each possible emotion, on 5 minute windows.

import copy

from sklearn.preprocessing import StandardScaler
from pickle import dump
import pandas as pd
import numpy as np
import os

# Score data #
# This first section of code describe the target variables, which are the designated score a kig gets depending
# on its performance during the lecture (MCE_scores), while the second score is the change of STEM-CIS interest,
# obtained using a psychometric test before and after the lecture (PSI_scores).

# Instead of using the child's name, an specific ID is being used (for privacy purposes), however, a dictionary is
# included in case any information needs to be changed. Each ID has their respective score, and so it can be used as a
# target variable. It can be noticed that each ID has three floats, these correspond to the score for each type of
# lecture of type [Programming, Robotics, 3D Design]. In addition, MCE_score comes with an specific category
# depending on the range of values from the kid's performance, the category dictionary is MCE_categories.

children_encoding = {'Alejandro Contreras': 'DJ04', 'Arturo Sanchez': 'ST01', 'Emiliano Ruiz': 'DJ01',
                     'Ernesto Daniel': 'MT01', 'Evelyn Rosas': 'DJ03', 'Israel Torres': 'DT02',
                     'Jezael Montano': 'DT01', 'Joaquin Orrante': 'DJ02', 'Jorge Ortega': 'MJ01',
                     'Luca Rocha': 'EJ01', 'Marcelo Contreras': 'ES01', 'Mateo Rodriguez': 'ES02',
                     'Patricio Sadot': 'DT03', 'Sofia Galvan': 'MJ02'}

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

# Biometric data #
# This data can be found inside the "data" folder, with various sub-folders corresponding to each biometric device:
# (EEG, Emotions, Empatica). Inside each sub-folder, more sub-folders are found, which separate the data depending
# on the take. This take corresponds to the type of lecture it was taken by the kid, with the following encoding:

# Toma 1 (Take 1) : Programming
# Toma 2 (Take 2) : Robotics
# Toma 3 (Take 3) : Robotics
# Toma 4 (Take 4) : 3D Design

# As class imbalance is not a problem on regression problems, all takes would be used, although, when classification,
# either take 2 or 3 should be taken as they are both robotics lecture. Depending on the folder's organization is
# where the processed data is included, although the folders with the processed CSV files would be further used.


def get_df(device, folder):
    """
    The current function receives a device and a folder as a parameters, this determines which CSV files would be read
    in order to create a DataFrame based on all the CSV files on that sub-folder.

    :param string device: Biometric devices from which the data would be gathered.
    :param string folder: Sub-folder of the take, based on the biometric device selected.
    :return pd.DataFrame: A DataFrame with all the CSV observations found on the sub-folder.
    """

    # The CSV files are listed, and so it creates a list of files from which a blank dataframe is initialized.
    path = 'data/' + device + '/' + folder + '/'
    file_list = os.listdir(path)
    header = None if device == 'EEG - Engagement' else 0

    df_file = pd.read_csv(path + file_list[0], na_values='--', nrows=0, header=header)
    columns = list(df_file.columns) + ['ID', 'Take', 'Session']
    df_file = pd.DataFrame(columns=columns)

    # The following for loop iterates over all the CSV files, concatenating a temporal DataFrame with the main, blank
    # DataFrame. It is worth noting that the DataFrame keeps track of the CSV file which is read, such as: the kid's ID
    # kid; Take; and Session, this is important because the target variables would be joined using these data.
    for i, file in enumerate(file_list):
        df_file_temp = pd.read_csv(path + file_list[0], na_values='--', header=header)
        df_file_temp['ID'], df_file_temp['Take'], df_file_temp['Session'] = (file[0:4], file[8], file[10])
        df_file = pd.concat([df_file, df_file_temp], ignore_index=True)

    df_file.dropna(axis=1, how='any', inplace=True)
    df_file.reset_index(drop=True, inplace=True)
    return df_file


def get_df_takes(device, folders):
    """
    This function is a continuation based on the previously declared function, as the current function takes multiple
    folders and concatenates the data into a single DataFrame of multiple takes.

    :param string device: Biometric devices from which the data would be gathered.
    :param list folders: Takes sub-folders, based on the biometric device selected.
    :return pd.DataFrame: A final DataFrame with all the takes concatenated.
    """

    # Assuming that the user uses this function for more than two takes, the first element of the list "folders" is
    # popped and used to create the first DataFrame, on which the next DataFrames would be concatenated.
    df = get_df(device, folders.pop(0))
    for folder in folders:
        df_temp = get_df(device, folder)
        df = pd.concat([df, df_temp], ignore_index=True)

    # The following if-clause is specific on the EEG device, because the data on the CSV file is non-labeled, and so
    # their columns must be named based on the signals that they represent:
    # 7 channels for each spectral signal [Alpha, Low_Beta, High_Beta, Gamma, Theta] (35)
    # 1 value for each combined feature [Fatigue, Load, Engagement Index] (3)
    # With a total of 37 column names, further on the if statement, the number of rows on the EEG DataFrame is also
    # subset on the first n observations, where n corresponds to the number of rows in the Empatica DataFrame, this was
    # done due to failure and error in the biometric device, and so it would be under-sampled to the Empatica device.
    if device == 'EEG - Engagement' or device == 'EEG':
        combined_features = ['Fatigue', 'Load', 'Engagement'] if device == 'EEG - Engagement' else ['Fatigue', 'Load']
        channels = ['FP1', 'F3', 'C3', 'PZ', 'C4', 'F4', 'FP2']
        signals = ['Alpha', 'LowBeta', 'HighBeta', 'Gamma', 'Theta']
        eeg_columns = [signal + '_' + channel for signal in signals for channel in channels] + combined_features
        df.columns = eeg_columns + ['ID', 'Take', 'Session']
        df = df.iloc[range(dfPPG.shape[0]), :]

    # The final DataFrame is sorted based on the ID variables, which are our data indicators
    df = df.sort_values(by=['ID', 'Take', 'Session'])
    print(df.shape)
    df = df.dropna(axis=1, how='any').reset_index(drop=True)
    print(df.shape)
    return df


# Based on each biometric device sub-folder, a DataFrame is extracted using first, second, third and fourth take.
dfPPG = get_df_takes('Empatica', ['Resultados Primera Toma', 'Resultados Segunda Toma', 'Resultados Cuarta Toma'])
dfEEG = get_df_takes('EEG', ['Toma 1', 'Toma 2', 'Toma 4'])
dfCV = get_df_takes('Emotions', ['Resultados Primera Toma DLIB', 'Resultados Segunda Toma DLIB',
                                 'Resultados Cuarta Toma DLIB'])

def feature_generation(df):
    """
    This function creates a variety of combined features using the provided features, it is only used in EEG and PPG
    features, because they are non-normalized and will further be normalized, and so their range varies between 0 and 1.

    :param pd.DataFrame df: Non-normalized DataFrame with continuous values, from PPG and EEG.
    :return pd.DataFrame: Returns a pandas Dataframe with combined features in addition to the previous features.
    """

    # The following variables are created:
    # df_features would be our pandas DataFrame where the normal and combined features are placed.
    # Epsilon is a constant to avoid dividing by 0.
    # Names and combinations would track the names of the combined features created.
    df_features = copy.deepcopy(df)
    epsilon = 0.000001
    names = list(df_features.columns)
    combinations = []

    # The following for loop creates a set of combined features based on the spectral signals that were generated.
    # It iterates over all the features on a separate DataFrame, and it applies a function. The result is further
    # saved on a column with the following encoding:

    # Name_i-I : Inverse on ith feature
    # Name_i-L : Logarithm on ith feature
    # Name_i-M-Name_j : Multiplication of ith feature with feature jth
    # Name_i-D-Name_j : Division of ith feature with feature jth

    # A small number on the form of a epsilon is being used to avoid NANs because some functions are 0 sensitive,
    # such as the natural logarithm and the division by 0. Moreover, a separate list "combinations" is used to keep
    # track the combinations of ith and jth features, and so not to generate duplicate features when multiplying
    # ith feature with jth feature and vice versa (as they are the same number).
    for i in range(len(df.columns)):
        names.append(df.columns[i] + '-I')
        df_features = pd.concat((df_features, np.divide(np.ones(df.shape[0]), df.loc[:, df.columns[i]])),
                                axis=1, ignore_index=True)

        names.append(df.columns[i] + '-L')
        df_features = pd.concat((df_features, pd.Series(np.log(np.abs(np.array(df.loc[:, df.columns[i]])) + 1))),
                                axis=1, ignore_index=True)

        for j in range(len(df.columns)):
            if i != j:
                current_combination = str(i) + str(j)
                if current_combination not in combinations:
                    combinations.append(current_combination)
                    names.append(df.columns[i] + '-M-' + df.columns[j])
                    df_features = pd.concat((df_features,
                                             np.multiply(df.loc[:, df.columns[i]], df.loc[:, df.columns[j]])),
                                            axis=1, ignore_index=True)
                names.append(df.columns[i] + '-D-' + df.columns[j])
                df_features = pd.concat((df_features,
                                         pd.Series(np.divide(df.loc[:, df.columns[i]],
                                                             np.array(df.loc[:, df.columns[j]]) + epsilon))),
                                        axis=1, ignore_index=True)

    # The generated feature names are placed, infinity values from columns are removed, and the DF is returned.
    df_features.columns = names
    print(df_features.shape)
    df_features = df_features.replace([np.inf, -np.inf], np.nan).dropna(axis='columns', how='any')
    print(df_features.shape)
    return df_features


def combine_dfs(df_ppg, df_eeg, df_cv):
    """
    The current function combines the previously generated DataFrames, making their granularity the same, and setting
    them up for analysis using the same number of rows. EEG and PPG data is standardized, and then the sigmoid function
    is applied on the scaled values, this function was selected as its range is between 0 and 1, similar to the
    probability distribution obtained on each emotion on the CV DataFrame.

    :param pd.DataFrame df_ppg: PPG DataFrame from get_df_takes or get_df.
    :param pd.DataFrame df_eeg: EEG DataFrame from get_df_takes or get_df.
    :param pd.DataFrame df_cv: CV DataFrame from get_df_takes or get_df.
    :return pd.DataFrame: A final DataFrame with biometric features concatenated.
    """

    # As the PPG and EEG DataFrames would be scaled, the identification variables would be saved on another DataFrame.
    del df_ppg['Segment_Indices']
    df_names = df_ppg[['ID', 'Take', 'Session']]

    # EEG and PPG
    def df_to_prom(df, device):
        """
        This function receives either the EEG or PPG DataFrame, scales their values and applied the sigmoid function,
        and so the domain resembles the one from the CV algorithm's probability distribution. The standard scaler
        is saved on the "processed" folder, in case of needing to rescale the data to their original values.

        :param pd.DataFrame df: EEG or PPG, non-scaled, raw DataFrame.
        :param string device: Type of device, as identification for the saved scaler.
        :return pd.DataFrame: Scaled DataFrame, with the sigmoid function applied to each datum.
        """

        df.drop(['ID', 'Take', 'Session'], axis=1, inplace=True)
        df = feature_generation(df)
        columns = df.columns
        scaler = StandardScaler().fit(df)
        dump(scaler, open('processed/' + device + '_scaler.pkl', 'wb'))
        df = pd.DataFrame(scaler.transform(df), columns=columns).applymap(lambda x: 1 / (1 + np.exp(x)))
        return df
    df_ppg = df_to_prom(df_ppg, 'PPG')
    df_eeg = df_to_prom(df_eeg, 'EEG')

    # CV
    def prom(x):
        """
        The following functions takes a pandas series and computes the prevalence of each unique emotion across the
        series. The 'Pass' emotion is not an emotion, rather the NAN definition of the CV algorithm when it does
        not find a particular emotion on the given subject, and so it is first removed on the series. It is possible
        that this emotion is the only on given window, and so it return 0 on each emotion if that is the case.

        :param pd.Series x: "EmotionDetected" column from the df_cv DataFrame, which has emotions as strings.
        :return dictionary: Probability distribution depending on the prevalence of emotion on the pd.Series.
        """

        x = x[x != 'Pass']
        # 0 Neutral, 1 Surprised, 2 Sad, 3 Happy, 4 Fear, 6 Angry, 7 Pass
        emotions = {'neutral': 0, 'surprise': 0, 'sad': 0, 'happy': 0, 'fear': 0, 'angry': 0}
        if len(x) == 0:
            return emotions

        # When an emotion different that 0 is detected, the probability distribution for each class is computed, using
        # the Laplace transformation to give a slight probability to each class and avoid getting a lot of zeros.
        for emotion in emotions.keys():
            emotions[emotion] = (len(x[x == emotion]) + 1) / (len(x) + len(emotions.keys()))
        return emotions

    # Since the CV DataFrame is on a different granularity than the EEG and PPG DataFrames, it is cut into n bins,
    # where n corresponds to the number of rows in the PPG DataFrame, this ensures a probability distribution for each
    # row on the EEG and PPG DataFrame (already with the same number of rows)
    df_cv['Bin'] = pd.cut(x=df_cv.Second, bins=df_ppg.shape[0], labels=[x for x in range(df_ppg.shape[0])])
    proportions = pd.DataFrame(df_cv.groupby('Bin').
                               EmotionDetected.apply(func=prom)).unstack().reset_index(drop=True)['EmotionDetected']

    # All the transformated DataFames are concatenated on one, as their rows are equal, the information variables
    # (included on df_names) are also concatenated, NANs are removed and the index is reset.
    dfcomb = pd.concat([df_ppg, df_eeg, proportions, df_names], axis=1)
    dfcomb.dropna(axis=1, how='any', inplace=True)
    dfcomb.reset_index(drop=True, inplace=True)
    return dfcomb


# The session column, from the information variables, is currently not used, and so it is dropped.
comb_df = combine_dfs(dfPPG, dfEEG, dfCV).drop('Session', axis=1)


def set_scores(df, col_name, score_dict):
    """
    This function combines the biometric DataFrame (source variables) with the scores dictionaries (target variables),
    it does so by using the information columns [ID, Take] and relating them to the scores dictionaries.

    :param pd.DataFrame df: Scaled biometric DataFrame, with the same granularity across devices.
    :param string col_name: Name of the score column that would be assigned on the given DataFrame.
    :param dictionary score_dict: Dictionary of scores, with the kid ID as keys, and a list of scores on the format
     [Programming, Robotics, 3D Design] as values for each key.
    :return pd.DataFrame: Given DataFrame with a new, score column.
    """

    scores = []
    take_column_index, id_column_index = df.columns.get_loc('Take'), df.columns.get_loc('ID')

    # The next for loop iterates over each row of the final DataFrame, it first obtains the index value that would
    # be used on the dictionaries list. It is worth noting that [Programming, Robotics, 3D Design] have index [0, 1, 2],
    # takes have the following values: [1, 2, 4], and so a pattern could be traced to directly obtain the desired index.
    # It subtracts the take number if it corresponds to the 1 and 2 take, the 4 take is hard-coded into index 2,
    # afterwards, the ID variable is extracted and the score for that row is obtained.
    for i in range(df.shape[0]):
        current_take = int(df.iloc[i, take_column_index])
        take_key = current_take - 1 if current_take < 3 else 1 if current_take == 3 else 2
        current_id = str(df.iloc[i, id_column_index])
        if current_id in score_dict.keys():
            scores.append(score_dict[current_id][take_key])
        else:
            scores.append(np.nan)

    # The list of scores is appended as a column on the given DataFrame, using the given name.
    df[col_name] = scores
    return df


# The previous function is used on both scores, designating a separate column for each score.
comb_df = set_scores(comb_df, 'PSI_Score', score_dict=PSI_scores)
comb_df = set_scores(comb_df, 'MCE_Score', score_dict=MCE_scores)

# A category is assigned for each score, the MCE score has a range, multi-class category, while the PSI score is
# actually a delta of a score, and so a binary encoding would be applied: whether its value is positive or negative.
comb_df['MCE_Category'] = pd.cut(comb_df.MCE_Score, [0, 1, 2, 3, 4, 5], labels=MCE_categories.values())
comb_df['PSI_Category'] = ['Positivo' if score > 0 else 'Negativo' for score in comb_df.PSI_Score]

# The DataFrame is exported and saved into the "processed" folder, it is further printed for visualization purposes.
comb_df.to_csv('processed/combined_df_2.csv', index=False)
print(comb_df)
