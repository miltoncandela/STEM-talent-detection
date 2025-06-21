import pandas as pd
import os


used_children = ['DJ01', 'DJ02', 'DJ03',
                 'DT01',
                 'ES01', 'ES02',
                 'MJ01',
                 'MT01',
                 'ST01']

electrode_names = ['FP1', 'F3', 'C3', 'PZ', 'C4', 'F4', 'FP2']

def get_df(take, sub):
    df_2 = pd.DataFrame(columns=['C_' + str(x) for x in range(1, 9)])

    for j, channels in enumerate(['4', '8']):
        df = pd.read_csv('data/EEG_Calib/{}/{}-{}.csv'.format(take, sub, channels))
        rows_with_nan = list(df[df.isnull().any(axis=1)].index) + [df.shape[0]]
        for i in range(len(rows_with_nan) - 1):
            df_2['C_' + str(i + 1 + j*4)] = df.iloc[(rows_with_nan[i] + 1):(rows_with_nan[i + 1]), 1].reset_index(drop=True)

    df_2 = df_2.drop('C_8', axis=1)
    df_2.columns = electrode_names

    return df_2

def process_take(take, sub):
    df = get_df(take, sub)
    # t = 5  # Removing first and last 5 seconds
    # df = df.iloc[(250 * 5 * t):(df.shape[0] - 250 * 5 * t), :]
    return df.reset_index(drop=True)


takes_to_class = dict(zip([x + 'Toma' for x in ['Primera', 'Segunda', 'Tercera', 'Cuarta']], ['Programming', 'Robotics', 'Robotics', 'Design']))

for take in [x + 'Toma' for x in ['Primera', 'Segunda', 'Tercera', 'Cuarta']]:
    for subject in list(set([x[:4] for x in os.listdir('data/EEG_Calib/' + take)])):
        df = process_take(take, subject)
        df.to_csv('data/Frontiers/EEG_Calib/{}_{}.txt'.format(subject, takes_to_class[take][:4]), index=False, header=False)