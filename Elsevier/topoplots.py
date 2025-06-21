
import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

channels = ['C3', 'C4', 'F3', 'F4', 'Fp1', 'Fp2', 'Pz']
df_all = pd.read_csv('data/Frontiers/lectureBandpower.csv')
bands = ['Alpha', 'Beta', 'Theta']

minmax = (df_all.iloc[:, 1:].min().min(), df_all.iloc[:, 1:].max().max())
fig, axes = plt.subplots(3, 3, dpi=1000)

for j, band in enumerate(bands):
    df = pd.concat([df_all.iloc[j, 1:8].reset_index(drop=True), df_all.iloc[j, 8:15].reset_index(drop=True), df_all.iloc[j, 15:23].reset_index(drop=True)], axis=1)
    df.columns = ['3D Design', 'Programming', 'Robotics']
    df.index = channels

    montage = mne.channels.make_standard_montage("biosemi256")
    n_channels = len(montage.ch_names)
    data = df.reindex(montage.ch_names)

    info = mne.create_info(ch_names=montage.ch_names, sfreq=250, ch_types='eeg')
    evoked = mne.EvokedArray(np.array(data), info)
    evoked.set_montage(montage)

    for i in range(df.shape[1]):
        mne.viz.plot_topomap(evoked.data[:, i], evoked.info, cmap='viridis', axes=axes[j][i], show=False, vlim=minmax)
        if j == 0:
            axes[j][i].set_title(list(data.columns)[i])

cbar = fig.colorbar(axes[0][0].images[0], ax=axes, orientation='vertical', shrink=0.8, pad=0.05)
    # cbar.set_label('Power')

    # plt.tight_layout()
    # plt.show()

plt.savefig('Power_Topoplot.png', dpi=1000, bbox_inches='tight')
