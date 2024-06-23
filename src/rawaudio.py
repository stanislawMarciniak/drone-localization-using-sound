import pandas as pd
import matplotlib.pylab as plt
import os
import librosa
import librosa.display



def plotRAW(filepath, outdir="..", tofile=False, channel_no=1, start=0, end=-1):
    """
    Plots raw audio signal
    @param filepath: path to the file
    @param outdir: path to save the plot
    @param tofile: if True, saves the plot to a file - default is False
    @param channel_no: if 1, plots the first channel, otherwise the second channel - default is 1
    @param start: start of the signal (sample number) - default is 0
    @param end: end of the signal (sample number) - default is length of the signal
    """
    filename = os.path.basename(filepath)
    y, sr = librosa.load(filepath, sr=None)
    x = int(len(y) / 2)
    if channel_no == 1:
        y = y[:x]
    else:
        y = y[x:]

    y = y[start:end]

    plt.figure(figsize=(10, 5))
    pd.Series(y).plot(figsize=(10, 5),
                      lw=1,
                      title=f'Raw audio {filename}',
                      xlabel='Samples',)

    if tofile:
        plt.savefig(f'{outdir}/{filename}_raw.png', dpi=300)
    else:
        plt.show()
    plt.close()


if __name__ == '__main__':
    drone_file1 = '..\data\DroneAudioDataset-master\Binary_Drone_Audio\yes_drone\B_S2_D1_099-bebop_003_.wav'
    drone_file2 = '../data/Membo_0_039-membo_004_.wav'
    file = "../outputs/sin.wav"

    plotRAW(file, end=200)
