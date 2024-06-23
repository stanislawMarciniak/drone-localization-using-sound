import os.path
import numpy as np
import matplotlib.pylab as plt
import librosa
import librosa.display


def plotMEL(file, outdir="..", tofile=False):
    """
    Plot Mel Spectogram
    @param file: path to the file
    @param outdir: path to the output directory
    @param tofile: is True, save the plot to the file - default is False
    """
    filename = os.path.basename(file)
    y, sr = librosa.load(file)
    S = librosa.feature.melspectrogram(y=y,
                                       sr=sr,
                                       n_mels=128 * 4, )
    S_db_mel = librosa.amplitude_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 5))
    # Plot the mel spectogram
    img = librosa.display.specshow(S_db_mel,
                                   x_axis='time',
                                   y_axis='log',
                                   ax=ax)
    ax.set_title(f'Mel Spektrogram: {filename}', fontsize=15)
    fig.colorbar(img, ax=ax, format=f'%0.2f', label='Natężenie')
    plt.xlabel("Czas [s]")
    plt.ylabel("Częstotliwość [Hz]")
    if tofile:
        plt.savefig(f'{outdir}/{filename}_mel.png', dpi=300)
    else:
        plt.show()
    plt.close()


if __name__ == '__main__':
    file = "../data/pilawlesie/cont_split/5m.wav"
    plotMEL(file)
