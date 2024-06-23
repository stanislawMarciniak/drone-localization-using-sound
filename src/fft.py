import os.path
import librosa
import numpy as np
import matplotlib.pyplot as plt


def plotFFT(filepath, outdir="", tofile=False, text="", channel_no=1):
    """
    Plots Fast Fourier Transform of the audio file
    @param filepath: path to the file
    @param outdir: path to the output directory
    @param tofile: if True, saves the plot to a file - default is False
    @param text: title of the plot
    @param channel_no: 1 - first channel otherwise second channel - default is 1
    """
    filename = os.path.basename(filepath)
    y, sr = librosa.load(filepath, sr=None)
    x = int(len(y) / 2)
    if channel_no == 1:
        y = y[:x]
    else:
        y = y[x:]


    # n_fft = sr
    n_fft = len(y)
    ft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=n_fft + 1))
    ft_mean = np.mean(ft, axis=1)
    frequencies = np.fft.rfftfreq(n_fft, d=1 / sr)

    ft_mean = ft_mean / np.max(ft_mean)

    plt.figure(figsize=(10, 5))
    plt.plot(frequencies, ft_mean)
    plt.title(f'(FFT) Widmo sygnału {filename} {text}')
    plt.xlabel('Częstotliwość [Hz]')
    plt.ylabel('Amplituda')
    if tofile:
        plt.savefig(f'{outdir}/{filename}_fft.png', dpi=300)
    else:
        plt.show()
    plt.close()


if __name__ == '__main__':
    drone_file1 = '..\data\DroneAudioDataset-master\Binary_Drone_Audio\yes_drone\B_S2_D1_099-bebop_003_.wav'
    drone_file2 = '../data/Membo_0_039-membo_004_.wav'
    file = '../data/untitled.wav'
    f = '../data/pilawlesie/cont_split/5m.wav'
    plotFFT(f)
