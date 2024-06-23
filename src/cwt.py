import numpy as np
import matplotlib.pyplot as plt
import pywt
import scipy.io.wavfile as wav
import os


def plotCWT(filepath, outdir="", tofile=0, start=0, end=0):
    """
    Function to plot Continuous Wavelet Transform of a given audio file
    @param filepath: path to file
    @param outdir: directory to save plot
    @param tofile: save plot to file - default is 0, set differently to save
    @param start: start sample of the audio file - default is 0
    @param end: end sample of the audio file - default is 0 which means end of the file
    """
    filename = os.path.basename(filepath)
    fs, data = wav.read(filepath)

    if len(data.shape) > 1:
        data = data[:, 0]

    data = data[start:]
    if end != 0:
        data = data[start:end]

    # normalization to [-1, 1]
    data = data / np.max(np.abs(data))

    # wavelet = 'gaus1'
    wavelet = 'cmor1.5-1.0'
    # wavelet = 'morl'
    scales = np.arange(1, 128)
    freqs = np.arange(1, 20)

    # calculating CWT
    coefficients, frequencies = pywt.cwt(data, freqs, wavelet, sampling_period=1/fs)

    plt.figure(figsize=(10, 5))
    #extent=(0, len(data), frequencies[-1], frequencies[0]),
    plt.imshow(np.abs(coefficients), extent=(0, len(data), 0, 20), aspect='auto', cmap='jet')
    plt.title(f'(CWT) Ciągłe przekształcenie falkowe {filename}')
    plt.ylabel('Częstotliwość [kHz]')
    plt.xlabel('Czas [próbki]')
    plt.colorbar(label='Natężenie')
    if tofile != 0:
        plt.savefig(f'{outdir}/{filename}_cwt.png', dpi=300)
    else:
        plt.show()
    plt.close()





if __name__ == '__main__':
    drone_file1 = '../data/DroneAudioDataset-master/Binary_Drone_Audio/yes_drone/B_S2_D1_067-bebop_000_.wav'
    drone_file2 = '../data/Membo_0_039-membo_004_.wav'
    file = '../data/ctirec/phonerec/wav/short/Drone2_10sec.wav'

    # plotCWT(drone_file1, 'outputs')
    # plotCWT(drone_file2, 'outputs')
    plotCWT(file)


