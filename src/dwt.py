import pywt
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


def plotDWT(file):
    """
    Plots Discrete Wavelet Transform of the audio file
    @param file: path to the file
    """
    sample_rate, audio_data = wavfile.read(file)
    audio_data = audio_data[:, 0]
    audio_data = audio_data/max(audio_data)
    t = np.arange(len(audio_data)) / sample_rate
    print(sample_rate)

    cA, cD = pywt.dwt(audio_data, 'bior6.8', 'per')
    y = pywt.idwt(cA, cD, 'bior6.8', 'per')

    L = len(audio_data)
    y = y[0:L]

    plt.figure(figsize=(10, 6))

    plt.subplot(3, 1, 1)
    plt.plot(t, audio_data, color='k')
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda")
    plt.title("Oryginalny dzwiek")

    plt.subplot(3, 1, 2)
    plt.plot(cA, color='r')
    plt.xlabel("Probki")
    plt.ylabel("cA")

    plt.subplot(3,1,3)
    plt.plot(cD, color='g')
    plt.xlabel("Probki")
    plt.ylabel("cD")

    plt.show()
    plt.close()