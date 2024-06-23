import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import bessel, lfilter
from scipy.io import wavfile


def plot_bessel(y, data, fs, freq=-1):
    """
    Plots original and filtered signal
    @param y: (1D array) filtered signal
    @param data: (1D array) original signal
    @param fs: sample rate
    @param freq: filtering frequency added to plot title
    """

    t = np.arange(len(data)) / fs

    plt.figure(figsize=(10, 6))
    plt.plot(data, label='Sygnał oryginalny', alpha=0.7)
    plt.plot(y, label='Sygnał przefiltrowany', alpha=0.7)
    plt.title(f'Filtr Bessela {freq}Hz')
    plt.xlabel('Czas [próbki]')
    plt.ylabel('Amplituda')
    plt.ylim(-32767, 32767)
    plt.legend()
    plt.grid()

    plt.show()
    plt.close()


def single_channel_bessel(data, order, cutoff, fs):
    """
    Filtering single channel with Bessel filter
    @param data: (1D array) data from single channel
    @param order: filter order
    @param cutoff: cutoff frequency
    @param fs: sampling rate
    @return:
    """

    Wn = cutoff / (fs / 2)

    b, a = bessel(order, Wn, btype='low', analog=False, norm='phase')

    y = lfilter(b, a, data)
    # y = filtfilt(b, a, data)
    return y


def bessel_filter(inputfilepath, order=6, cutoff=335, output_filename='bessel.wav', do_plot=True, do_save_file=True):
    """
    Filtering signal with Bessel filter
    @param inputfilepath: path to raw .wav file
    @param  order: order of Bessel filter - default is 6
    @param cutoff: cutoff frequency - default is 335Hz
    @param output_filename: path to save filtered .wav file - default is 'bessel.wav'
    @param do_plot: if True, plot original and filtered signal - default is True
    @param do_save_file: if True, save filtered signal to output_filename - default is True
    """

    input_filename = inputfilepath

    fs, data = wavfile.read(input_filename)

    if len(data.shape) > 1:
        channel1 = data[:, 0]
        channel2 = data[:, 1]

        channel1 = channel1 / np.max(np.abs(channel1))
        channel1 = channel1 * 32767
        channel2 = channel2 / np.max(np.abs(channel2))
        channel2 = channel2 * 32767

        filtred_ch1 = single_channel_bessel(channel1, order, cutoff, fs)
        if do_plot: plot_bessel(filtred_ch1, channel1, fs, cutoff)
        filtred_ch2 = single_channel_bessel(channel2, order, cutoff, fs)
        if do_plot: plot_bessel(filtred_ch2, channel2, fs, cutoff)
        y = np.vstack((filtred_ch1, filtred_ch2)).T
    else:
        data = data / np.max(np.abs(data))
        data = data * 32767

        y = single_channel_bessel(data, order, cutoff, fs)
        if do_plot: plot_bessel(y, data, fs, cutoff)

    filtered_data = y.astype(np.int16)

    if do_save_file:
        print(f"Saved {output_filename}")
        wavfile.write(output_filename, fs, filtered_data)





if __name__ == "__main__":
    file = '../data/ctirec/toa/drone2_mics.wav'

    bessel_filter(file, order=6, cutoff=335, output_filename="saw_bessel.wav")

