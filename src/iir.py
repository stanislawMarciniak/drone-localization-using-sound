import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt


def plot_iir(data, filtered_data, freq=-1):
    plt.figure(figsize=(10, 6))
    plt.plot(data, label='Oryginalny sygnał', alpha=0.7)
    plt.plot(filtered_data, label='Sygnał po filtracji', alpha=0.7)
    plt.legend()
    plt.xlabel('Czas [próbki]')
    plt.ylabel('Amplituda')
    plt.title(f'Filtr Butterwortha ({freq}Hz)')
    plt.ylim(-32767, 32767)
    plt.show()


def butter_lowpass(cutoff, fs, order):
    """
    Butterworth low-pass filter
    @param cutoff: cutoff frequency
    @param fs: sampling rate
    @param order: order of the filter
    @return: coefficients of the filter
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order):
    """
    Filtering signal with Butterworth filter
    @param data: 1D array of the signal (one channel)
    @param cutoff: cutoff frequency
    @param fs: sampling rate
    @param order: order of the filter
    @return: 1D array of the filtered signal
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def iir_filter(infilepath, outfilepath, cutoff_frequency=1000, order=6):
    """
    Filtering signal with Butterworth filter and saving to a new file
    @param infilepath: path to the raw .wav file
    @param outfilepath: path to save filtered .wav file
    @param cutoff_frequency: cutoff frequency - default is 1000Hz
    @param order: order of the filter - default is 6
    """
    sample_rate, data = wavfile.read(infilepath)

    if len(data.shape) > 1:
        channel1 = data[:, 0]
        channel2 = data[:, 1]

        channel1 = channel1 / np.max(np.abs(channel1))
        channel1 = channel1 * 32767
        channel2 = channel2 / np.max(np.abs(channel2))
        channel2 = channel2 * 32767

        filtered_channel1 = butter_lowpass_filter(channel1, cutoff_frequency, sample_rate, order)
        plot_iir(channel1, filtered_channel1, cutoff_frequency)
        filtered_channel2 = butter_lowpass_filter(channel2, cutoff_frequency, sample_rate, order)
        plot_iir(channel2, filtered_channel2, cutoff_frequency)

        filtered_data = np.vstack((filtered_channel1, filtered_channel2)).T
    else:
        data = data / np.max(np.abs(data))
        data = data * 32767
        filtered_data = butter_lowpass_filter(data, cutoff_frequency, sample_rate, order)
        plot_iir(data, filtered_data, cutoff_frequency)

    filtered_data_int = filtered_data.astype(np.int16)
    wavfile.write(outfilepath, sample_rate, filtered_data_int)


if __name__ == "__main__":
    file1 = '../data/ctirec/toa/drone2_mics.wav'
    cutoff = 335

    iir_filter(file1, "iir.wav", order=6, cutoff_frequency=cutoff)
