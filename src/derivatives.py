import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import find_peaks
import matplotlib.lines as mlines
import os


# Minimal distance between peaks
PEAK_DST = 10


def moving_average(data, window_size):
    """
    Calculates moving average of the data. It helps to smooth the derivative.
    @param data: data from single channel
    @param window_size: size of the window
    @return: 1D array - smoothed data
    """
    return np.convolve(data, np.ones(window_size), 'valid') / window_size


def calculate_1derivative(inputfile):
    """
    Calculates 1st derivative of the signal.
    @param inputfile: path to the file
    @return: 1st derivative of the signal (1D array) - if stereo, returns 2 channels (2x1D arrays)
    """
    fs, data = wavfile.read(inputfile)
    dt = 1 / fs
    if len(data.shape) == 2:
        channel1 = data[:, 0]
        channel2 = data[:, 1]

        d_chan1 = np.gradient(channel1, dt)
        d_chan2 = np.gradient(channel2, dt)

        return d_chan1, d_chan2

    elif len(data.shape) == 1:
        d_data = np.gradient(data, dt)
        return d_data


def calculate_2derivative(ch1, ch2, fs):
    """
    Calculates 2nd derivative of the signal.
    @param ch1: (1D array) first channel
    @param ch2: (1D array) second channel
    @param fs: sample rate of the signal
    @return: 2x1D arrays - 2nd derivative of the 1st channel and 2nd channel
    """
    dt = 1 / fs
    dchan1 = np.gradient(ch1, dt)
    dchan2 = np.gradient(ch2, dt)
    dchan1 = moving_average(dchan1, 5)
    dchan2 = moving_average(dchan2, 5)

    return dchan1, dchan2


def plot_derivatives_single_ch(data, d1, d2, start=0, end=2000, text="", ori_alpha=0.8, d1_alpha=0.6, d2_alpha=0.5,
                               set_to_zero=False, maxims=False, minims=False, which_extremes=0):
    """
    Plots original signal from single channel and its derivatives on single plot.
    @param data: (1D array) original signal
    @param d1: (1D array) 1st derivative of the signal
    @param d2: (1D array) 2nd derivative of the signal
    @param start: start index of the array (sample number) - default is 0
    @param end: end index of the array (sample number) - default is 2000
    @param text: title of the plot
    @param ori_alpha: transparency of the original signal on the plot - default is 0.8
    @param d1_alpha: transparency of the 1st derivative on the plot - default is 0.6
    @param d2_alpha: transparency of the 2nd derivative on the plot - default is 0.5
    @param set_to_zero: if True, plots the extreme points on zero - default is False
    @param maxims: if True, plots the maximum points - default is False
    @param minims: if True, plots the minimum points - default is False
    @param which_extremes: choose which extremes to plot - 0 - original signal, 1 - 1st derivative, 2 - 2nd derivative
    """

    data = data[start:end]
    data = data / np.max(np.abs(data))
    d1 = d1[start:end]
    d1 = d1 / np.max(np.abs(d1))
    d2 = d2[start:end]
    d2 = d2 / np.max(np.abs(d2))

    extremes_data = data
    if which_extremes == 1: extremes_data = d1
    if which_extremes == 2: extremes_data = d2

    data_max, _ = find_peaks(extremes_data, distance=PEAK_DST)
    # data_max = data_max[extremes_data[data_max] > 0]

    data_min, _ = find_peaks(-extremes_data, distance=PEAK_DST)
    # data_min = data_min[extremes_data[data_min] < 0]

    plt.figure(figsize=(10, 6))
    if ori_alpha != 0:
        plt.plot(data, label="Sygnal", alpha=ori_alpha)

    if set_to_zero:
        if maxims:
            plt.plot(data_max, np.zeros(len(data_max)), 'o', color="red", alpha=0.5)
        if minims:
            plt.plot(data_min, np.zeros(len(data_min)), 'o', color="red", alpha=0.5)
    else:
        if maxims:
            plt.plot(data_max, extremes_data[data_max], 'o', color="red", alpha=0.5)
        if minims:
            plt.plot(data_min, extremes_data[data_min], 'o', color="red", alpha=0.5)

    if d1_alpha != 0:
        plt.plot(d1, label="1wsza Pochodna", alpha=d1_alpha)

    if d2_alpha != 0:
        plt.plot(d2, label="2ga Pochodna", alpha=d2_alpha)

    legend = plt.legend(bbox_to_anchor=(-0.1, 1.05), loc='lower left', ncol=6)
    legend.get_frame().set_alpha(0)
    plt.xlabel('Czas [próbki]')
    plt.ylabel('Amplituda')
    plt.title(f'{text}')
    plt.show()
    plt.close()


def plot_extreme_points(ch1, ch2, d_ch1, d_ch2, d2_ch1, d2_ch2, start=0, end=2000, text="", set_to_zero=False,
                        maxims=True, mins=True,
                        which_extremes=0, ori_alpha=0.0):
    """
    Plots original signal from both channels and its extreme points
    @param ch1: original signal from 1st channel
    @param ch2: original signal from 2nd channel
    @param d_ch1: 1st derivative of the 1st channel
    @param d_ch2: 1st derivative of the 2nd channel
    @param d2_ch1: 2nd derivative of the 1st channel
    @param d2_ch2: 2nd derivative of the 2nd channel
    @param start: start index of the array (sample number) - default is 0
    @param end: end index of the array (sample number) - default is 2000
    @param text: title of the plot
    @param set_to_zero: if True, plots the extreme points on zero - default is False
    @param maxims: if True, plots the maximum points - default is True
    @param mins: if True, plots the minimum points - default is True
    @param which_extremes: choose which extremes to plot: 0 - original signal, 1 - 1st derivative, 2 - 2nd derivative
    @param ori_alpha: transparency of the original signal on the plot - default is 0.0
    """
    ch1 = ch1[start:end]
    ch1 = ch1 / np.max(np.abs(ch1))
    ch2 = ch2[start:end]
    ch2 = ch2 / np.max(np.abs(ch2))

    d_ch1 = d_ch1[start:end]
    d_ch1 = d_ch1 / np.max(np.abs(d_ch1))
    d_ch2 = d_ch2[start:end]
    d_ch2 = d_ch2 / np.max(np.abs(d_ch2))

    d2_ch1 = d2_ch1[start:end]
    d2_ch1 = d2_ch1 / np.max(np.abs(d2_ch1))
    d2_ch2 = d2_ch2[start:end]
    d2_ch2 = d2_ch2 / np.max(np.abs(d2_ch2))

    extremes_data_ch1 = ch1
    extremes_data_ch2 = ch2
    if which_extremes == 1:
        extremes_data_ch1 = d_ch1
        extremes_data_ch2 = d_ch2
    if which_extremes == 2:
        extremes_data_ch1 = d2_ch1
        extremes_data_ch2 = d2_ch2

    ch1_max, _ = find_peaks(extremes_data_ch1, distance=PEAK_DST)
    ch2_max, _ = find_peaks(extremes_data_ch2, distance=PEAK_DST)
    ch1_min, _ = find_peaks(-extremes_data_ch1, distance=PEAK_DST)
    ch2_min, _ = find_peaks(-extremes_data_ch2, distance=PEAK_DST)

    plt.figure(figsize=(10, 6))
    if ori_alpha != 0:
        plt.plot(ch1, alpha=ori_alpha)
        plt.plot(ch2, alpha=ori_alpha)

    if set_to_zero:
        if maxims:
            plt.plot(ch1_max, np.zeros(len(ch1_max)), 'o', color="blue", alpha=0.8)
            plt.plot(ch2_max, np.zeros(len(ch2_max)), 'o', color="orange", alpha=0.6)
        if mins:
            plt.plot(ch1_min, np.zeros(len(ch1_min)), 'o', color="blue", alpha=0.8)
            plt.plot(ch2_min, np.zeros(len(ch2_min)), 'o', color="orange", alpha=0.6)
    else:
        if maxims:
            plt.plot(ch1_max, extremes_data_ch1[ch1_max], 'o', color="blue", alpha=0.8)
            plt.plot(ch2_max, extremes_data_ch2[ch2_max], 'o', color="orange", alpha=0.6)
        if mins:
            plt.plot(ch1_min, extremes_data_ch1[ch1_min], 'o', color="blue", alpha=0.8)
            plt.plot(ch2_min, extremes_data_ch2[ch2_min], 'o', color="orange", alpha=0.6)

    blue_line = mlines.Line2D([], [], color='blue', label='Kanał 1')
    orange_line = mlines.Line2D([], [], color='orange', label='Kanał 2')
    plt.legend(handles=[blue_line, orange_line])
    plt.xlabel('Czas [próbki]')
    plt.ylabel('Amplituda')
    plt.title(f'Ekstrema - oba kanały - {text}')
    plt.show()
    plt.close()


def choose_close_extremes(file, start=0, end=-1, threshold=20, do_plot=False, xlim_start=10000, xlim_end=12000,
                                set_to_zero=False, text="", sample_rate=44100):
    """
    Chooses the closest extreme points from both channels.
    @param file: path to the .wav file
    @param start: start index of the array (sample number) - default is 0
    @param end: end index of the array (sample number) - default is -1 - length of the signal
    @param threshold: maximum distance between the extreme points - default is 20
    @param do_plot: if True, plots the signal with extreme points - default is False
    @param xlim_start: start index of the x-axis on the plot - default is 10000
    @param xlim_end: end index of the x-axis on the plot - default is 12000
    @param set_to_zero: if True, plots the extreme points on zero - default is False
    @param text: title of the plot
    @param sample_rate: sample rate of the signal, used for calculating time - default is 44100
    @return: 2x1D array - y-values of the extreme points from both channels
    """
    sr, data = wavfile.read(file)
    channel1 = data[start:end, 0]
    channel2 = data[start:end, 1]

    ch1_max, _ = find_peaks(channel1, distance=PEAK_DST)
    ch2_max, _ = find_peaks(channel2, distance=PEAK_DST)
    ch1_min, _ = find_peaks(-channel1, distance=PEAK_DST)
    ch2_min, _ = find_peaks(-channel2, distance=PEAK_DST)

    ch1_peaks = sorted(np.concatenate((ch1_max, ch1_min)))
    ch2_peaks = sorted(np.concatenate((ch2_max, ch2_min)))

    filtered_ch1_peaks = []
    filtered_ch2_peaks = []

    if len(ch1_peaks) > len(ch2_peaks):
        for p in ch1_peaks:
            closest_peak = min(ch2_peaks, key=lambda x: abs(x - p))
            if abs(p - closest_peak) < threshold:
                filtered_ch1_peaks.append(p)
                filtered_ch2_peaks.append(closest_peak)
    else:
        for p in ch2_peaks:
            closest_peak = min(ch1_peaks, key=lambda x: abs(x - p))
            if abs(p - closest_peak) < threshold:
                filtered_ch2_peaks.append(p)
                filtered_ch1_peaks.append(closest_peak)


    filtered_ch1_peaks = np.array(filtered_ch1_peaks)
    filtered_ch2_peaks = np.array(filtered_ch2_peaks)
    phase_shift = filtered_ch2_peaks - filtered_ch1_peaks
    avg_phase_shift = np.mean(phase_shift)
    print(f"Avg phase shift: {avg_phase_shift}")

    channel1 = np.array(channel1)
    channel2 = np.array(channel2)
    valid_ch1 = filtered_ch1_peaks[filtered_ch1_peaks < len(channel1)]
    valid_ch2 = filtered_ch2_peaks[filtered_ch2_peaks < len(channel2)]
    y_ch1 = channel1[valid_ch1]
    y_ch2 = channel2[valid_ch2]

    filename = os.path.basename(file)
    if do_plot:
        ex_ch1 = y_ch1
        ex_ch2 = y_ch2
        if set_to_zero:
            ex_ch1 = np.zeros(len(y_ch1))
            ex_ch2 = np.zeros(len(y_ch2))
        plt.figure(figsize=(10, 6))
        plt.plot(channel1, alpha=0.4, color="blue")
        plt.plot(channel2, alpha=0.4, color="orange")
        plt.plot(valid_ch1, ex_ch1, 'o', color="blue", alpha=0.8)
        plt.plot(valid_ch2, ex_ch2, 'o', color="orange", alpha=0.8)
        plt.xlim(xlim_start, xlim_end)
        plt.title(f'{filename} {text}')
        plt.show()
        plt.close()

    return y_ch1, y_ch2


def choose_close_extremes_pairs(file, start=0, end=-1, threshold=20, do_plot=False, xlim_start=10000, xlim_end=12000,
                                set_to_zero=False, text="", sample_rate=44100):
    """
    Chooses the closest extreme points from both channels.
    It always chooses the pair - every extreme point from channel 1 has its pair in channel 2.
    @param file: path to the .wav file
    @param start: start index of the array (sample number) - default is 0
    @param end: end index of the array (sample number) - default is -1 - length of the signal
    @param threshold: maximum distance between the extreme points - default is 20
    @param do_plot: if True, plots the signal with extreme points - default is False
    @param xlim_start: start index of the x-axis on the plot - default is 10000
    @param xlim_end: end index of the x-axis on the plot - default is 12000
    @param set_to_zero: if True, plots the extreme points on zero - default is False
    @param text: title of the plot
    @param sample_rate: sample rate of the signal, used for calculating time - default is 44100
    @return: 2x1D array - y-values of the extreme points from both channels
    """
    sr, data = wavfile.read(file)
    channel1 = data[start:end, 0]
    channel2 = data[start:end, 1]

    ch1_max, _ = find_peaks(channel1, distance=PEAK_DST)
    ch2_max, _ = find_peaks(channel2, distance=PEAK_DST)
    ch1_min, _ = find_peaks(-channel1, distance=PEAK_DST)
    ch2_min, _ = find_peaks(-channel2, distance=PEAK_DST)

    ch1_peaks = sorted(np.concatenate((ch1_max, ch1_min)))
    ch2_peaks = sorted(np.concatenate((ch2_max, ch2_min)))
    # print(f"{ch1_peaks}, {ch2_peaks}")

    filtered_ch1_peaks = []
    filtered_ch2_peaks = []

    i, j = 0, 0
    while i < len(ch1_peaks) and j < len(ch2_peaks):
        if abs(ch1_peaks[i] - ch2_peaks[j]) < threshold:
            filtered_ch1_peaks.append(ch1_peaks[i])
            filtered_ch2_peaks.append(ch2_peaks[j])
            i += 1
            j += 1
        elif ch1_peaks[i] < ch2_peaks[j]:
            i += 1
        else:
            j += 1

    # print(f"{filtered_ch1_peaks}, {filtered_ch2_peaks}")
    filtered_ch1_peaks = np.array(filtered_ch1_peaks)
    filtered_ch2_peaks = np.array(filtered_ch2_peaks)
    phase_shift = filtered_ch2_peaks - filtered_ch1_peaks
    avg_phase_shift = np.mean(phase_shift)
    time = avg_phase_shift / sample_rate
    print(f"Avg phase shift: {avg_phase_shift}, time: {time}s")

    channel1 = np.array(channel1)
    channel2 = np.array(channel2)
    valid_ch1 = filtered_ch1_peaks[filtered_ch1_peaks < len(channel1)]
    valid_ch2 = filtered_ch2_peaks[filtered_ch2_peaks < len(channel2)]
    y_ch1 = channel1[valid_ch1]
    y_ch2 = channel2[valid_ch2]
    # print(f"Valid peaks: {len(y_ch1)}, {len(y_ch2)}")

    filename = os.path.basename(file)
    if do_plot:
        ex_ch1 = y_ch1
        ex_ch2 = y_ch2
        if set_to_zero:
            ex_ch1 = np.zeros(len(y_ch1))
            ex_ch2 = np.zeros(len(y_ch2))
        plt.figure(figsize=(10, 6))
        plt.plot(channel1, alpha=0.4, color="blue")
        plt.plot(channel2, alpha=0.4, color="orange")
        plt.plot(valid_ch1, ex_ch1, 'o', color="blue", alpha=0.8)
        plt.plot(valid_ch2, ex_ch2, 'o', color="orange", alpha=0.8)
        plt.xlim(xlim_start, xlim_end)
        plt.title(f'{filename} {text}')
        plt.show()
        plt.close()

    return y_ch1, y_ch2



if __name__ == "__main__":
    file1 = "../outputs/bessel.wav"
    file2 = "../outputs/iir.wav"
    file_sin = "../outputs/sin_bessel.wav"
    file_saw = "../outputs/saw_bessel.wav"
    file_drone = "../outputs/drone_bessel.wav"

    ofile1 = '../data/ctirec/toa/drone2_mics.wav'
    ofile2 = "../outputs/sin.wav"
    ofile3 = "../outputs/saw.wav"

    pilawlesie270 = "../outputs/bessel_alldst/270hz/5m_1.wav"

    bessel5m1 = "../outputs/bessel_alldst/270hz/5m_1.wav"
    bessel5m2 = "../outputs/bessel_alldst/270hz/5m_2.wav"
    bessel5m3 = "../outputs/bessel_alldst/270hz/5m_3.wav"
    bessel5m4 = "../outputs/bessel_alldst/270hz/5m_4.wav"
    bessel5m5 = "../outputs/bessel_alldst/270hz/5m_5.wav"
    ori5m1 = "../data/pilawlesie/alldst_split/5m_1.wav"
    ori5m2 = "../data/pilawlesie/alldst_split/5m_2.wav"
    ori5m3 = "../data/pilawlesie/alldst_split/5m_3.wav"
    ori5m4 = "../data/pilawlesie/alldst_split/5m_4.wav"
    ori5m5 = "../data/pilawlesie/alldst_split/5m_5.wav"

    bessel25m5 = "../outputs/filt/bessel_alldst/270hz/5m_5.wav"


    choose_close_extremes_pairs(bessel5m1)
