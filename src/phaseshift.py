import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft as fft2
import os


def phase_shift(filepath, start=0, end=44100, text=""):
    """
    Plots phase shift between two channels
    @param filepath: path to .wavee file
    @param start: start of the signal - default is 0
    @param end: end of the signal - default is 44100
    @param text: title of the plot
    """
    sample_rate, data = wavfile.read(filepath)

    channel_1 = data[start:end, 0]
    channel_2 = data[start:end, 1]

    channel_1 = channel_1 / np.max(np.abs(channel_1))
    channel_2 = channel_2 / np.max(np.abs(channel_2))

    plt.figure(figsize=(10, 6))
    plt.plot(channel_1, label='Kanał 1', alpha=1.0)
    plt.plot(channel_2, label='Kanał 2', alpha=0.5)
    plt.ylim(-1.1, 1.1)
    plt.legend()
    plt.xlabel('Czas [próbki]')
    plt.ylabel('Amplituda')
    plt.title(f'Przesuniecie fazowe {text}')
    plt.show()


def mc_phase_shift(channel1, channel2, start=0, end=44100, text=""):
    """
    Plots phase shift between two channels
    @param channel1: 1D array of the first channel
    @param channel2: 1D array of the second channel
    @param start: start of the signal (sample number) - default is 0
    @param end: end of the signal (sample number) - default is 44100
    @param text: title of the plot
    """
    channel1 = channel1[start:end]
    channel2 = channel2[start:end]

    channel_1 = channel1 / np.max(np.abs(channel1))
    channel_2 = channel2 / np.max(np.abs(channel2))

    plt.figure(figsize=(10, 6))
    plt.plot(channel_1, label='Kanał 1', alpha=1.0)
    plt.plot(channel_2, label='Kanał 2', alpha=0.5)
    plt.ylim(-1.1, 1.1)
    plt.legend()
    plt.xlabel('Czas [próbki]')
    plt.ylabel('Amplituda')
    plt.title(f'Przesuniecie fazowe {text}')
    plt.show()


def calculate_ps(frequency, sample_rate=44100, mics_dist=0.25, xdist=5, zdist=0):
    """
    Calculates phase shift of the wave for given parameters
    @param frequency: frequency of the wave
    @param sample_rate: sample rate of the signal (for calculating shift in samples)
    @param mics_dist: distance between microphones
    @param xdist: distance straight ahead between the source and the point between microphones - default is 5
    @param zdist: distance to the side between the source and the point between microphones - default is 0
    @return:
    """
    print("calculate phaseshift")
    speed_of_sound = 340.0  # m/s

    sample_time = 1 / sample_rate

    d1 = (mics_dist / 2) + xdist
    dist = np.sqrt((d1 * d1) + (zdist * zdist))
    rad_ps = (2 * np.pi * frequency * dist) / speed_of_sound
    sec_ps = rad_ps / (2 * np.pi * frequency)
    sample_ps = sec_ps / sample_time
    deg_ps = np.rad2deg(rad_ps)

    r1 = np.sqrt((xdist * xdist) + (zdist * zdist))
    d = mics_dist + xdist
    r2 = np.sqrt((d * d) + (zdist * zdist))
    delta_r = r2 - r1
    wave_len = speed_of_sound / frequency

    analytical_ps_rad = (2 * np.pi * delta_r) / wave_len
    analytical_ps_deg = np.rad2deg(analytical_ps_rad)
    analytical_ps_sec = analytical_ps_rad / (2 * np.pi * frequency)
    analytical_ps_samples = analytical_ps_sec / sample_time
    print(
        f'difference - samples: {analytical_ps_samples}, time: {analytical_ps_sec}s')
        # f',  rad: {analytical_ps_rad}, deg: {analytical_ps_deg}')
    print(f"straight line - samples: {sample_ps}, time: {sec_ps}")


def shift_cc_file(file, start=0, end=-1, plot_cc=False, cc_range=50, zoom=False):
    """
    Calculates correlation between two channels and finds the phase shift. Uses file as an input.
    @param file: path to the .wav file
    @param start: start of the signal (sample number) - default is 0
    @param end: end of the signal (sample number) - default is -1 - whole signal
    @param plot_cc: if True, plots the correlation - default is False
    @param cc_range: Parameter to zoom the cross-correlation plot - default is 50
    @param zoom: if True, plots second correlation plot - default is False
    """

    sample_rate, data = wavfile.read(file)
    channel1 = data[start:end, 0]
    channel2 = data[start:end, 1]
    filename = os.path.basename(file)

    cc = np.correlate(channel1 - np.mean(channel1), channel2 - np.mean(channel2), mode='full')
    shift = np.argmax(cc) - (len(channel1) - 1)
    if plot_cc:
        norm_cc = cc / np.max(np.abs(cc))
        lags = np.arange(-len(channel1) + 1, len(channel1))

        plt.figure(figsize=(10, 6))
        plt.plot(lags, norm_cc)
        plt.ylim(-1.1, 1.1)
        plt.xlabel("Przesunięcie - próbki")
        plt.ylabel("Współczynniki korelacji")
        plt.title(filename)
        plt.show()

        if zoom:
            plt.figure(figsize=(10, 6))
            plt.plot(lags, norm_cc)
            plt.scatter(shift, norm_cc[shift + len(channel1) - 1], color="red")
            plt.xlim(-cc_range, cc_range)
            plt.ylim(-1.1, 1.1)
            plt.xlabel("Przesunięcie - próbki")
            plt.ylabel("Współczynniki korelacji")
            plt.title(filename)
            plt.show()

    # print(cc)
    # print(shift)
    delay_in_seconds = shift / sample_rate
    d_m = delay_in_seconds * 340.0
    print(f"samples: {shift}, time: {delay_in_seconds}s, distance: {d_m}m")


def shift_cc_channels(channel1, channel2, sample_rate=44100, start=0, end=-1, plot_cc=False, cc_range=50, zoom=False, text=""):
    """
    Calculates correlation between two channels and finds the phase shift. Uses 1D arrays as an input.

    @param channel1: (1D array) first channel
    @param channel2: (1D array) second channel
    @param sample_rate: sample rate of the signal - default is 44100
    @param start: start of the signal (sample number) - default is 0
    @param end: end of the signal (sample number) - default is -1 - whole signal
    @param plot_cc: if True, plots the correlation - default is False
    @param cc_range: Parameter to zoom the cross-correlation plot - default is 50
    @param zoom: if True, plots second correlation plot - default is False
    @param text: title of the plot
    """

    channel1 = channel1[start:end]
    channel2 = channel2[start:end]

    cc = np.correlate(channel1 - np.mean(channel1), channel2 - np.mean(channel2), mode='full')
    shift = np.argmax(cc) - (len(channel1) - 1)
    if plot_cc:
        norm_cc = cc / np.max(np.abs(cc))
        lags = np.arange(-len(channel1) + 1, len(channel1))

        plt.figure(figsize=(10, 6))
        plt.plot(lags, norm_cc)
        plt.ylim(-1.1, 1.1)
        plt.xlabel("Przesunięcie - próbki")
        plt.ylabel("Współczynniki korelacji")
        plt.title(text)
        plt.show()

        if zoom:
            plt.figure(figsize=(10, 6))
            plt.plot(lags, norm_cc)
            plt.scatter(shift, norm_cc[shift + len(channel1) - 1], color="red")
            plt.xlim(-cc_range, cc_range)
            plt.ylim(-1.1, 1.1)
            plt.xlabel("Przesunięcie - próbki")
            plt.ylabel("Współczynniki korelacji")
            plt.title(text)
            plt.show()

    # print(cc)
    # print(shift)
    delay_in_seconds = shift / sample_rate
    d_m = delay_in_seconds * 340.0
    print(f"Correlation shift: samples: {shift}, time: {delay_in_seconds}s, distance: {d_m}m")



def shift_peaks(file, start=0, end=44100, sr=44100):
    """
    Calculates phase shift between two channels using wave peaks
    @param file: path to the .wav file
    @param start: start of the signal (sample number) - default is 0
    @param end: end of the signal (sample number) - default is 44100
    @param sr: sample rate of the signal - default is 44100
    """
    sample_rate, data = wavfile.read(file)
    channel1 = data[start:end, 0]
    channel2 = data[start:end, 1]
    ch1 = channel1
    ch1 = ch1 / np.max(np.abs(ch1))
    ch2 = channel2
    ch2 = ch2 / np.max(np.abs(ch2))
    channel2 = ch2
    channel1 = ch1

    dist = 20

    ch1max, _ = find_peaks(channel1, distance=dist)
    ch1min, _ = find_peaks(-channel1, distance=dist)
    ch2max, _ = find_peaks(channel2, distance=dist)
    ch2min, _ = find_peaks(-channel2, distance=dist)

    ch1peaks = np.concatenate((ch1min, ch1max))
    ch1peaks.sort()

    ch2peaks = np.concatenate((ch2min, ch2max))
    ch2peaks.sort()

    if len(ch1peaks) < len(ch2peaks):
        num_of_points = len(ch1peaks)
    else:
        num_of_points = len(ch2peaks)

    avg = 0
    for i in range(0, num_of_points):
        avg += (ch2peaks[i] - ch1peaks[i])
    avg = avg / num_of_points
    t = avg / sr
    dst_f = t * 340.0
    print(f"samples: {avg}, time: {t}s, distance: {dst_f}m")


def shift_xcross(d_ch1, d_ch2, start=0, end=44100):
    """
    Calculates phase shift between two channels using zero-crossing of the signal
    @param d_ch1: (1D array) first derivative of the first channel
    @param d_ch2: (1D array) first derivative of the second channel
    # @param file: path to the .wav file
    @param start: start of the signal (sample number) - default is 0
    @param end: end of the signal (sample number) - default is 44100
    """

    # sample_rate, data = wavfile.read(file)
    # channel1 = data[start:end, 0]
    # channel2 = data[start:end, 1]

    # d_ch1, d_ch2 = calculate_1derivative(file)
    d_ch1 = d_ch1[start:end]
    d_ch2 = d_ch2[start:end]
    dst = 20
    dch1max, _ = find_peaks(d_ch1, distance=dst)
    dch1min, _ = find_peaks(-d_ch1, distance=dst)
    dch2max, _ = find_peaks(d_ch2, distance=dst)
    dch2min, _ = find_peaks(-d_ch2, distance=dst)

    ch1ex = np.concatenate((dch1min, dch1max))
    ch2ex = np.concatenate((dch2min, dch2max))
    ch1ex.sort()
    ch2ex.sort()
    ch1xcrlen = len(dch1max) + len(dch1min)
    ch2xcrlen = len(dch2max) + len(dch2min)

    if ch1xcrlen < ch2xcrlen:
        num_of_samples = ch1xcrlen
    else:
        num_of_samples = ch2xcrlen

    avg = 0
    for i in range(0, num_of_samples):
        avg += (ch2ex[i] - ch1ex[i])
    avg = avg / num_of_samples
    print(avg)


def calculate_phase_shift_for_frequency(filepath, frequency, start=0, end=44100):
    """
    Calculates phase shift of the wave for given frequency
    @param filepath: path to the .wav file
    @param frequency: frequency of the wave
    @param start: start of the signal (sample number) - default is 0
    @param end: end of the signal (sample number) - default is 44100
    @return: phase shift in samples
    """

    sample_rate, data = wavfile.read(filepath)

    channel_1 = data[start:end, 0]
    channel_2 = data[start:end, 1]

    channel_1 = channel_1 / np.max(np.abs(channel_1))
    channel_2 = channel_2 / np.max(np.abs(channel_2))

    fft_ch1 = fft2(channel_1)
    fft_ch2 = fft2(channel_2)

    freq_index = int(frequency * len(channel_1) / sample_rate)

    phase_shift = np.angle(fft_ch2[freq_index]) - np.angle(fft_ch1[freq_index])
    print(phase_shift)
    return phase_shift


def calculate_ps_triangle(sample_rate=44100, mics_dist=0.25, xdist=5, zdist=5):
    """
    Calculates phase shift of the wave for given parameters. Uses triangle method.
    @param sample_rate: sample rate of the signal (for calculating shift in samples)
    @param mics_dist: distance between microphones
    @param xdist: distance straight ahead between the source and the point between microphones
    @param zdist: distance to the side between the source and the point between microphones
    """
    if xdist == 0:
        print("calculate_ps_triangle: difference: dst 0m, time: 0s, samples: 0")
        return
    speed_of_sound = 340.0
    mic1_x = xdist - (mics_dist / 2)
    mic2_x = xdist + (mics_dist / 2)
    mic1_len = np.sqrt((mic1_x * mic1_x) + (zdist * zdist))
    mic2_len = np.sqrt((mic2_x * mic2_x) + (zdist * zdist))

    delta_d = abs(mic2_len - mic1_len)
    delta_t = delta_d / speed_of_sound
    samples_diff = sample_rate * delta_t

    # x = np.sqrt((mics_dist * mics_dist) - (delta_d * delta_d))
    # p2 = zdist / x

    p1 = xdist / delta_d
    real_dist = p1 * mics_dist

    time = real_dist / speed_of_sound
    samples = sample_rate * time
    print(f"calculate_ps_triangle:\n"
          f"difference: dst {delta_d}m, time: {delta_t}s, samples: {samples_diff}\n"
          f"straight line: dst {real_dist}m, time: {time}s, samples: {samples}")




if __name__ == "__main__":

    bessel5m1 = "bessel_alldst/270hz/5m_1.wav"
    bessel5m2 = "bessel_alldst/270hz/5m_2.wav"
    bessel5m3 = "bessel_alldst/270hz/5m_3.wav"
    bessel5m4 = "bessel_alldst/270hz/5m_4.wav"
    bessel5m5 = "bessel_alldst/270hz/5m_5.wav"
    ori5m1 = "../data/pilawlesie/alldst_split/5m_1.wav"
    ori5m2 = "../data/pilawlesie/alldst_split/5m_2.wav"
    ori5m3 = "../data/pilawlesie/alldst_split/5m_3.wav"
    ori5m4 = "../data/pilawlesie/alldst_split/5m_4.wav"
    ori5m5 = "../data/pilawlesie/alldst_split/5m_5.wav"

    file = ori5m5
    dst = 4

