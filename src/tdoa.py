import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


def tdoa_plot(x, y, text):
    """
    Plots TDOA (Time Difference of Arrival) between two channels in time
    @param x: (1D array) time in seconds
    @param y: (1D array) TDOA data in seconds
    @param text: title of the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    for sec in x:
        plt.axvline(x=sec, color='grey', linestyle='--', alpha=0.3)
    plt.title(text)
    plt.xlabel("Czas [s]")
    plt.ylabel("TDOA [s]")
    plt.grid(True)
    plt.show()


def dst_plot(x, y, ref_y, text, do_plot_ref=False):
    """
    Plots distance differences of the arrived signals in time
    @param x: (1D array) time in seconds
    @param y: (1D array) TDOA data in meters
    @param ref_y: (1D array) reference data
    @param text: title of the plot
    @param do_plot_ref: if True, plots reference data - default is False
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    if do_plot_ref: plot_ref(ref_y)
    for sec in x:
        plt.axvline(x=sec, color='grey', linestyle='--', alpha=0.3)
    plt.title(text)
    plt.xlabel("Time [s]")
    plt.ylabel("Distance [m]")
    plt.xlim(np.min(x), np.max(x))
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_ref(y):
    """
    Plots reference data on TDOA plot
    @param y: (1D array) reference data
    """

    x = list(range(1, len(y)+1))
    plt.plot(x, y, label="Referencja", linestyle="--", color="orange")



def split_list(input_list, size):
    return [input_list[i:i + size] for i in range(0, len(input_list), size)]


def cc_diff(channel1, channel2, fs, dst_between_mics):
    """
    Calculates phase shift between two channels using correlation
    @param channel1: (1D array) first channel
    @param channel2: (1D array) second channel
    @param fs: sample rate in Hz
    @param dst_between_mics: distance between microphones in meters
    @return: distance in meters, delay in seconds
    """
    cross_corelation = np.correlate(channel1 - np.mean(channel1), channel2 - np.mean(channel2), mode='full')
    delay = np.argmax(cross_corelation) - (len(channel1) - 1)

    delay_sec = delay / fs

    sound_speed = 340.0
    distance = sound_speed * delay_sec


    max_time_diff = dst_between_mics / sound_speed
    if abs(delay_sec) > max_time_diff:
        print("Calculated delay is too big")
    return distance, delay_sec



def tdoa_mics(file, ref_y, text="", dst_between_mics=0.25, interval=44100):
    """
    Calculates TDOA (Time Difference of Arrival) between two channels
    @param file: path to .wav file
    @param ref_y: reference data - pass [] if not used
    @param text: title of the plot
    @param dst_between_mics: distance between microphones in meters - default is 0.25
    @param interval: interval of correlation calculation in samples - default is 44100
    """

    audio, fs = sf.read(file)

    channel1 = audio[:, 0]
    channel2 = audio[:, 1]

    channel1sec = split_list(channel1, interval)
    channel2sec = split_list(channel2, interval)


    toa_diffs = []
    dst_diffs = []

    for i in range(0, len(channel1sec)):
        print(f"{i + 1} / {len(channel1sec)}")
        distance, time = cc_diff(channel1sec[i], channel2sec[i], fs, dst_between_mics)
        toa_diffs.append(time)
        dst_diffs.append(distance)


    print(f"Time: {toa_diffs}")
    print(f"Distance: {dst_diffs}")
    tdoa_plot(list(range(1, len(toa_diffs) + 1)), toa_diffs, text)
    do_plot_ref = False
    if len(ref_y) > 0: do_plot_ref = True
    dst_plot(list(range(1, len(dst_diffs) + 1)), dst_diffs, ref_y, text, do_plot_ref=do_plot_ref)




if __name__ == '__main__':

    file1 = '../data/ctirec/fl_dron2_30sec.wav'
    file2 = '../data/ctirec/DRON_SND/drone2cuts/long/dron2_30sec.wav'
    file3 = '../data/ctirec/toa/drone2_mics.wav'
    file4 = '../data/ctirec/toa/drone2_phone.wav'
    file5 = '../outputs/bessel.wav'
    file6 = "../data/pilawlesie/cont_split/5m.wav"
    pilawlesie270 = "../outputs/bessel_cont/270hz/5m.wav"

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

    tdoa_mics(ori5m5, [])

    # plot_ref()
