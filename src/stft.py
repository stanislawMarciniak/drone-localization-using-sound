import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


def plotSTFT(filepath, outdir="..", tofile=False, text="", channel_no=1, start=0, end=-1, frame_size=2048, hop_size=512):
    """
    Plots Short Time Fourier Transform of the audio file
    @param filepath: path to the .wav file
    @param outdir: directory to save the plot - default is ".."
    @param tofile: if True, saves the plot to a file - default is False
    @param text: title of the plot
    @param channel_no: if 1, plots the first channel, otherwise the second channel - default is 1
    @param start: start of the signal (sample number) - default is 0
    @param end: end of the signal (sample number) - default is length of the signal
    @param frame_size: size of the frame in samples - default is 2048
    @param hop_size: size of the hop in samples - default is 512
    """
    filename = os.path.basename(filepath)
    y, sr = librosa.load(filepath, sr=None)
    x = int(len(y) / 2)
    if channel_no == 1:
        y = y[:x]
    else:
        y = y[x:]

    y = y[start:end]

    s_scale = librosa.stft(y, n_fft=frame_size, hop_length=hop_size)

    Y_scale = np.abs(s_scale) ** 2

    Y_logscale = librosa.amplitude_to_db(Y_scale)

    plt.figure(figsize=(10, 5))
    librosa.display.specshow(Y_logscale,
                             sr=sr,
                             hop_length=hop_size,
                             x_axis="time",
                             y_axis="log",
                             cmap='inferno')
    plt.colorbar(format="%+2.f", label="Natężenie [dB]")
    plt.title(f'(STFT) Spectrogram {filename} {text}')
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    if tofile:
        plt.savefig(f'{outdir}/{filename}_stft.png', dpi=300)
    else:
        plt.show()
    plt.close()


if __name__ == '__main__':

    drone_file1 = '../data/DroneAudioDataset-master/Binary_Drone_Audio/yes_drone/B_S2_D1_067-bebop_000_.wav'
    drone_file2 = '../data/Membo_0_039-membo_004_.wav'
    f = '../outputs/sin.wav'
    # plotSTFT(file)
    plotSTFT(f)
