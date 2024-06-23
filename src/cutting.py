import soundfile as sf
import os


def split_list(input_list, size):
    return [input_list[i:i + size] for i in range(0, len(input_list), size)]


def cut_audio(filepath, outdir, targetfs):
    """
    Cuts the audio file into short segments
    @param filepath: path to the stereo .wav file
    @param outdir: directory to save the segments
    @param targetfs: length of the segment in samples
    @return:
    """
    audio, fs = sf.read(filepath)

    audio = audio[:len(audio) - (len(audio) % targetfs)]


    ch1 = audio[:, 0]
    ch2 = audio[:, 1]

    ch1seg = split_list(ch1, targetfs)
    ch2seg = split_list(ch2, targetfs)


    if not os.path.exists(outdir):
        os.makedirs(outdir)


    for i, segment in enumerate(ch1seg):
        sf.write(f'{outdir}/ch1seg_{i + 1}.wav', segment, targetfs)

    for i, segment in enumerate(ch2seg):
        sf.write(f'{outdir}/ch2seg_{i + 1}.wav', segment, targetfs)
