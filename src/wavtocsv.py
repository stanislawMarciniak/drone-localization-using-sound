import pandas as pd
import soundfile as sf
import csv


def savetocsv(filepath, numofseconds, filename):
    """
    Saves the audio data to a csv file
    @param filepath: path to .wav file
    @param numofseconds: length of the audio in seconds to save
    @param filename: name of the csv file
    """
    audio, fs = sf.read(filepath)

    desiredLen = numofseconds * fs
    channel1 = audio[:desiredLen, 0]
    channel2 = audio[:desiredLen, 1]
    print(len(channel1))
    print(len(channel2))

    df = pd.DataFrame({
        'channel1': channel1,
        'channel2': channel2
    })

    df.to_csv(filename, index=False, header=False)


def readfromcsv(filename):
    """
    Reads the audio data from a csv file
    @param filename: name of the csv file
    @return: (1D array) channel1, (1D array) channel2
    """
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        channel1 = []
        channel2 = []
        for row in reader:
            channel1.append(float(row[0]))
            channel2.append(float(row[1]))
    return channel1, channel2


if __name__ == '__main__':
    file = '../data/pilawlesie/alldst_split/5m_1.wav'
