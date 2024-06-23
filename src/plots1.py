import cwt
import fft
import rawaudio
import stft
import os


# drone = '../data/drone1.wav'
# rawaudio.plotRAW(drone)
# fft.plotFFT(drone)
# stft.plotSTFT(drone)
# cwt.plotCWT(drone)


def gitdatabse():
    directory_path = "inputs"

    file_names = os.listdir(directory_path)

    file_list = [os.path.join(directory_path, file) for file in file_names]

    for file in file_list:
        rawaudio.plotRAW(file, "outputs")
        fft.plotFFT(file, "outputs")
        stft.plotSTFT(file, "outputs")
        cwt.plotCWT(file, "outputs")


def recs1():
    file = '../data/ctirec/phonerec/wav/short/Drone2_10sec.wav'
    outdir = 'outputs_cti'
    rawaudio.plotRAW(file, outdir)
    fft.plotFFT(file, outdir)
    stft.plotSTFT(file, outdir)
    cwt.plotCWT(file, outdir)


def recs2(param=0):
    # MIKROFONY
    outdir = './outputs_cti/mics'
    # czysty dron
    f1 = '../data/ctirec/DRON_SND/drone2cuts/long/dronemix1.wav'
    rawaudio.plotRAW(f1, outdir, param)
    fft.plotFFT(f1, outdir, param)
    stft.plotSTFT(f1, outdir, param)
    cwt.plotCWT(f1, outdir, param)

    # dron i bg noise
    f2 = '../data/ctirec/DRON_SND/drone3cuts/drone3_drone_bgnoise.wav'
    rawaudio.plotRAW(f2, outdir, param)
    fft.plotFFT(f2, outdir, param)
    stft.plotSTFT(f2, outdir, param)
    cwt.plotCWT(f2, outdir, param)

    # dron i sinus 335Hz
    f3 = '../data/ctirec/DRON_SND/drone4cuts/dron4sin_drone.wav'
    rawaudio.plotRAW(f3, outdir, param)
    fft.plotFFT(f3, outdir, param)
    stft.plotSTFT(f3, outdir, param)
    cwt.plotCWT(f3, outdir, param)

    # dron i saw 335Hz
    f4 = '../data/ctirec/DRON_SND/drone5cuts/dron5_saw_drone.wav'
    rawaudio.plotRAW(f4, outdir, param)
    fft.plotFFT(f4, outdir, param)
    stft.plotSTFT(f4, outdir, param)
    cwt.plotCWT(f4, outdir, param)

    outdir = './outputs_cti/phone'
    # czysty dron
    f1 = '../data/ctirec/phonerec/wav/short/Drone2stop.wav'
    rawaudio.plotRAW(f1, outdir, param)
    fft.plotFFT(f1, outdir, param)
    stft.plotSTFT(f1, outdir, param)
    cwt.plotCWT(f1, outdir, param)

    # dron i bg noise
    f2 = '../data/ctirec/phonerec/wav/short/Drone3stop.wav'
    rawaudio.plotRAW(f2, outdir, param)
    fft.plotFFT(f2, outdir, param)
    stft.plotSTFT(f2, outdir, param)
    cwt.plotCWT(f2, outdir, param)

    # dron i sinus 335Hz
    f3 = '../data/ctirec/phonerec/wav/short/Drone4stop.wav'
    rawaudio.plotRAW(f3, outdir, param)
    fft.plotFFT(f3, outdir, param)
    stft.plotSTFT(f3, outdir, param)
    cwt.plotCWT(f3, outdir, param)

    # dron i saw 335Hz
    f4 = '../data/ctirec/phonerec/wav/short/Drone5stop.wav'
    rawaudio.plotRAW(f4, outdir, param)
    fft.plotFFT(f4, outdir, param)
    stft.plotSTFT(f4, outdir, param)
    cwt.plotCWT(f4, outdir, param)


if __name__ == '__main__':
    recs1()
    # outdir = './outputs_cti/phone'
    # f4 = '../data/ctirec/phonerec/wav/short/Drone5stop.wav'
    # # rawaudio.plotRAW(f4, outdir)
    # # fft.plotFFT(f4, outdir)
    # stft.plotSTFT(f4, outdir)
    # cwt.plotCWT(f4, outdir)