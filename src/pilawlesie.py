import src.tdoa as tdoa

if __name__ == "__main__":
    # 18 sekund
    file_5m_ori = "../data/pilawlesie/cont_split/5m.wav"
    file_5m_270hz = "../outputs/bessel_cont/270hz/5m.wav"
    file_5m_310hz = "../outputs/bessel_cont/310hz/5m.wav"
    file_5m_350hz = "../outputs/bessel_cont/350hz/5m.wav"

    # 20 sekund
    file_7m_ori = "../data/pilawlesie/cont_split/7m.wav"
    file_7m_270hz = "../outputs/bessel_cont/270hz/7m.wav"
    file_7m_310hz = "../outputs/bessel_cont/310hz/7m.wav"
    file_7m_350hz = "../outputs/bessel_cont/350hz/7m.wav"

    # 17 sekund
    file_9m_ori = "../data/pilawlesie/cont_split/9m.wav"
    file_9m_270hz = "../outputs/bessel_cont/270hz/9m.wav"
    file_9m_310hz = "../outputs/bessel_cont/310hz/9m.wav"
    file_9m_350hz = "../outputs/bessel_cont/350hz/9m.wav"

    # 18 sekund
    file_11m_ori = "../data/pilawlesie/cont_split/11m.wav"
    file_11m_270hz = "../outputs/bessel_cont/270hz/11m.wav"
    file_11m_310hz = "../outputs/bessel_cont/310hz/11m.wav"
    file_11m_350hz = "../outputs/bessel_cont/350hz/11m.wav"

    # 22 sekundy
    file_13m_ori = "../data/pilawlesie/cont_split/13m.wav"
    file_13m_270hz = "../outputs/bessel_cont/270hz/13m.wav"
    file_13m_310hz = "../outputs/bessel_cont/310hz/13m.wav"
    file_13m_350hz = "../outputs/bessel_cont/350hz/13m.wav"

    # 23 sekundy
    file_15m_ori = "../data/pilawlesie/cont_split/15m.wav"
    file_15m_270hz = "../outputs/bessel_cont/270hz/15m.wav"
    file_15m_310hz = "../outputs/bessel_cont/310hz/15m.wav"
    file_15m_350hz = "../outputs/bessel_cont/350hz/15m.wav"


    # 38 sekund
    kwadrat = "../data/pilawlesie/kwadrat.wav"
    # iir.iir_filter(file_5m_ori, "irr_5m_270.wav", cutoff_frequency=270)
    # bessel.bessel_filter(file_5m_ori, cutoff=1000, output_filename="bessel_cont/5m_1000.wav")
    # bessel.bessel_filter(kwadrat, cutoff=350, output_filename="kwadrat.wav")
    file_5m_1000hz = "../outputs/bessel_cont/5m_1000.wav"
    iir_5m_270 = "irr_5m_270.wav"
    kwadrat_350hz = "kwadrat.wav"
    # 32 sekduny
    join_file = "../data/pilawlesie/alldst_split/alldst_join.wav"
    join_350hz = "bessel_alldst/350hz/alldst_join.wav"
    join_270hz = "bessel_alldst/270hz/alldst_join.wav"
    # stft.plotSTFT(iir_5m_270)
    # fft.plotFFT(iir_5m_270)
    # tdoa.tdoa_mics(iir_5m_270, [], 18)

    jf_ref = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]
    # tdoa.tdoa_mics(joint_file, [], 32)

    # tdoa2.task1(joint_file, jf_ref, 32)
    # tdoa2.task1(file_5m_ori, jf_ref, 32)

    # 5m
    # bessel.bessel_filter(file_5m_ori, order=6, cutoff=270, do_save_file=False)
    # stft.plotSTFT(file_5m_ori)
    # fft.plotFFT(file_5m_ori)
    # stft.plotSTFT(file_5m_270hz, text="Bessel 270Hz")
    # fft.plotFFT(file_5m_270hz, text="Bessel 270Hz")
    # stft.plotSTFT(file_5m_350hz, text="Bessel 350Hz")
    # fft.plotFFT(file_5m_350hz, text="Bessel 350Hz")
    # # deri.task2(file_5m_ori, end=200)
    # deri.task2(file_5m_270hz)
    # deri.task2(file_5m_350hz)



    # 7m
    # bessel.bessel_filter(file_7m_ori, order=6, cutoff=270, do_save_file=False)
    # stft.plotSTFT(file_7m_ori)
    # fft.plotFFT(file_7m_ori)
    # stft.plotSTFT(file_7m_270hz, text="Bessel 270Hz")
    # fft.plotFFT(file_7m_270hz, text="Bessel 270Hz")
    # stft.plotSTFT(file_7m_350hz, text="Bessel 350Hz")
    # fft.plotFFT(file_7m_350hz, text="Bessel 350Hz")
    # # deri.task2(file_7m_ori, end=200)
    # deri.task2(file_7m_270hz)
    # deri.task2(file_7m_350hz)

    # 15m
    # bessel.bessel_filter(file_15m_ori, order=6, cutoff=270, do_save_file=False)
    # stft.plotSTFT(file_15m_ori)
    # fft.plotFFT(file_15m_ori)
    # stft.plotSTFT(file_15m_270hz, text="Bessel 270Hz")
    # fft.plotFFT(file_15m_270hz, text="Bessel 270Hz")
    # stft.plotSTFT(file_15m_350hz, text="Bessel 350Hz")
    # fft.plotFFT(file_15m_350hz, text="Bessel 350Hz")
    # # deri.task2(file_5m_ori, end=200)
    # deri.task2(file_15m_270hz)
    # deri.task2(file_15m_350hz)

    # tdoa.tdoa_mics(file_5m_270hz, [])
    # tdoa.tdoa_mics(file_5m_350hz, [])
    tdoa.tdoa_mics(file_5m_ori, [])

    # tdoa.tdoa_mics(file_7m_270hz, [], text="7m 270Hz")
    # tdoa.tdoa_mics(file_7m_ori, [], text="7m oryginał")
    # tdoa.tdoa_mics(file_9m_270hz, [], text="9m 270Hz")
    # tdoa.tdoa_mics(file_9m_ori, [], text="9m oryginał")
    # tdoa.tdoa_mics(file_11m_270hz, [], text="11m 270Hz")
    # tdoa.tdoa_mics(file_11m_ori, [], text="11m oryginał")
    # tdoa.tdoa_mics(file_13m_270hz, [], text="13m 270Hz")
    # tdoa.tdoa_mics(file_13m_ori, [], text="13m oryginał")
    # tdoa.tdoa_mics(file_15m_270hz, [], text="15m 270Hz")
    # tdoa.tdoa_mics(file_15m_ori, [], text="15m oryginał")
    #
    # tdoa.tdoa_mics(joint_file, [], text="próbki")
    #
    # tdoa.tdoa_mics(kwadrat, [], text="kwadrat")

    # tdoa.tdoa_mics(join_350hz, [], text="próbki 350hz")
    # tdoa.tdoa_mics(kwadrat_350hz, [], text="kwadrat 350hz")
    tdoa.tdoa_mics(join_270hz, [],  text="próbki 270hz")