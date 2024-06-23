import src.derivatives as deri
import src.phaseshift as ps


def main():
    file = 'outputs/bessel_alldst/270hz/5m_5.wav'

    file2 = 'data/pilawlesie/alldst_split/5m_5.wav'

    ps.shift_cc_file(file2, plot_cc=True)
    ch1, ch2 = deri.choose_close_extremes_pairs(file2, threshold=30, do_plot=True, set_to_zero=True, xlim_start=44000,
                                                xlim_end=44100)
    print("-------calculate_ps_triangle-------")
    # ps.calculate_ps_triangle(xdist=1, zdist=5)
    print("-------shift_cc_channels-------")
    ps.shift_cc_channels(ch1, ch2, plot_cc=False)


if __name__ == "__main__":
    main()
