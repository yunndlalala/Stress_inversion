#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: yunnaidan
@time: 2021/05/13
@file: function_checking.py Testing functions of python.
"""
import numpy as np
from scipy import signal, fft
import matplotlib.pyplot as plt


def test_arrow_plotting_v1():
    """
    The head shape is symmetric regardless of the angle of the arrow in the 'annotate',
    but not in the 'arrow'.
    The units of the width and length of the head and tail are different
    in the 'annotate' and 'arrow'.
    """
    tail_width = 0.1
    head_width = 0.2
    head_length = 0.2

    fig, ax = plt.subplots()
    ax.arrow(
        0, 0, 0, 5,
        width=tail_width,
        head_width=head_width,
        head_length=head_length,
        length_includes_head=True
    )
    ax.arrow(
        0, 0, 1, 5,
        width=tail_width,
        head_width=head_width,
        head_length=head_length,
        length_includes_head=True
    )
    ax.arrow(
        0, 0, 2, 5,
        width=tail_width,
        head_width=head_width,
        head_length=head_length,
        length_includes_head=True
    )

    tail_width = 0.5
    head_width = 1.0
    head_length = 1.0
    ax.annotate(
        "",
        xy=(0, 5),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle='simple,tail_width=%f,head_width=%f,head_length=%f' %
                                   (tail_width, head_width, head_length)),
    )
    ax.annotate(
        "",
        xy=(1, 5),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle='simple,tail_width=%f,head_width=%f,head_length=%f' %
                                   (tail_width, head_width, head_length)),
    )
    ax.annotate(
        "",
        xy=(2, 5),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle='simple,tail_width=%f,head_width=%f,head_length=%f' %
                                   (tail_width, head_width, head_length)),
    )
    plt.show()

    return None


def test_arrow_plotting_v2():
    # plt.arrow(
    #     0,  0, 0, 5,
    #     width=0.1,
    #     head_width=0.2,
    #     head_length=0.2,
    #     length_includes_head=True
    # )
    # plt.arrow(
    #     0, 0, 3, 4,
    #     width=0.1,
    #     head_width=0.2,
    #     head_length=0.2,
    #     length_includes_head=True
    # )

    plt.annotate(
        '',
        xytext=(0, 0),
        xy=(0, 5),
        arrowprops=dict(arrowstyle='simple,tail_width=%f,head_width=%f,head_length=%f' %
                                   (0.5, 1.0, 1.0)),
    )
    plt.annotate(
        '',
        xytext=(0, 0),
        xy=(3, 4),
        arrowprops=dict(arrowstyle='simple,tail_width=%f,head_width=%f,head_length=%f' %
                                   (0.5, 1.0, 1.0)),
    )
    plt.show()
    return None


def test_cal_psd():
    fs = 100.0
    N = 1000
    amp = 3.0
    freq = 2

    points = np.arange(N)
    time = points / fs
    y = amp * np.sin(2 * np.pi * freq * time)

    plt.plot(time, y)
    plt.xlabel('Time')
    plt.ylabel('Count')
    plt.show()


    Amp = np.abs(fft.fft(y)) / fs
    f = fft.fftfreq(N, 1/fs)
    index = int(N/2)
    plt.plot(f[:index], Amp[:index])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (Count/Hz)')
    plt.show()
    plt.plot(f[:index], 2 * Amp[:index])
    plt.xlim([0, 10])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (Count/Hz)')
    plt.show()
    plt.plot(f[:index], 2 * Amp[:index] ** 2 * fs / N, label='scipy.fft.fft')

    P_f, Pxx_den = signal.welch(
        y, fs,
        window=[1.0] * N, nperseg=N, noverlap=0, nfft=N, detrend='constant',
        return_onesided=True, scaling='density', axis=-1, average='mean'
    )
    plt.plot(P_f, Pxx_den, label='scipy.signal.welch')

    S_f, S_t, Sxx_den = signal.spectrogram(
        y, fs,
        window=[1.0] * N, nperseg=N, noverlap=0, nfft=N, detrend='constant',
        return_onesided=True, scaling='density', axis=-1, mode='psd'
    )
    plt.plot(S_f, Sxx_den, label='scipy.signal.spectrogram')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (Count**2/Hz)')
    plt.xlim([0, 10])
    plt.legend()
    plt.show()

    return None


def test_cal_psd_parameter():
    fs = 100.0
    N = 2000
    amp = 1.0
    freq = 0.1

    points = np.arange(N)
    time = points / fs

    y = amp * np.sin(2 * np.pi * freq * time)

    # y = [0] * int(N/4)
    # y = y + [1] * int(N/2)
    # y = y + [0] * int(N/4)

    # noise = np.random.random(len(y))
    # y = y + noise

    plt.plot(time, y)
    plt.xlabel('Time')
    plt.ylabel('Count')
    plt.show()

    # Amp = np.abs(fft.fft(y)) / fs
    # f = fft.fftfreq(N, 1/fs)
    # index = int(N / 2)
    # plt.plot(f[:index], 2 * Amp[:index])
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Amplitude (Count/Hz)')
    # plt.show_samples()

    nperseg = 100
    nfft = 128

    f, t, Sxx_den = signal.spectrogram(
        y, fs,
        window='hanning',
        nperseg=nperseg, noverlap=int(nperseg/2), nfft=nfft, detrend=None,
        return_onesided=True, scaling='density', axis=-1, mode='psd'
    )
    for i in range(np.shape(Sxx_den)[1]):
        plt.plot(f, Sxx_den[:, 0], marker='o', markersize=3, label=str(t[i]))

    plt.xlim([0, 10])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (Count**2/Hz)')
    plt.legend()
    plt.show()
    plt.plot(f, Sxx_den, marker='o', markersize=3, label='length=%f second' % (N/fs))

    length = int(256)
    f, t, Sxx_den = signal.spectrogram(
        y[:length], fs,
        window='boxcar', nperseg=length, noverlap=0, nfft=length, detrend='constant',
        return_onesided=True, scaling='density', axis=-1, mode='psd'
    )
    plt.plot(f, Sxx_den, marker='o', markersize=3, label='length=%f second' % (length / fs))

    plt.xlim([0, 1])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (Count**2/Hz)')
    plt.legend()
    plt.show()
    return None


if __name__ == '__main__':
    # test_arrow_plotting_vx2()
    # test_cal_psd()
    test_cal_psd_parameter()
    pass
