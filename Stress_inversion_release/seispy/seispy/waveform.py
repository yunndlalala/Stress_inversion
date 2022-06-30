"""
@version:
author:yunnaidan
@time: 2019/10/17
@file: waveform.py
@function:
"""
import os
import glob
import logging
import subprocess
import numpy as np
import obspy
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from obspy.core import *

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

os.putenv("SAC_DISPLAY_COPYRIGHT", '0')


def _sac_error(out):
    for o in out:
        if "ERROR" in o:
            raise ValueError(o)

    return None


def cut(
        data_path=None,
        output_path=None,
        catalog_file=None):
    """
        input:
        catalog; stream_path; out_path; time window
        output:
        ./events/[origin time]/net.sta.ot.chn.SAC
        head file:
        stlo stla stel
        evlo evla evdp
        kztime
        b = 0
        other:
        The structure of data storage must be [year]/[yearmonthday]/sac file
        The catalog must be
                            time,latitude,longitude,depth,mag
                            Y-m-dTH:M:SZ,lat,lon,depth,mag
                            .............................
                            .............................
    """
    if os.path.exists(output_path):
        os.system('rm -rf %s' % output_path)
    else:
        os.makedirs(output_path)

    with open(catalog_file, 'r') as f:
        catalog = f.readlines()

    for ctlg_line in catalog[1:]:
        print('cutting event {}'.format(ctlg_line))
        ot, lat, lon, depth, mag, win_before, win_after = ctlg_line[:-1].split(',')

        win_before = float(win_before)
        win_after = float(win_after)
        ot = UTCDateTime(ot)

        # make output dir
        time_key = ''.join(['%04d' %
                            ot.year, '%02d' %
                            ot.month, '%02d' %
                            ot.day, '%02d' %
                            ot.hour, '%02d' %
                            ot.minute, '%02d' %
                            ot.second, '%03d' %
                            (ot.microsecond / 1000)])
        output_event_dir = os.path.join(output_path, time_key)
        if not os.path.exists(output_event_dir):
            os.makedirs(output_event_dir)

        # time window for slicing
        ts = ot - win_before
        ts_date_str = '%04d' % ts.year + '%02d' % ts.month + '%02d' % ts.day
        te = ot + win_after
        te_date_str = '%04d' % te.year + '%02d' % te.month + '%02d' % te.day

        start_day_path = os.path.join(data_path, str(ts.year), ts_date_str)
        # use sac
        if os.path.exists(start_day_path):
            os.chdir(start_day_path)
            streams = sorted(glob.glob('*'))
            for stream in streams:
                print(stream)
                _, net, sta, chn = stream.split('.')
                fname = '.'.join([net, sta, chn])
                output_file = os.path.join(output_event_dir, fname)

                b = ts - UTCDateTime(ts_date_str)
                e = te - UTCDateTime(ts_date_str)

                # cut event and change the head file
                long_stream_flag = False
                if te.day != ts.day:
                    next_stream = os.path.join(data_path, str(
                        te.year), te_date_str, '.'.join([te_date_str, net, sta, chn]))
                    if os.path.exists(next_stream):
                        long_stream_flag = True

                        p = subprocess.Popen(
                            ['sac'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                        s = "wild echo off \n"
                        s += "r %s \n" % os.path.join(start_day_path, stream)
                        s += "merge GAP ZERO o a %s \n" % next_stream
                        s += "w %s \n" % (os.path.join(start_day_path,
                                                       'long.' + stream))
                        s += "q \n"
                        out, err = p.communicate(s.encode())
                        out = out.decode().split('\n')
                        print('SAC out: ' + str(out))
                        _sac_error(out)

                        stream = 'long.' + stream
                    else:
                        logging.warning('The data of next day is not found!')

                p = subprocess.Popen(
                    ['sac'],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE)
                s = "wild echo off \n"
                s += "cuterr fillz \n"
                s += "cut b %s %s \n" % (b, e)
                s += "r %s \n" % os.path.join(start_day_path, stream)
                s += "ch LCALDA TRUE \n"
                s += "ch LOVROK TRUE \n"
                s += "ch nzyear %s nzjday %s \n" % (
                    str(ot.year), str(ot.julday))
                s += "ch nzhour %s nzmin %s nzsec %s \n" % (
                    str(ot.hour), str(ot.minute), str(ot.second))
                s += "ch nzmsec %s \n" % str(ot.microsecond / 1000)
                # Otherwise it will report warning:reference time is not equal
                # to zero.
                s += "ch iztype IO \n"
                # b is not needed to be modified.
                s += "ch b %s \n" % str(-win_before)
                s += "ch evlo %s evla %s evdp %s \n" % (lon, lat, depth)
                s += "ch mag %s \n" % mag
                s += "w %s \n" % output_file
                s += "q \n"
                out, err = p.communicate(s.encode())
                out = out.decode().split('\n')
                print('SAC out: ' + str(out))
                _sac_error(out)

                if long_stream_flag:
                    print('remove ' + os.path.join(start_day_path, stream))
                    os.system('rm %s' % os.path.join(start_day_path, stream))
        else:
            logging.warning('%s is not exist!' % str(ot))


def select_event_waveform(full_sta, tb_b, te_e, waveform_path):
    b_day = ''.join([str(tb_b.year),
                     str(tb_b.month).zfill(2),
                     str(tb_b.day).zfill(2)])
    e_day = ''.join([str(te_e.year),
                     str(te_e.month).zfill(2),
                     str(te_e.day).zfill(2)])

    e_day_waveform = os.path.join(waveform_path,
                                  str(te_e.year),
                                  e_day,
                                  e_day + '.' + full_sta)

    if not os.path.exists(e_day_waveform):
        print('No event waveform for %s!' % full_sta)
        event_tr = None
    else:
        if b_day == e_day:
            event_st = obspy.read(e_day_waveform)
            event_tr = event_st[0]
        else:
            b_day_waveform = os.path.join(waveform_path,
                                          str(tb_b.year),
                                          b_day,
                                          b_day + '.' + full_sta)
            if not os.path.exists(b_day_waveform):
                print('No waveform of the day before the event for %s!' % full_sta)
                event_tr = None
            else:
                event_st = obspy.read(b_day_waveform)
                event_st += obspy.read(e_day_waveform)
                event_st.merge(method=0, fill_value=0.0)
                event_tr = event_st[0]
    return event_tr


def plot_spec(
        event_tr, nperseg, ot, tb_e, te_b, te_e, f_integral_min, f_integral_max,
        before_show_l=0.2,
        after_show_l=0.05,
        cmap='coolwarm',
        ax=None,
        colorbar_ax=None,
        vmin=None,
        vmax=None
):
    tr = event_tr.copy()
    fs = tr.stats.sampling_rate

    show_e = te_e + (te_e - te_b) * after_show_l
    show_b = tb_e - (te_e - te_b) * before_show_l

    cal_b = show_b - 60
    cal_e = show_e + 60

    cla_tr = tr.slice(cal_b, cal_e)
    cla_tr.detrend('linear')
    cla_tr.detrend('constant')
    data = cla_tr.data

    f_min = 0.1
    f_max = fs / 2.0
    f, t, Sxx = spectrogram(
        data, fs,
        window='hanning', nperseg=nperseg, noverlap=int(nperseg / 2), nfft=nperseg, detrend=None,
        return_onesided=True, scaling='density', axis=-1, mode='psd')
    t = t + (cal_b - ot)

    if vmin is None or vmax is None:
        img = ax.pcolormesh(t, f, 10 * np.log10(Sxx), cmap=cmap)
    else:
        img = ax.pcolormesh(t, f, 10 * np.log10(Sxx), vmin=vmin, vmax=vmax, cmap=cmap)

    ax.plot([show_b - ot, show_e - ot], [f_integral_min, f_integral_min], '--k')
    ax.plot([show_b - ot, show_e - ot], [f_integral_max, f_integral_max], '--k')

    ax.set_ylim(f_min, f_max)
    ax.set_xlim([show_b - ot, show_e - ot])

    if colorbar_ax is not None:
        cbar = plt.colorbar(img, cax=colorbar_ax)
        cbar.ax.set_xlabel('dB')

    return None


def plot_raw(
        event_tr, ot, tb_b, tb_e, te_b, te_e,
        downsample=1,
        before_show_l=0.2,
        after_show_l=0.05,
        ax=None
):
    tr = event_tr.copy()
    fs = tr.stats.sampling_rate

    show_e = te_e + (te_e - te_b) * after_show_l
    show_b = tb_e - (te_e - te_b) * before_show_l

    show_tr = tr.slice(show_b, show_e)
    data = show_tr.data
    downsample_index = np.arange(0, len(data), downsample)
    data_downsample = data[downsample_index]
    time = downsample_index * (1 / fs) + (show_b - ot)
    ax.plot(time, data_downsample, 'gray', linewidth=0.5)

    ax.plot([tb_b - ot, tb_b - ot], [np.min(data), np.max(data)], '--k')
    ax.plot([tb_e - ot, tb_e - ot], [np.min(data), np.max(data)], '--k')
    ax.plot([te_b - ot, te_b - ot], [np.min(data), np.max(data)], '-k')
    ax.plot([te_e - ot, te_e - ot], [np.min(data), np.max(data)], '-k')

    if tb_b < show_b:
        tb_text_x = ((tb_e - ot) + (show_b - ot)) * 1 / 3.0
    else:
        tb_text_x = ((tb_e - ot) + (tb_b - ot)) * 1 / 3.0
    te_text_x = ((te_e - ot) + (te_b - ot)) * 1.2 / 3.0
    text_y = np.max(data_downsample) * 2.3 / 3.0
    ax.text(tb_text_x, text_y, '$\mathregular{T_b}$', style='italic', size=12, color='k')
    ax.text(te_text_x, text_y, '$\mathregular{T_e}$', style='italic', size=12, color='k')
    ax.annotate("",
                xy=((tb_e - ot) + (tb_text_x - (tb_e - ot)) * 2/3, text_y),
                xytext=(tb_e - ot, text_y),
                arrowprops=dict(facecolor='k', edgecolor='k', width=1.5, headwidth=7)
                )
    ax.annotate("",
                xy=((te_b - ot) + (te_text_x - (te_b - ot)) * 2/3, text_y),
                xytext=(te_b - ot, text_y),
                arrowprops=dict(facecolor='k', edgecolor='k', width=1.5, headwidth=7)
                )
    ax.annotate("",
                xy=(te_text_x + ((te_e - ot) - te_text_x) * 1/4, text_y),
                xytext=(te_e - ot, text_y),
                arrowprops=dict(facecolor='k', edgecolor='k', width=1.5, headwidth=7)
                )

    ax.set_xlim([show_b - ot, show_e - ot])
    ax.set_ylim(ax.get_ylim())

    # Scientific notation
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(xfmt)

    return None


def plot_bandpass(
        event_tr, ot, tb_e, te_b, te_e, freq_min, freq_max,
        downsample=1,
        before_show_l=0.2,
        after_show_l=0.05,
        ax=None
):
    tr = event_tr.copy()
    fs = tr.stats.sampling_rate

    show_e = te_e + (te_e - te_b) * after_show_l
    show_b = tb_e - (te_e - te_b) * before_show_l

    show_tr = tr.slice(show_b, show_e)
    show_tr.taper(0.05)
    show_tr.filter('bandpass', freqmin=freq_min, freqmax=freq_max)
    data = show_tr.data

    downsample_index = np.arange(0, len(data), downsample)
    data_downsample = data[downsample_index]
    time = downsample_index * (1 / fs) + (show_b - ot)
    ax.plot(time, data_downsample, 'k', linewidth=0.5)

    ax.set_xlim([show_b - ot, show_e - ot])

    # Scientific notation
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(xfmt)
