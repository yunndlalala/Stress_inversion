#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: yunnaidan
@time: 2020/10/26
@file: smoothing.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters

from seispy import FFM


def _smoothing_slip(rakes=None,
                    slips=None,
                    sigma=None,
                    row_n=12,
                    column_n=21
                    ):

    sigma = [1, 1] if sigma is None else sigma

    xslip_array = np.zeros([row_n, column_n])
    yslip_array = xslip_array.copy()

    arrow_b_x = np.array([int(i / row_n)
                          for i in range(len(rakes))])
    arrow_b_y = np.array([row_n - 1 - i % row_n
                          for i in range(len(rakes))])
    slip_x_observed = \
        np.array([slips[i] * np.cos(np.deg2rad(rakes[i]))
                  for i in range(len(slips))])
    slip_y_observed = \
        np.array([slips[i] * np.sin(np.deg2rad(rakes[i]))
                  for i in range(len(slips))])

    xslip_array[arrow_b_y, arrow_b_x] = slip_x_observed
    yslip_array[arrow_b_y, arrow_b_x] = slip_y_observed

    xslip_filtered_array = filters.gaussian_filter(
        xslip_array, sigma, mode='constant', cval=0.0)
    x_ref_array = np.ones(np.shape(xslip_filtered_array))
    x_ref_filtered_array = filters.gaussian_filter(
        x_ref_array, sigma, mode='constant', cval=0.0)
    xslip_filtered_array = xslip_filtered_array / x_ref_filtered_array

    yslip_filtered_array = filters.gaussian_filter(
        yslip_array, sigma, mode='constant', cval=0.0)
    y_ref_array = np.ones(np.shape(yslip_filtered_array))
    y_ref_filtered_array = filters.gaussian_filter(
        y_ref_array, sigma, mode='constant', cval=0.0)
    yslip_filtered_array = yslip_filtered_array / y_ref_filtered_array

    smoothing_rakes = np.array([np.rad2deg(
        np.arctan2(yslip_filtered_array[arrow_b_y[i], arrow_b_x[i]],
                   xslip_filtered_array[arrow_b_y[i], arrow_b_x[i]]))
        for i in range(len(rakes))])

    smoothing_slips = np.array([np.sqrt(
        xslip_filtered_array[arrow_b_y[i], arrow_b_x[i]]**2 +
        yslip_filtered_array[arrow_b_y[i], arrow_b_x[i]]**2)
        for i in range(len(rakes))])

    return smoothing_rakes, smoothing_slips


def _smoothing_rake(rakes=None,
                    sigma=None,
                    row_n=12,
                    column_n=21
                    ):
    sigma = [1, 1] if sigma is None else sigma

    # -180 to 180 -> 0 to 360
    rakes[rakes < 0.0] = rakes[rakes < 0.0] + 360

    rake_array = np.zeros([row_n, column_n])
    arrow_b_x = np.array([int(i / row_n)
                          for i in range(len(rakes))])
    arrow_b_y = np.array([row_n - 1 - i % row_n
                          for i in range(len(rakes))])
    rake_array[arrow_b_y, arrow_b_x] = rakes

    rake_filtered_array = filters.gaussian_filter(
        rake_array, sigma, mode='constant', cval=0.0)
    ref_array = np.ones(np.shape(rake_array))
    ref_filtered_array = filters.gaussian_filter(
        ref_array, sigma, mode='constant', cval=0.0)
    rake_filtered_array = rake_filtered_array / ref_filtered_array

    smoothing_rakes = np.array([rake_filtered_array[arrow_b_y[i], arrow_b_x[i]]
                                for i in range(len(rakes))])
    # 0 to 360 -> -180 to 180
    smoothing_rakes[smoothing_rakes > 180] = smoothing_rakes[smoothing_rakes > 180] - 360

    return smoothing_rakes


def gen_smoothing_ffm(raw_ffm_df=None,
                      row_n=12,
                      column_n=21,
                      smoothing_sigma=None,
                      smoothing_mode='rake',
                      output_file='smoothing_ffm.csv'
                      ):
    if raw_ffm_df is None:
        raise ValueError('Please input raw ffm file!')

    # Smoothing
    rakes = raw_ffm_df['rake'].values
    slips = raw_ffm_df['slip_all'].values
    smoothing_sigma = np.array(
        [0.1, 0.1]) if smoothing_sigma is None else smoothing_sigma

    if smoothing_mode == 'rake':
        smoothed_rakes = _smoothing_rake(
            rakes=rakes,
            sigma=smoothing_sigma,
            row_n=row_n,
            column_n=column_n,
        )
        smoothing_ffm_df = raw_ffm_df.copy()
        smoothing_ffm_df['rake'] = smoothed_rakes

    elif smoothing_mode == 'slip':
        smoothed_rakes, smoothed_slips = _smoothing_slip(
            rakes=rakes,
            slips=slips,
            sigma=smoothing_sigma,
            row_n=row_n,
            column_n=column_n,
        )
        smoothing_ffm_df = raw_ffm_df.copy()
        smoothing_ffm_df['rake'] = smoothed_rakes
        smoothing_ffm_df['raw_rake'] = rakes
        smoothing_ffm_df['slip_all'] = smoothed_slips
        smoothing_ffm_df['raw_slip'] = slips
    else:
        raise ValueError('No this smoothing mode!')

    # Output
    smoothing_ffm_df.to_csv(output_file, index=False)

    return smoothing_ffm_df


def show_smoothing_result(
        ax=None,
        row_n=12,
        raw_ffm_df=None,
        smoothing_ffm_df=None,
):
    if raw_ffm_df is None:
        raise ValueError('Please input raw ffm DataFrame!')
    if smoothing_ffm_df is None:
        raise ValueError('Please input smoothing ffm DataFrame!')
    if ax is None:
        fig, ax = plt.subplots()

    # Estimate slip zoom
    slip_magnitude = len(
        str(int(np.max(raw_ffm_df['slip_all'])))) - 1
    slip_zoom = 10 ** slip_magnitude * 2

    #
    raw_ffm = FFM.FFM(raw_ffm_df, row_n=row_n)
    raw_ffm.plot_slip_plane(ax=ax,
                            zoom=slip_zoom,
                            color='k',
                            label='raw slip',
                            line_width=0.01,
                            head_width=0.2)
    smoothing_ffm = FFM.FFM(smoothing_ffm_df, row_n=row_n)
    smoothing_ffm.plot_slip_plane(ax=ax,
                                  zoom=slip_zoom,
                                  color='r',
                                  label='smoothed slip',
                                  line_width=0.01,
                                  head_width=0.2)
    return ax
