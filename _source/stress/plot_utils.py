#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: yunnaidan
@time: 2020/04/15
@file: plot_utils.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from seispy import FFM
from stress.utils import \
    get_dip_shear_stress, \
    get_strike_shear_stress,\
    get_max_shear_stress, \
    get_norm_stress


def plot_stress_surface(
        FFM_df=None,
        stress=None,
        ax=None,
        patch=False,
        point=True,
        item_filling=False,
        item_arrow=True,
        zoom=1.0,
        patch_color='k',
        point_color='k',
        arrow_color='k',
        arrow_width=0.001
):
    if FFM_df or stress is None:
        raise ValueError('Please input FFM_df and stress!')

    FFM_df['stress'] = stress
    ax = FFM.surface_plotting(FFM_df,
                              'stress',
                              ax=ax,
                              patch=patch,
                              point=point,
                              item_filling=item_filling,
                              item_arrow=item_arrow,
                              zoom=zoom,
                              patch_color=patch_color,
                              point_color=point_color,
                              arrow_color=arrow_color,
                              arrow_width=arrow_width
                              )
    return ax


def plot_stress_plane(
        FFM_df=None,
        row_n=12,
        on_depth=False,
        by_rake=True,
        rake_stresses=None,
        strike_stresses=None,
        dip_stresses=None,
        ax=None,
        zoom=1.0,
        color='gray',
        label='',
        arrow_linewidth=0.001,
        arrow_head_length_r=0.001,
        arrow_head_width_r=0.001,
):

    if by_rake:
        if rake_stresses is None:
            raise ValueError('Please input rake_stress!')
        FFM_df['rake_stress'] = rake_stresses
    else:
        if strike_stresses is None or dip_stresses is None:
            raise ValueError('Please input strike_stress and dip_stress!')
        FFM_df['strike_stress'] = strike_stresses
        FFM_df['dip_stress'] = dip_stresses

    ax = FFM.plane_plotting(FFM_df,
                            row_n=row_n,
                            on_depth=on_depth,
                            by_rake=by_rake,
                            rake_item_name='rake_stress',
                            strike_item_name='strike_stress',
                            dip_item_name='dip_stress',
                            ax=ax,
                            zoom=zoom,
                            color=color,
                            label=label,
                            arrow_linewidth=arrow_linewidth,
                            arrow_head_length_r=arrow_head_length_r,
                            arrow_head_width_r=arrow_head_width_r,
                            )

    return ax


def plot_tensors_plane(
        FFM_df=None,
        row_n=12,
        on_depth=False,
        tensors_list=None,
        label=None,
        color=None,
        zoom=None,
        slip=True,
        slip_zoom=1.0,
        slip_color='gray',
        arrow_linewidth=0.001,
        arrow_head_length_r=0.001,
        arrow_head_width_r=0.001,
        ax=None,
):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    # Plot stress tensors
    strikes = FFM_df['strike'].values
    dips = FFM_df['dip'].values
    for i in range(len(tensors_list)):
        tensors = tensors_list[i]
        strike_stresses = get_strike_shear_stress(
            strike=strikes,
            dip=dips,
            stress_tensor=tensors,
            angle='degrees')
        dip_stresses = get_dip_shear_stress(
            strike=strikes,
            dip=dips,
            stress_tensor=tensors,
            angle='degrees')
        zero_index = FFM_df[FFM_df['slip_all'] == 0.0].index
        strike_stresses[zero_index] = 0.0
        dip_stresses[zero_index] = 0.0

        ax = plot_stress_plane(
            FFM_df=FFM_df,
            row_n=row_n,
            on_depth=on_depth,
            by_rake=False,
            rake_stresses=None,
            strike_stresses=strike_stresses,
            dip_stresses=dip_stresses,
            ax=ax,
            zoom=zoom[i],
            color=color[i],
            label=label[i],
            arrow_linewidth=arrow_linewidth,
            arrow_head_length_r=arrow_head_length_r,
            arrow_head_width_r=arrow_head_width_r,
        )
        ax.arrow(0, - (i + 1) * 0.5 + 1, 1.0, 0,
                 color=color[i],
                 width=arrow_linewidth,
                 head_width=arrow_head_length_r * arrow_head_width_r,
                 head_length=arrow_head_length_r,
                 length_includes_head=True,
                 )
        ax.text(1, - (i + 1) * 0.5 + 0.9, label[i] + ' %.1E' % zoom[i] + ' Pa')

    if slip:
        # Plot slip
        ffm = FFM.FFM(FFM_df, row_n=row_n)
        ax = ffm.plot_slip_plane(ax=ax,
                                 on_depth=on_depth,
                                 zoom=slip_zoom,
                                 color=slip_color,
                                 label='slip',
                                 arrow_linewidth=arrow_linewidth,
                                 arrow_head_length_r=arrow_head_length_r,
                                 arrow_head_width_r=arrow_head_width_r,
                                 )
        ax.arrow(0, 1, 1.0, 0,
                 color=slip_color,
                 width=arrow_linewidth,
                 head_width=arrow_head_length_r * arrow_head_width_r,
                 head_length=arrow_head_length_r,
                 length_includes_head=True,
                 )
        ax.text(1.0, 0.9, 'slip' + ' %.1E' % slip_zoom + ' m')

    # plt.legend(loc='lower right')

    if 'fig' in dir():
        return fig, ax
    else:
        return ax


def plot_shear_stress_values_from_tensors(
        ffm_df=None,
        tensors_list=None,
        label=None,
        color=None,
):
    fig, ax = plt.subplots()
    for t_index, tensors in enumerate(tensors_list):
        max_shear_stress, _ = get_max_shear_stress(
            strike=ffm_df['strike'].values,
            dip=ffm_df['dip'].values,
            stress_tensor=tensors,
            angle='degrees')
        mean_stress = np.mean(max_shear_stress)
        ax.plot(max_shear_stress,
                 label=label[t_index] + '%.2f' % np.log10(mean_stress),
                 color=color[t_index])

    plt.legend()
    return fig, ax
