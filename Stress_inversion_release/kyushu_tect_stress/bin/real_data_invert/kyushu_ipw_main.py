#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: yunnaidan
@time: 2020/12/02
@file: FFM_model_ipw03d456.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from stress import utils as stu
from stress import plot_utils as stpu
from stress import searcher
from seispy.FFM import surface_plotting

import sys
sys.path.append('../service')
import FFM_model
import topographic_stress


row_n = 7
column_n = 11
topo_lat = [32, 33.5]
topo_lon = [130, 132]
fault_lat = [32.6, 33.0]
fault_lon = [130.6, 131.2]

only_ffm_file = '../../data/real_data_invert/FFM/FM_ipw03d456/FM_ipw03d456.csv'
topography_file = '../../data/real_data_invert/topography/combined_topo_130-132_32-33.5.npy'

density = 2.6
ffm_path = '../../data/real_data_invert/ipw_topo_stress_combined_python'
if not os.path.exists(ffm_path):
    os.makedirs(ffm_path)
ffm_file = os.path.join(ffm_path, 'FFM_and_topo_seg1_%.1f.csv' % density)

# Model parameters
data_sigma = 1.5
mode = 'weight2'
stress_log = True
parameters = 'Shmin'
# Available value of parameters:
# 'Shmin':  uniaxial tensional stress mode
# 'SHmax': uniaxial compressional stress mode
# 'both': plane stress mode for strike-slip fault type
# 'both pull': plane stress mode for normal fault type

if parameters == 'both_pull':
    both_pull = True
else:
    both_pull = False

grid_path = '../../result/real_data_invert/ipw_combined_topo_stress_python' + \
    '/mode_' + str(mode) + \
    '/stress_log_' + str(stress_log) + \
    '/parameters_' + str(parameters)
if not os.path.exists(grid_path):
    os.makedirs(grid_path)

if parameters in ['both', 'both_pull', 'both_press']:
    max_mag_bin = 0.2
    min_mag_bin = 0.2
    angle_bin = 10.0
    prior_shape = {'max_mag_loc': 4.0,
                   'max_mag_scale': 10.0,
                   'min_mag_loc': 4.0,
                   'min_mag_scale': 10.0,
                   'angle_loc': -90.0,
                   'angle_scale': 180.0
                   }

    max_mag_num = int(prior_shape['max_mag_scale'] / max_mag_bin)
    max_mag_min = prior_shape['max_mag_loc'] + max_mag_bin / 2
    max_mag_max = prior_shape['max_mag_loc'] + \
        prior_shape['max_mag_scale'] - max_mag_bin / 2

    min_mag_num = int(prior_shape['min_mag_scale'] / min_mag_bin)
    min_mag_min = prior_shape['min_mag_loc'] + min_mag_bin / 2
    min_mag_max = prior_shape['min_mag_loc'] + \
        prior_shape['min_mag_scale'] - min_mag_bin / 2

    angle_num = int(prior_shape['angle_scale'] / angle_bin)
    angle_min = prior_shape['angle_loc'] + angle_bin / 2
    angle_max = prior_shape['angle_loc'] + \
        prior_shape['angle_scale'] - angle_bin / 2

    grid_search_file = os.path.join(
        grid_path,
        'datasigma_' + str(data_sigma) +
        '_maxmag_' + str(max_mag_min) + '_' + str(max_mag_max) +
        '_minmag_' + str(min_mag_min) + '_' + str(min_mag_max) +
        '_angle_' + str(angle_min) + '_' + str(angle_max) +
        '_maxmagbin_' + str(max_mag_bin) +
        '_minmagbin_' + str(min_mag_bin) +
        '_anglebin_' + str(angle_bin) +
        '_grid_search' +
        '_%.1f' % density +
        '.npy'
    )
    if parameters == 'both_pull':
        reprocessed_grid_file = grid_search_file[:-4] + '_reprocessed.npy'

elif parameters == 'SHmax':
    max_mag_bin = 0.1
    angle_bin = 1.0
    prior_shape = {'max_mag_loc': 4.0,
                   'max_mag_scale': 10.0,
                   'angle_loc': -90.0,
                   'angle_scale': 180.0
                   }

    max_mag_num = int(prior_shape['max_mag_scale'] / max_mag_bin)
    max_mag_min = prior_shape['max_mag_loc'] + max_mag_bin / 2
    max_mag_max = prior_shape['max_mag_loc'] + \
        prior_shape['max_mag_scale'] - max_mag_bin / 2

    min_mag_min = None
    min_mag_max = None
    min_mag_num = None

    angle_num = int(prior_shape['angle_scale'] / angle_bin)
    angle_min = prior_shape['angle_loc'] + angle_bin / 2
    angle_max = prior_shape['angle_loc'] + \
        prior_shape['angle_scale'] - angle_bin / 2

    grid_search_file = os.path.join(
        grid_path,
        'datasigma_' + str(data_sigma) +
        '_maxmag_' + str(max_mag_min) + '_' + str(max_mag_max) +
        '_angle_' + str(angle_min) + '_' + str(angle_max) +
        '_maxmagbin_' + str(max_mag_bin) +
        '_anglebin_' + str(angle_bin) +
        '_grid_search'
        '_%.1f' % density +
        '.npy'
    )

elif parameters == 'Shmin':
    min_mag_bin = 0.1
    angle_bin = 1.0  # azimuth of x_stress, i.e. between x_stress and East (-90~90)
    prior_shape = {
        'min_mag_loc': 4.0,
        'min_mag_scale': 10.0,
        'angle_loc': -90.0,
        'angle_scale': 180.0
    }

    max_mag_min = None
    max_mag_max = None
    max_mag_num = None

    min_mag_num = int(prior_shape['min_mag_scale'] / min_mag_bin)
    min_mag_min = prior_shape['min_mag_loc'] + min_mag_bin / 2
    min_mag_max = prior_shape['min_mag_loc'] + \
        prior_shape['min_mag_scale'] - min_mag_bin / 2

    angle_num = int(prior_shape['angle_scale'] / angle_bin)
    angle_min = prior_shape['angle_loc'] + angle_bin / 2
    angle_max = prior_shape['angle_loc'] + \
        prior_shape['angle_scale'] - angle_bin / 2

    grid_search_file = os.path.join(
        grid_path,
        'datasigma_' + str(data_sigma) +
        '_minmag_' + str(min_mag_min) + '_' + str(min_mag_max) +
        '_angle_' + str(angle_min) + '_' + str(angle_max) +
        '_minmagbin_' + str(min_mag_bin) +
        '_anglebin_' + str(angle_bin) +
        '_grid_search' +
        '_%.1f' % density +
        '.npy'
    )

else:
    pass


def show_ffm():

    ax = FFM_model.show(
        ffm_file=only_ffm_file,
        topography_file=topography_file,
        fault_lon=fault_lon,
        fault_lat=fault_lat,
        row_n=row_n,
        column_n=column_n,
        patch=True,
        patch_color='gray',
        patch_linewidth=0.5,
        point=True,
        point_color='k',
        point_size=0.00001,
        item_filling=True,
        filling_cb='Greys',
        filling_alpha=0.5,
        item_arrow=True,
        zoom=300.0,
        arrow_color='k',
        arrow_linewidth=0.0005,
        arrow_head_width_r=0.9,
        arrow_head_length_r=0.65,
        topo_fill=True,
        topo_cb='terrain',
        topo_cb_show=True
    )

    return ax


def cal_topographic_stress_tensor():

    topographic_stress.from_ffm(
        ffm_file=only_ffm_file,
        topography_file=topography_file,
        output_file=ffm_file,
        series=0,
        density=density * 1e3
    )

    return None


if os.path.exists(ffm_file):
    ffm_df = pd.read_csv(ffm_file)
    topo_tensors = stu.list_tensors2array_tensors(
        ffm_df[['xx', 'xy', 'xz', 'yy', 'yz', 'zz']].values)
    tar_df = ffm_df[ffm_df['slip_all'] != 0.0]
    tar_topo_tensors = stu.list_tensors2array_tensors(
        tar_df[['xx', 'xy', 'xz', 'yy', 'yz', 'zz']].values)


def plot_stresses_on_FFM_and_topography():
    # topography data
    topo_data_file = topography_file
    topo_data = np.load(topo_data_file)
    X = topo_data[0, :, :].T
    Y = topo_data[1, :, :].T
    Z = topo_data[2, :, :].T

    lat = [32.6, 33.1]
    lon = [130.6, 131.2]
    lon_list = X[:, 0]
    lat_list = Y[0, :]
    X_i = np.where((lon_list >= lon[0]) & (lon_list <= lon[1]))[0]
    Y_i = np.where((lat_list >= lat[0]) & (lat_list <= lat[1]))[0]
    new_X = X[X_i[0]:X_i[-1], Y_i[0]:Y_i[-1]]
    new_Y = Y[X_i[0]:X_i[-1], Y_i[0]:Y_i[-1]]
    new_Z = Z[X_i[0]:X_i[-1], Y_i[0]:Y_i[-1]]

    # stress data
    tensors_file = ffm_file
    tensor_df = pd.read_csv(tensors_file)
    strikes = tensor_df['strike'].values
    dips = tensor_df['dip'].values
    tensors = tensor_df[['xx', 'xy', 'xz', 'yy', 'yz', 'zz']].values
    tensors = stu.list_tensors2array_tensors(tensors)
    norm_stresses = stu.get_norm_stress(
        strike=strikes,
        dip=dips,
        stress_tensor=tensors,
        angle='degrees'
    )
    strike_stresses = stu.get_strike_shear_stress(
        strike=strikes,
        dip=dips,
        stress_tensor=tensors,
        angle='degrees')
    dip_stresses = stu.get_dip_shear_stress(
        strike=strikes,
        dip=dips,
        stress_tensor=tensors,
        angle='degrees')
    stresses = [norm_stresses / 1e6, dip_stresses / 1e6, strike_stresses / 1e6]

    # plotting
    fig = plt.figure(figsize=[10, 7])
    axes = [
        fig.add_axes([0.01, 0.55, 0.45, 0.4]),
        fig.add_axes([0.5, 0.55, 0.45, 0.4]),
        fig.add_axes([0.01, 0.08, 0.45, 0.4])
    ]
    cb_axes = [
        fig.add_axes([0.44, 0.55, 0.01, 0.4]),
        fig.add_axes([0.92, 0.55, 0.01, 0.4]),
        fig.add_axes([0.44, 0.08, 0.01, 0.4])
    ]
    stress_type = ['norm_stress',
                   'dip_stress',
                   'strike_stress']
    titles = [
        'Norm',
        'Dip-slip',
        'Strike-slip'
    ]
    stress_type_name = [
        'Norm stress (MPa)',
        'Dip-slip shear stress (MPa)',
        'Strike-slip shear stress (MPa)']

    for i in range(3):
        ax = axes[i]
        cb_ax = cb_axes[i]
        ct = ax.contour(
            new_X, new_Y, new_Z,
            alpha=0.5,
            cmap='binary',
            vmin=-250,
            vmax=1000)

        stress_data = stresses[i]
        tensor_df[stress_type[i]] = stress_data
        limit = np.max(abs(stress_data))
        # limit = limits[i]
        ax, sm = surface_plotting(tensor_df,
                                  stress_type[i],
                                  ax=ax,
                                  patch=True,
                                  patch_linewidth=0.5,
                                  patch_color='gray',
                                  point=False,
                                  point_size=0.1,
                                  point_color='k',
                                  item_filling=True,
                                  filling_cb='coolwarm',
                                  filling_vmin=-1 * limit,
                                  filling_vmax=limit,
                                  filling_alpha=0.8,
                                  item_arrow=False,
                                  zoom=1.0,
                                  arrow_color='k',
                                  arrow_linewidth=0.001,
                                  arrow_head_width_r=0.001,
                                  )
        ax.set_title(titles[i])
        cb = plt.colorbar(sm, cax=cb_ax)
        cb_ax.set_ylabel(stress_type_name[i])
        ax.set_aspect('equal')
        ax.set_xlim(lon)
        ax.set_ylim(lat)
        if i == 0:
            ax.set_xticks([])
            ax.set_ylabel('Latitude')
        if i == 1:
            ax.set_xlabel('Longitude')
            ax.set_yticks([])
        if i == 2:
            ax.set_ylabel('Latitude')
            ax.set_xlabel('Longitude')

    return fig


def grid_search():
    searcher.grid(FFM_df=tar_df,
                  topo_tensors=tar_topo_tensors,
                  density=density * 1e3,
                  angle_min=angle_min,
                  angle_max=angle_max,
                  angle_num=angle_num,
                  max_mag_min=max_mag_min,
                  max_mag_max=max_mag_max,
                  max_mag_num=max_mag_num,
                  min_mag_min=min_mag_min,
                  min_mag_max=min_mag_max,
                  min_mag_num=min_mag_num,
                  data_sigma=data_sigma,
                  likelihood_mode=mode,
                  stress_log=stress_log,
                  both_pull=both_pull,
                  output_file=grid_search_file)

    return None


def show_grid_search():
    if parameters == 'both':
        fig = plt.figure(figsize=[10, 8])
        axes = fig.subplots(3, 3)
        plt.subplots_adjust(left=0.08, bottom=0.08, wspace=0.15, hspace=0.15)

        bar_axes = [
            plt.axes([0.23, 0.57, 0.08, 0.01]),
            plt.axes([0.23, 0.29, 0.08, 0.01]),
            plt.axes([0.52, 0.29, 0.08, 0.01])
        ]

        searcher.show_grid(
            grid_search_file=grid_search_file,
            para_label=['Direction (degree)', 'log(SHmax) (Pa)', 'log(Shmin) (Pa)'],
            fig=fig,
            axes=axes,
            bar_axes=bar_axes
        )
    elif parameters == 'both_pull':
        fig = plt.figure(figsize=[10, 8])
        axes = fig.subplots(3, 3)
        plt.subplots_adjust(left=0.08, bottom=0.08, wspace=0.15, hspace=0.15)

        bar_axes = [
            plt.axes([0.23, 0.57, 0.08, 0.01]),
            plt.axes([0.23, 0.29, 0.08, 0.01]),
            plt.axes([0.52, 0.29, 0.08, 0.01])
        ]

        searcher.show_grid(
            grid_search_file=reprocessed_grid_file,
            para_label=['Direction (degree)', 'log(SHmax) (Pa)', 'log(Shmin) (Pa)'],
            fig=fig,
            axes=axes,
            bar_axes=bar_axes,
            uncertainty=False
        )
    elif parameters == 'SHmax':
        searcher.show_grid_2D(
            grid_search_file=grid_search_file,
            para_label=['Direction (degree)', 'log(SHmax) (Pa)']
        )
    elif parameters == 'Shmin':
        searcher.show_grid_2D(
            grid_search_file=grid_search_file,
            para_label=['Direction (degree)', 'log(Shmin) (Pa)'],
            limitation_file=None
        )
    else:
        pass

    return None


def show_fitting_result(
        max_mag=-np.inf,
        direction=6.5,
        min_mag=7.55,
        tectonic=True,
        topographic=True,
        combined=True,
        slip_flag=True,
        zoom_k=np.array([2, 0.5, 2]),
        slip_k=6,
        arrow_linewidth=0.04,
        arrow_head_length_r=0.35,
        arrow_head_width_r=0.9,
        ax=None,
):
    tensors_list = []
    label = []
    color = []
    zoom = []
    if tectonic or combined:
        tectonic_tensor = \
            stu.gen_tectonic_tensor(
                shmax=max_mag,
                direction=direction,
                shmin=min_mag,
                stress_log=stress_log,
                both_pull=both_pull,
            )
        tectonic_tensors = np.expand_dims(tectonic_tensor, 0). \
            repeat(len(ffm_df), axis=0)
    if tectonic:
        tensors_list.append(tectonic_tensors)
        zoom_tect_magnitude = len(str(int(np.max(abs(tectonic_tensors))))) - 1
        zoom.append(10**zoom_tect_magnitude)
        label.append('tectonic')
        color.append('tab:blue')
    if topographic:
        tensors_list.append(topo_tensors)
        zoom_topo_magnitude = len(str(int(np.max(abs(topo_tensors))))) - 1
        label.append('topographic')
        color.append('tab:green')
        zoom.append(10**zoom_topo_magnitude)
    if combined:
        tensors_list.append(topo_tensors + tectonic_tensors)
        zoom_comb_magnitude = len(
            str(int(np.max(abs(topo_tensors + tectonic_tensors))))) - 1
        zoom.append(10**zoom_comb_magnitude)
        label.append('total')
        color.append('tab:red')

    zoom = np.array(zoom)
    zoom = zoom * zoom_k

    slip_magnitude = len(
        str(int(np.max(ffm_df['slip_all'])))) - 1
    if ax is None:
        fig, ax = plt.subplots()
    ax = stpu.plot_tensors_plane(
        FFM_df=ffm_df,
        row_n=row_n,
        tensors_list=tensors_list,
        label=label,
        color=color,
        zoom=zoom,
        slip=slip_flag,
        slip_zoom=10**slip_magnitude * slip_k,
        slip_color='k',
        arrow_linewidth=arrow_linewidth,
        arrow_head_length_r=arrow_head_length_r,
        arrow_head_width_r=arrow_head_width_r,
        ax=ax,
    )

    plt.xlim([-1, column_n])
    plt.ylim([-1, row_n])
    plt.ylabel('Dip')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_yticks(np.arange(row_n))
    ax.set_yticklabels(np.arange(row_n)[::-1].astype('str'))
    ax.set_ylabel('# of sub-fault along dip')

    ax.set_xticks(np.arange(column_n))
    ax.set_xticklabels(np.arange(column_n).astype('str'))
    ax.set_xlabel('# of sub-faults along strike')

    return None


def reprocess_both_pull_grid_search_result():
    search_data = np.load(
        grid_search_file, allow_pickle=True)
    angles, max_mags, min_mags, L_array_raw, _ = search_data
    for i in range(L_array_raw.shape[1]):
        L_array_raw[:, i, :i] = 0.0
    norm_factor = max_mag_bin * min_mag_bin * angle_bin
    normalization = np.sum(L_array_raw) * norm_factor
    L_array = L_array_raw / normalization
    saving_data = [angles, max_mags, min_mags, L_array_raw, L_array]
    np.save(reprocessed_grid_file, np.array(saving_data))
    return None


if __name__ == '__main__':

    show_ffm()
    plt.show()

    cal_topographic_stress_tensor()

    plot_stresses_on_FFM_and_topography()
    plt.show()

    grid_search()

    # If the stress mode is both pull
    # reprocess_both_pull_grid_search_result()

    show_grid_search()
    plt.show()

    show_fitting_result()
    plt.show()

    pass
