#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: yunnaidan
@time: 2020/12/02
@file: FFM_model_raw_ffm.py
"""
import os
import re
import matplotlib.pyplot as plt
from geopy.distance import Point, distance
import numpy as np
import pandas as pd
import scipy

from seispy.FFM import FFM


def translate(raw_ffm_file=None,
              segment_num=None,
              output_file=None,
              ):
    with open(raw_ffm_file) as f:
        raw_ffm = f.readlines()

    data = raw_ffm[8:]
    data = [re.split(' +', line)[1:-1] for line in data]

    columns = raw_ffm[7]
    columns = re.split(' +', columns[:-1])

    raw_ffm_df = pd.DataFrame(data=data, columns=columns)

    output_ffm_df = raw_ffm_df.copy()

    if segment_num is not None:
        output_ffm_df = output_ffm_df[output_ffm_df['segnum'] == segment_num]

    output_ffm_df = output_ffm_df.astype(float)

    lat1 = np.zeros(len(output_ffm_df))
    lon1 = np.zeros(len(output_ffm_df))
    lat2 = np.zeros(len(output_ffm_df))
    lon2 = np.zeros(len(output_ffm_df))
    lat3 = np.zeros(len(output_ffm_df))
    lon3 = np.zeros(len(output_ffm_df))
    lat4 = np.zeros(len(output_ffm_df))
    lon4 = np.zeros(len(output_ffm_df))
    up_middle_lat = np.zeros(len(output_ffm_df))
    up_middle_lon = np.zeros(len(output_ffm_df))
    down_middle_lat = np.zeros(len(output_ffm_df))
    down_middle_lon = np.zeros(len(output_ffm_df))
    for row_i, row in output_ffm_df.iterrows():
        strike = row['strike']
        dip = row['dip']
        x_inc = row['xinc']
        y_inc = row['yinc'] * np.cos(dip / 180 * np.pi)

        lat = row['lat']
        lon = row['lon']

        middle = Point(lat, lon)
        up_middle = distance(kilometers=0.5 * y_inc).\
            destination(point=middle, bearing=strike - 90)
        up_middle = Point(up_middle.latitude, up_middle.longitude)
        point3 = distance(kilometers=0.5 * x_inc).\
            destination(point=up_middle, bearing=strike - 180)
        point4 = distance(kilometers=0.5 * x_inc).\
            destination(point=up_middle, bearing=strike)
        down_middle = distance(kilometers=0.5 * y_inc).\
            destination(point=middle, bearing=strike + 90)
        down_middle = Point(down_middle.latitude, down_middle.longitude)
        point1 = distance(kilometers=0.5 * x_inc).\
            destination(point=down_middle, bearing=strike)
        point2 = distance(kilometers=0.5 * x_inc).\
            destination(point=down_middle, bearing=strike - 180)

        lon1[row_i] = point1.longitude
        lat1[row_i] = point1.latitude
        lon2[row_i] = point2.longitude
        lat2[row_i] = point2.latitude
        lon3[row_i] = point3.longitude
        lat3[row_i] = point3.latitude
        lon4[row_i] = point4.longitude
        lat4[row_i] = point4.latitude
        up_middle_lon[row_i] = up_middle.longitude
        up_middle_lat[row_i] = up_middle.latitude
        down_middle_lon[row_i] = down_middle.longitude
        down_middle_lat[row_i] = down_middle.latitude

    output_ffm_df['lon1'] = lon1
    output_ffm_df['lat1'] = lat1
    output_ffm_df['lon2'] = lon2
    output_ffm_df['lat2'] = lat2
    output_ffm_df['lon3'] = lon3
    output_ffm_df['lat3'] = lat3
    output_ffm_df['lon4'] = lon4
    output_ffm_df['lat4'] = lat4
    output_ffm_df['up_middle_lon'] = up_middle_lon
    output_ffm_df['up_middle_lat'] = up_middle_lat
    output_ffm_df['down_middle_lon'] = down_middle_lon
    output_ffm_df['down_middle_lat'] = down_middle_lat

    output_ffm_df = output_ffm_df.rename(
        columns={'slip(m)': 'slip_all', 'depth': 'dep'})
    output_ffm_df.to_csv(output_file, index=False)

    return output_ffm_df


def show(
        fig=None,
        ffm_file=None,
        topography_file=None,
        topo_fill=True,
        topo_cb='terrain',
        topo_cb_show=True,
        fault_lon=None,
        fault_lat=None,
        row_n=None,
        column_n=None,
        patch=False,
        patch_linewidth=0.5,
        patch_color='k',
        point=True,
        point_size=0.01,
        point_color='k',
        item_filling=False,
        filling_cb='coolwarm',
        filling_alpha=1.0,
        item_arrow=True,
        zoom=200.0,
        arrow_color='k',
        arrow_linewidth=0.001,
        arrow_head_width_r=0.005,
        arrow_head_length_r=0.001,
):

    topo_data = np.load(topography_file)
    lon_mesh = topo_data[0, :, :].T
    lat_mesh = topo_data[1, :, :].T
    h_mesh = topo_data[2, :, :].T
    if fault_lat is not None and fault_lon is not None:
        lon_list = lon_mesh[:, 0]
        lat_list = lat_mesh[0, :]
        lon_i = np.where(
            (lon_list >= fault_lon[0]) & (
                lon_list <= fault_lon[1]))[0]
        lat_i = np.where(
            (lat_list >= fault_lat[0]) & (
                lat_list <= fault_lat[1]))[0]
        lon_mesh = lon_mesh[lon_i[0]:lon_i[-1], lat_i[0]:lat_i[-1]]
        lat_mesh = lat_mesh[lon_i[0]:lon_i[-1], lat_i[0]:lat_i[-1]]
        h_mesh = h_mesh[lon_i[0]:lon_i[-1], lat_i[0]:lat_i[-1]]

    if fig is None:
        fig = plt.figure(figsize=[12, 8])
    if topo_cb_show:
        main_loc = [0.1, 0.05, 0.8, 0.9]
        main_ax = fig.add_axes(main_loc)
        topo_cb_loc = [0.92, 0.1, 0.02, 0.35]
        topo_cb_ax = fig.add_axes(topo_cb_loc)
    if item_filling:
        ffm_cb_loc = [0.92, 0.55, 0.02, 0.35]
        ffm_cb_ax = fig.add_axes(ffm_cb_loc)
    if topo_cb_show is False and item_filling is False:
        main_ax = fig.add_subplot(111)
    if topo_fill:
        # h_mesh[np.where(h_mesh <= 0.0)] = None
        p = main_ax.pcolormesh(
            lon_mesh,
            lat_mesh,
            h_mesh,
            alpha=1.0,
            cmap=topo_cb,
            vmin=-600,
            vmax=2000
        )
        h_mesh = scipy.ndimage.gaussian_filter(h_mesh, sigma=2.0, order=0)
        main_ax.contour(
            lon_mesh,
            lat_mesh,
            h_mesh,
            [0.0],
            color='k'
        )
    else:
        p = main_ax.contour(
            lon_mesh,
            lat_mesh,
            h_mesh,
            alpha=0.7,
            cmap=topo_cb,
            vmin=-250,
            vmax=1000
        )
    if topo_cb_show:
        fig.colorbar(p, cax=topo_cb_ax)
        topo_cb_ax.set_ylabel('Height(m)')
    main_ax.set_aspect('equal')

    ffm_df = pd.read_csv(ffm_file)
    ffm = FFM(ffm_df, row_n=row_n, column_n=column_n)
    handles = ffm.plot_slip_surface(
        ax=main_ax,
        patch=patch,
        patch_linewidth=patch_linewidth,
        patch_color=patch_color,
        point=point,
        point_size=point_size,
        point_color=point_color,
        item_filling=item_filling,
        filling_cb=filling_cb,
        filling_alpha=filling_alpha,
        item_arrow=item_arrow,
        zoom=zoom,
        arrow_color=arrow_color,
        arrow_linewidth=arrow_linewidth,
        arrow_head_width_r=arrow_head_width_r,
        arrow_head_length_r=arrow_head_length_r
    )
    if item_filling is True:
        main_ax, cb = handles
        fig.colorbar(cb, cax=ffm_cb_ax)
        ffm_cb_ax.set_ylabel('Slip(m)')
    else:
        main_ax = handles
    main_ax.set_xlim(fault_lon)
    main_ax.set_ylim(fault_lat)

    return main_ax
