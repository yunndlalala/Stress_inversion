#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: yunnaidan
@time: 2022/02/09
@file: catalog.py
"""
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from obspy import UTCDateTime
from geopy import distance
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.image import imread
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from PIL import ImageFile
from PIL import Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from seispy import points


def _determine_space_range(catalog_df=None, lat_lim=None, lon_lim=None):
    if catalog_df is not None:
        latitudes = catalog_df['latitude'].values
        longitudes = catalog_df['longitude'].values
        lon_lim = np.array([np.nanmin(longitudes), np.nanmax(longitudes)])
        lat_lim = np.array([np.nanmin(latitudes), np.nanmax(latitudes)])
    else:
        if lat_lim is None and lon_lim is None:
            raise ValueError('At least input catalog or limitations of longitude and latitude.')
    space_range = np.append(lon_lim, lat_lim)
    return space_range


def _plot_map(catalog_df, mag_min=None, lat_lim=None, lon_lim=None, **kwargs):
    longitudes = catalog_df['longitude'].values
    latitudes = catalog_df['latitude'].values
    magnitudes = catalog_df['magnitude'].values

    tif_path = '/Users/yunnaidan/.local/share/cartopy/HYP_HR_SR_OB_DR/'
    
    if 'fig_size' in kwargs.keys():
        fig_size = kwargs['fig_size']
    else:
        fig_size = None
    fig = plt.figure(figsize=fig_size)
    proj = ccrs.PlateCarree()
    ax = fig.add_subplot(111, projection=proj)
    if lat_lim is None and lon_lim is None:
        extent = _determine_space_range(catalog_df=catalog_df)
    else:
        extent = _determine_space_range(lat_lim=lat_lim, lon_lim=lon_lim)
    ax.set_extent(extent, crs=proj)

    ax.add_feature(cfeat.COASTLINE.with_scale('50m'), linewidth=0.6, zorder=10)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=1.2,
        color='k',
        alpha=0.5,
        linestyle='--'
    )
    gl.xlabels_top = False  # 关闭顶端的经纬度标签
    gl.ylabels_right = False  # 关闭右侧的经纬度标签
    gl.xformatter = LONGITUDE_FORMATTER  # x轴设为经度的格式
    gl.yformatter = LATITUDE_FORMATTER

    # topography
    ax.imshow(
        imread(tif_path + 'HYP_HR_SR_OB_DR.tif'),
        origin='upper', transform=proj,
        extent=[-180, 180, -90, 90]
    )

    # events
    if mag_min is None:
        tar_indexs = np.where(
            (longitudes >= extent[0]) &
            (longitudes <= extent[1]) &
            (latitudes >= extent[2]) &
            (latitudes <= extent[3])
        )
    else:
        tar_indexs = np.where(
            (longitudes >= extent[0]) &
            (longitudes <= extent[1]) &
            (latitudes >= extent[2]) &
            (latitudes <= extent[3]) &
            (magnitudes >= mag_min)
        )
    tar_event_data = np.array([
        longitudes[tar_indexs],
        latitudes[tar_indexs],
        magnitudes[tar_indexs]
    ])
    tar_event_data = tar_event_data.T
    event_points = points.Points(tar_event_data)

    if 'event_attribute' in kwargs.keys():
        event_attribute = kwargs['event_attribute']
        event_attribute['transform'] = proj
    else:
        event_attribute = {'transform': proj}
    event_mappable = event_points.colored_plot(ax=ax, **event_attribute)

    cax = fig.add_axes([
        ax.get_position().x1 + 0.01,
        ax.get_position().y0,
        0.02,
        ax.get_position().height
        ])
    cbar = plt.colorbar(event_mappable, ax=ax, cax=cax)
    cbar.ax.set_ylabel('magnitude')

    fig.tight_layout()

    return fig


def _cut_catalog(catalog_df, event_row, time_window=None, space_window=None, sorted=False, step=1000):
    if time_window is not None:
        if sorted:
            step_n = 1
            while True:
                if step * step_n >= len(catalog_df) or catalog_df.iloc[step * step_n - 1]['time'] >= time_window[1]:
                    tar_df = catalog_df[:step * step_n]
                    break
                else:
                    step_n += 1
        else:
            tar_df = catalog_df

        tar_df = tar_df[
            (tar_df['time'] >= time_window[0]) &
            (tar_df['time'] < time_window[1])
            ]

    else:
        tar_df = catalog_df

    if space_window is not None:
        tar_df = tar_df[
            (abs(tar_df['longitude'] - event_row['longitude']) <= space_window / 100) &
            (abs(tar_df['latitude'] - event_row['latitude']) <= space_window / 100)
            ]
        distance_flag = [
            distance.distance(
                (row['latitude'], row['longitude']),
                (event_row['latitude'], event_row['longitude'])
            ).km <= space_window
            for _, row in tar_df.iterrows()
        ]
        tar_df = tar_df[distance_flag]
    else:
        pass

    return tar_df


def _plot_mt(catalog_df, ax=None, **kwargs):
    source_times = np.array([UTCDateTime(t) for t in catalog_df['time'].values])
    magnitudes = catalog_df['magnitude'].values

    start_time = np.min(source_times)
    end_time = np.max(source_times)
    relative_times = source_times - start_time
    ax.scatter(relative_times, magnitudes, **kwargs)

    linear_xticks = np.linspace(0, end_time - start_time, 5)
    xtick_labels = [str((start_time + linear_xticks[i]).date) for i in range(5)]
    xticks = [
        UTCDateTime((start_time + linear_xticks[i]).date) - start_time
        for i in range(5)
    ]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Magnitude')
    
    return ax


def _plot_ct(catalog_df, ax=None, **kwargs):
    source_times = np.array([UTCDateTime(t) for t in catalog_df['time'].values])
    start_time = np.min(source_times)
    end_time = np.max(source_times)
    relative_times = source_times - start_time

    relative_times = np.array(sorted(relative_times))
    cumulative_counts = np.arange(len(relative_times)) + 1

    plot_indexs = np.linspace(0, len(relative_times) - 1, 1000).astype(int)

    ax.plot(
        relative_times[plot_indexs],
        cumulative_counts[plot_indexs],
        **kwargs
    )

    linear_xticks = np.linspace(0, end_time - start_time, 5)
    xtick_labels = [str((start_time + linear_xticks[i]).date) for i in range(5)]
    xticks = [
        UTCDateTime((start_time + linear_xticks[i]).date) - start_time
        for i in range(5)
    ]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)

    ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')

    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative count')

    return ax


def _plot_mc(catalog_df, ax=None, uniform=True, precision=1, **kwargs):
    magnitudes = catalog_df['magnitude'].values
    magnitudes = np.around(magnitudes, precision)
    if uniform:
        plot_m = np.arange(np.min(magnitudes), np.max(magnitudes) + 0.1, 0.1)
        plot_m = np.around(plot_m, precision)
    else:
        plot_m = sorted(np.unique(magnitudes))

    cumulative_count_list = []
    for m in plot_m:
        cumulative_count = len(np.where(magnitudes >= m)[0])
        cumulative_count_list.append([m, cumulative_count])
    cumulative_count_list = np.array(cumulative_count_list)
    ax.scatter(
        cumulative_count_list[:, 0],
        cumulative_count_list[:, 1],
        **kwargs
    )

    raw_ylim = ax.get_ylim()
    ax.set_ylim(1, raw_ylim[1])
    ax.set_yscale('log')

    ax.set_xlabel('Magnitude')
    ax.set_ylabel('Cumulative count')

    return ax


def _foreshock_mainshock(catalog_df, output_file=None):
    if output_file is not None:
        if os.path.exists(output_file):
            os.remove(output_file)
        with open(output_file, 'a+') as f:
            f.write('foreshock_index,large_shock_indexs\n')

    # catalog_df['time'] = [UTCDateTime(t) for t in catalog_df['time'].values]
    sorted_catalog_df = catalog_df.sort_values('time')

    if output_file is None:
        event_pairs = []
    for i in tqdm(range(len(sorted_catalog_df)), total=len(sorted_catalog_df)):
        foreshock_row = sorted_catalog_df.iloc[i]
        foreshock_index = sorted_catalog_df.index[i]
        foreshock_time = foreshock_row['time']
        foreshock_loc = [foreshock_row['latitude'], foreshock_row['longitude']]

        subsequent_df = sorted_catalog_df.iloc[i+1:]

        large_shock_df = subsequent_df[
            (subsequent_df['magnitude'] > foreshock_row['magnitude'])
            ]

        near_shock_df = large_shock_df[
            (large_shock_df['latitude'] >= foreshock_loc[0] - 0.3) &
            (large_shock_df['latitude'] <= foreshock_loc[0] + 0.3) &
            (large_shock_df['longitude'] >= foreshock_loc[1] - 0.3) &
            (large_shock_df['longitude'] <= foreshock_loc[1] + 0.3)
        ]

        real_time = str(UTCDateTime(foreshock_time) + 86400 * 30)
        potential_mainshock_df = near_shock_df[near_shock_df['time'] <= real_time]

        potential_mainshock_indexs = potential_mainshock_df.index.tolist()

        if output_file is None:
            event_pairs.append([foreshock_index, potential_mainshock_indexs])
        else:
            df_line = pd.DataFrame(columns=['foreshock_index', 'large_shock_indexs'])
            df_line = df_line.append([{
                'foreshock_index': foreshock_index,
                'large_shock_indexs': potential_mainshock_indexs
            }])
            df_line.to_csv(output_file, mode='a', header=False, index=False)

            del df_line

        del foreshock_row
        del subsequent_df
        del large_shock_df
        del near_shock_df
        del potential_mainshock_df
        del potential_mainshock_indexs

    if output_file is None:
        output_df = pd.DataFrame(event_pairs, columns=['foreshock_index', 'large_shock_indexs'])
        ret = output_df
    else:
        ret = None

    return ret


class Catalog:
    def __init__(self, csv_catalog_file=None, catalog_df=None):
        if csv_catalog_file is not None:
            self.csv_file = csv_catalog_file
        else:
            self.csv_file = None
        if catalog_df is not None:
            self.df = catalog_df
        else:
            self.df = None

    def map(self, mag_min=None, lat_lim=None, lon_lim=None, **kwargs):
        if self.csv_file is None:
            catalog_df = self.df
        else:
            catalog_df = pd.read_csv(self.csv_file)
        fig = _plot_map(catalog_df, mag_min=mag_min, lat_lim=lat_lim, lon_lim=lon_lim, **kwargs)
   
        return fig

    def cut(self, event_row=None, time_window=None, space_window=None, sorted=False):
        if self.csv_file is None:
            catalog_df = self.df
        else:
            catalog_df = pd.read_csv(self.csv_file)

        tar_df = _cut_catalog(
            catalog_df,
            event_row,
            time_window=time_window,
            space_window=space_window,
            sorted=sorted
        )

        return tar_df

    def mt(self, ax=None, **kwargs):
        if self.csv_file is None:
            catalog_df = self.df
        else:
            catalog_df = pd.read_csv(self.csv_file)
        if ax is None:
            fig, ax = plt.subplots()
        ax = _plot_mt(catalog_df, ax=ax, **kwargs)
        
        return ax

    def ct(self, ax=None, **kwargs):
        if self.csv_file is None:
            catalog_df = self.df
        else:
            catalog_df = pd.read_csv(self.csv_file)
        if ax is None:
            fig, ax = plt.subplots()
        ax = _plot_ct(catalog_df, ax=ax, **kwargs)

        return ax

    def mc(self, ax=None, uniform=True, precision=1, **kwargs):
        if self.csv_file is None:
            catalog_df = self.df
        else:
            catalog_df = pd.read_csv(self.csv_file)
        if ax is None:
            fig, ax = plt.subplots()
        ax = _plot_mc(catalog_df, ax=ax, uniform=uniform, precision=precision, **kwargs)

        return ax

    def foreshock_mainshock(self, output_file=None):
        if self.csv_file is None:
            catalog_df = self.df
        else:
            catalog_df = pd.read_csv(self.csv_file)

        output_df = _foreshock_mainshock(catalog_df, output_file=output_file)

        return output_df
