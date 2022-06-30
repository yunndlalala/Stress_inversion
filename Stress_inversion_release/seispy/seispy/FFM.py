#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: yunnaidan
@time: 2020/01/08
@file: FFM.py
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D


class FFM(object):
    def __init__(self, FFM_df, row_n=12, column_n=21):
        """
        :param FFM_df: pandas.DataFrame with columns
        ['lon', 'lat', 'dep', 'lon1', 'lat1', 'dep1', 'lon2', 'lat2', 'dep2', 'lon3', 'lat3', 'dep3',
        'lon4', 'lat4', 'dep4', 'strike', 'dip', 'rake', 'slip_all']
        """
        self.FFM_df = FFM_df
        self.row_n = row_n
        self.column_n = column_n

    def plot_slip_surface(self,
                          ax=None,
                          patch=False,
                          patch_linewidth=0.5,
                          patch_color='k',
                          point=True,
                          point_size=0.1,
                          point_color='k',
                          item_filling=False,
                          filling_cb='coolwarm',
                          filling_alpha=1.0,
                          item_arrow=True,
                          zoom=1.0,
                          arrow_color='k',
                          arrow_linewidth=0.001,
                          arrow_head_width_r=0.001,
                          arrow_head_length_r=0.001
                          ):

        handles = surface_plotting(self.FFM_df,
                                   'slip_all',
                                   ax=ax,
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
                                   arrow_head_length_r=arrow_head_length_r,
                                   )
        return handles

    def plot_slip_plane(self,
                        ax=None,
                        on_depth=False,
                        zoom=1.0,
                        color='gray',
                        label='',
                        arrow_linewidth=0.01,
                        arrow_head_length_r=0.01,
                        arrow_head_width_r=0.01,
                        ):
        ax = plane_plotting(self.FFM_df,
                            row_n=self.row_n,
                            on_depth=on_depth,
                            by_rake=True,
                            rake_item_name='slip_all',
                            strike_item_name=None,
                            dip_item_name=None,
                            ax=ax,
                            zoom=zoom,
                            color=color,
                            label=label,
                            arrow_linewidth=arrow_linewidth,
                            arrow_head_length_r=arrow_head_length_r,
                            arrow_head_width_r=arrow_head_width_r,
                            )
        return ax

    def plot_slip_3D(self,
                     ax=None,
                     patch=True,
                     patch_color='k',
                     patch_linewidth=0.1,
                     point=True,
                     point_size=0.1,
                     point_color='k',
                     item_filling=False,
                     item_arrow=True,
                     zoom=1.0,
                     arrow_color='k',
                     arrow_linewidth=1,
                     arrow_ratio=0.3,
                     grid=False
                     ):

        ax = axes3D_plotting(self.FFM_df,
                             'slip_all',
                             ax=ax,
                             patch=patch,
                             patch_color=patch_color,
                             patch_linewidth=patch_linewidth,
                             point=point,
                             point_size=point_size,
                             point_color=point_color,
                             item_filling=item_filling,
                             item_arrow=item_arrow,
                             zoom=zoom,
                             arrow_color=arrow_color,
                             arrow_linewidth=arrow_linewidth,
                             arrow_ratio=arrow_ratio,
                             grid=grid)

        return ax


def project_to_surface(item, strike, dip, rake):
    strike_rad = np.deg2rad(strike)
    dip_rad = np.deg2rad(dip)
    rake_rad = np.deg2rad(rake)

    dx = np.cos(rake_rad) * np.sin(strike_rad) \
        - np.sin(rake_rad) * np.cos(dip_rad) * np.cos(strike_rad)
    dy = np.cos(rake_rad) * np.cos(strike_rad) \
        + np.sin(rake_rad) * np.cos(dip_rad) * np.sin(strike_rad)
    return item * dx, item * dy


def surface_plotting(FFM_df,
                     item_name,
                     ax=None,
                     patch=False,
                     patch_linewidth=0.5,
                     patch_color='k',
                     point=True,
                     point_size=0.1,
                     point_color='k',
                     item_filling=False,
                     filling_cb='Greys',
                     filling_vmin=None,
                     filling_vmax=None,
                     filling_alpha=1.0,
                     item_arrow=True,
                     zoom=1.0,
                     arrow_color='k',
                     arrow_linewidth=0.001,
                     arrow_head_width_r=0.001,
                     arrow_head_length_r=0.001
                     ):

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    result = [ax]

    for row_i, row in FFM_df.iterrows():
        if patch or item_filling:
            x = [
                row['lon1'],
                row['lon2'],
                row['lon3'],
                row['lon4'],
                row['lon1']]
            y = [
                row['lat1'],
                row['lat2'],
                row['lat3'],
                row['lat4'],
                row['lat1']]
        if patch:
            ax.plot(x, y, '-', color=patch_color, linewidth=patch_linewidth)
        if item_filling:
            # cmap = plt.cm.Paired
            filling_cb = plt.get_cmap(filling_cb)
            if filling_vmin is None:
                filling_vmin = np.min(FFM_df[item_name])
            if filling_vmax is None:
                filling_vmax = np.max(FFM_df[item_name])

            norm = matplotlib.colors.Normalize(
                vmin=filling_vmin,
                vmax=filling_vmax)
            p = ax.fill(
                x,
                y,
                color=filling_cb(
                    norm(
                        row[item_name])),
                alpha=filling_alpha)

    for row_i, row in FFM_df.iterrows():
        arrow_b_x = row['lon']
        arrow_b_y = row['lat']
        if point:
            ax.plot(
                arrow_b_x,
                arrow_b_y,
                '.',
                color=point_color,
                linewidth=point_size)
        if item_arrow and row[item_name] != 0.0:
            item_zoomed = row[item_name] / zoom
            surface_x, surface_y = project_to_surface(
                item_zoomed, row['strike'], row['dip'], row['rake'])
            head_length = arrow_head_length_r * \
                          np.sqrt(surface_x ** 2 + surface_y ** 2)
            head_width = head_length * arrow_head_width_r
            ax.arrow(
                arrow_b_x,
                arrow_b_y,
                surface_x,
                surface_y,
                width=arrow_linewidth,
                head_width=head_width,
                head_length=head_length,
                overhang=0.3,
                color=arrow_color)
    if item_filling:
        cmap = plt.get_cmap(filling_cb)
        # Get the colormap colors
        my_cmap = cmap(np.arange(cmap.N))
        # Define the alphas in the range from 0 to 1
        alphas = filling_alpha
        # Define the background as white
        BG = np.asarray([1., 1., 1., ])
        # Mix the colors with the background
        for i in range(cmap.N):
            my_cmap[i, :-1] = my_cmap[i, :-1] * alphas + BG * (1. - alphas)
        # Create new colormap which mimics the alpha values
        my_cmap = ListedColormap(my_cmap)

        sm = plt.cm.ScalarMappable(
            cmap=my_cmap,
            norm=norm
        )
        result.append(sm)

    return tuple(result) if len(result) > 1 else result[0]


def decompose_on_plane(item, rake):
    rake_rad = np.deg2rad(rake)
    dx = np.cos(rake_rad)
    dy = np.sin(rake_rad)
    return item * dx, item * dy


def plane_plotting(FFM_df,
                   row_n=12,
                   on_depth=False,
                   by_rake=True,
                   rake_item_name=None,
                   strike_item_name=None,
                   dip_item_name=None,
                   ax=None,
                   zoom=1.0,
                   color='gray',
                   label='',
                   arrow_linewidth=0.001,
                   arrow_head_length_r=0.001,
                   arrow_head_width_r=0.001,
                   ):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    for row_i, row in FFM_df.iterrows():
        # row_i=161
        # row=FFM_df.iloc[161]

        arrow_b_x = int((row_i + 0.1) / row_n)
        arrow_b_y = row_n - 1 - row_i % row_n
        if on_depth:
            arrow_b_y = row['dep']

        if by_rake:
            if row[rake_item_name] != 0.0:
                item_value = row[rake_item_name] / zoom
                rake_value = row['rake']
                plane_x, plane_y = decompose_on_plane(item_value, rake_value)
                if on_depth:
                    plane_y = -plane_y
                # The xlim and ylim cannot auto-adjust when only use ax.arrow,
                # so we add the ax.plot step.
                ax.plot(arrow_b_x, arrow_b_y, color)
                head_length = arrow_head_length_r * \
                    np.sqrt(plane_x ** 2 + plane_y ** 2)
                head_width = head_length * arrow_head_width_r
                ax.arrow(
                    arrow_b_x,
                    arrow_b_y,
                    plane_x,
                    plane_y,
                    width=arrow_linewidth,
                    head_width=head_width,
                    head_length=head_length,
                    length_includes_head=True,
                    color=color)

        else:
            if row[strike_item_name] != 0.0 or \
                    row[dip_item_name] != 0.0:
                strike_item_value = row[strike_item_name] / zoom
                dip_item_value = row[dip_item_name] / zoom
                if on_depth:
                    dip_item_value = -dip_item_value
                # The xlim and ylim cannot auto-adjust when only use ax.arrow,
                # so we add the ax.plot step.
                ax.plot(arrow_b_x, arrow_b_y, color)

                head_length = arrow_head_length_r * \
                    np.sqrt(strike_item_value ** 2 + dip_item_value ** 2)
                head_width = head_length * arrow_head_width_r
                ax.arrow(
                    arrow_b_x,
                    arrow_b_y,
                    strike_item_value,
                    -dip_item_value,
                    width=arrow_linewidth,
                    head_width=head_width,
                    head_length=head_length,
                    length_includes_head=True,
                    color=color)

    ax.plot(arrow_b_x, arrow_b_y, color, label=label)

    return ax


def decompose_3D(item, strike, dip, rake):
    strike_rad = np.deg2rad(strike)
    dip_rad = np.deg2rad(dip)
    rake_rad = np.deg2rad(rake)

    dx = np.cos(rake_rad) * np.sin(strike_rad) \
        - np.sin(rake_rad) * np.cos(dip_rad) * np.cos(strike_rad)
    dy = np.cos(rake_rad) * np.cos(strike_rad) \
        + np.sin(rake_rad) * np.cos(dip_rad) * np.sin(strike_rad)
    dz = np.sin(rake_rad) * np.sin(dip_rad)
    return item * dx, item * dy, item * dz


def axes3D_plotting(FFM_df,
                    item_name,
                    ax=None,
                    patch=True,
                    patch_color='k',
                    patch_linewidth=0.1,
                    point=True,
                    point_size=0.1,
                    point_color='k',
                    item_filling=False,
                    item_arrow=True,
                    zoom=1.0,
                    arrow_color='k',
                    arrow_linewidth=1,
                    arrow_ratio=0.3,
                    grid=False):

    if ax is None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    for row_i, row in FFM_df.iterrows():
        x = np.array([
            row['lon1'],
            row['lon2'],
            row['lon3'],
            row['lon4'],
            row['lon1']])
        y = np.array([
            row['lat1'],
            row['lat2'],
            row['lat3'],
            row['lat4'],
            row['lat1']])
        z = -1 * np.array([
            row['dep1'],
            row['dep2'],
            row['dep3'],
            row['dep4'],
            row['dep1']])
        arrow_b_x = row['lon']
        arrow_b_y = row['lat']
        arrow_b_z = -1 * row['dep']

        if patch:
            ax.plot3D(
                x,
                y,
                z,
                '-',
                color=patch_color,
                linewidth=patch_linewidth)
        if point:
            ax.scatter3D(
                arrow_b_x,
                arrow_b_y,
                arrow_b_z,
                '.',
                c=point_color,
                s=point_size)
        if item_filling:
            cmap = plt.get_cmap('coolwarm')
            # cmap = cmap_from_cptcity_url('ncl/precip2_17lev.cpt')
            norm = matplotlib.colors.Normalize(
                vmin=np.min(FFM_df[item_name]),
                vmax=np.max(FFM_df[item_name]))
            X = np.array([[row['lon3'], row['lon2']],
                          [row['lon4'], row['lon1']]])
            Y = np.array([[row['lat3'], row['lat2']],
                          [row['lat4'], row['lat1']]])
            Z = np.array([[-1 * row['dep3'], -1 * row['dep2']],
                          [-1 * row['dep4'], -1 * row['dep1']]])
            ax.plot_surface(X, Y, Z, color=cmap(
                norm(row[item_name])), alpha=0.5)
        if item_arrow:
            item_zoomed = row[item_name] / zoom
            dx_3D, dy_3D, dz_3D = decompose_3D(
                item_zoomed, row['strike'], row['dip'], row['rake'])
            ax.quiver(
                arrow_b_x,
                arrow_b_y,
                arrow_b_z,
                dx_3D,
                dy_3D,
                dz_3D,
                linewidth=arrow_linewidth,
                arrow_length_ratio=arrow_ratio,
                color=arrow_color)
    if item_filling:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm)
    ax.grid(grid)
    scaling = np.array([getattr(ax, 'get_{}lim'.format(dim))()
                        for dim in 'xy'])
    ax.set_xlim3d([np.min(scaling), np.max(scaling)])
    ax.set_ylim3d([np.min(scaling), np.max(scaling)])
    return ax
