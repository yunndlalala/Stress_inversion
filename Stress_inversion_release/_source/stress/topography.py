#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: yunnaidan
@time: 2020/04/09
@file: topography.py
"""
import numpy as np
from scipy import signal
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from geopy.distance import great_circle


def cut_topo(
        lon_range,
        lat_range,
        topography_data
):
    """
    Cut target space range of the topography.
    Args:
        lon_range: range of longitude
        lat_range: range of latitude
        topography_data: [m, n, 3(longitude, latitude, height)] where m and n are the point num
        along the longitude and latitude

    Returns: [m', n', 3]

    """
    lon_list = topography_data[:, 0, 0]
    lat_list = topography_data[0, :, 1]
    lon_i = np.where(
        (lon_list >= lon_range[0]) & (
            lon_list <= lon_range[1]))[0]
    lat_i = np.where(
        (lat_list >= lat_range[0]) & (
            lat_list <= lat_range[1]))[0]
    tar_topography_data = topography_data[lon_i[0]:lon_i[-1] + 1, lat_i[0]:lat_i[-1] + 1, :]

    return tar_topography_data


def topo_gradient(
        topography_data
):
    """
    Gradients of the topography data along the longitude and latitude directions.
    Args:
        topography_data: [m, n, 3(longitude, latitude, height)]

    Returns: (gradients along longitude and latitude)

    """
    x_array = topography_data[:, :, 0].T
    y_array = topography_data[:, :, 1].T
    h_array = topography_data[:, :, 2].T

    x_bin = (x_array[:, 1:] - x_array[:, :-1]) * 110 * 1000 * np.cos(np.deg2rad(30))
    y_bin = (y_array[1:, :] - y_array[:-1, :]) * 110 * 1000

    x_gradient = (h_array[:, 2:] - h_array[:, :-2]) / (x_bin[:, :-1] + x_bin[:, 1:])
    y_gradient = (h_array[2:, :] - h_array[:-2, :]) / (y_bin[:-1, :] + y_bin[1:, :])

    return x_gradient, y_gradient


def vertical_loading_kernel(
        x,
        y,
        z,
        poisson
):
    """
    Args:
        x, y, z: locations
        poisson: poisson ratio

    Returns: Solution of the Boussinesq's problem.

    """

    r = np.sqrt(pow(x, 2.0) + pow(y, 2.0) + pow(z, 2.0))
    ans = 1.0 / (2.0 * np.pi)

    xx = ans * (-3.0 * pow(x, 2.0) * z / pow(r, 5.0)
                + (1.0 - 2.0 * poisson)
                * (((pow(y, 2.0) + pow(z, 2.0)) / (pow(r, 3.0) * (-z + r)))
                   + (z / pow(r, 3.0))
                   - (pow(x, 2.0) / (pow(r, 2.0) * pow((-z + r), 2.0)))))

    yy = ans * (-3.0 * pow(y, 2.0) * z / pow(r, 5.0)
                + (1.0 - 2.0 * poisson)
                * (((pow(x, 2.0) + pow(z, 2.0)) / (pow(r, 3.0) * (-z + r)))
                   + (z / pow(r, 3.0))
                   - (pow(y, 2.0) / (pow(r, 2.0) * pow((-z + r), 2.0)))))

    zz = -ans * 3.0 * pow(z, 3.0) / pow(r, 5.0)

    xy = ans * (-(3.0 * x * y * z / pow(r, 5.0)) - ((1.0 - 2.0 * poisson)
                                                    * (x * y * (-z + 2.0 * r) / (pow(r, 3.0) * pow((-z + r), 2.0)))))

    yz = -ans * 3.0 * pow(z, 2.0) * y / pow(r, 5.0)

    xz = -ans * 3.0 * pow(z, 2.0) * x / pow(r, 5.0)

    return xx, xy, xz, yy, yz, zz


def horizontal_loading_kernel(
        x,
        y,
        z,
        poisson
):
    """
    Args:
        x, y, z: locations
        poisson: poisson ratio

    Returns: Solution of the Cerruti's problem.

    """
    r = np.sqrt(pow(x, 2.0) + pow(y, 2.0) + pow(z, 2.0))
    ans = 1.0 / (2.0 * np.pi * pow(r, 3.0))

    xx = ans * x * (
            3 * pow(x, 2.0) / pow(r, 2.0) -
            ((1.0 - 2.0 * poisson) / pow((-z + r), 2.0)) *
                (pow(r, 2.0) - pow(y, 2.0) - 2.0 * r * pow(y, 2.0) / (r - z))
    )

    yy = ans * x * (
        3 * pow(y, 2.0) / pow(r, 2.0) -
        ((1.0 - 2.0 * poisson) / pow((-z + r), 2.0)) *
            (3.0 * pow(r, 2.0) - pow(x, 2.0) - 2.0 * r * pow(x, 2.0) / (r - z))
    )

    zz = ans * 3.0 * x * pow(z, 2.0) / pow(r, 2.0)

    xy = ans * y * (
        3.0 * pow(x, 2.0) / pow(r, 2.0) +
        (1.0 - 2.0 * poisson) / pow((-z + r), 2.0) *
            (pow(r, 2.0) - pow(x, 2.0) - 2.0 * r * pow(x, 2.0) / (r - z))
    )

    yz = ans * 3.0 * x * y * z / pow(r, 2.0)

    xz = ans * 3.0 * pow(x, 2.0) * z / pow(r, 2.0)

    return xx, xy, xz, yy, yz, zz


def gen_kernel(
        topography,
        depth,
        reference_depth=0.0,
        series=0,
        poisson=0.3
):
    """
    Args:
        topography: topography data
        depth: target depth of the kernel
        reference_depth: the reference depth used to replace depth = 0
        series: '0' calculate the stress caused by vertical loading of the topography; '1' calculate the stress
        caused by the coupling of the topography and the half-space; '-1' calculate the summary of '0' and '-1'.
        poisson: poisson ratio

    Returns: Green's function

    """

    depth = depth - reference_depth

    lon_mesh = topography[:, :, 0].T
    lat_mesh = topography[:, :, 1].T

    lon = lon_mesh[0, :]
    lat = lat_mesh[:, 0]

    reference = [lat[0], lon[0]]
    xary = np.array(
        [great_circle(reference, [lat[0], lon_i]).km for lon_i in lon])
    yary = np.array(
        [great_circle(reference, [lat_i, lon[0]]).km for lat_i in lat])
    xary = xary - np.mean(xary)
    yary = yary - np.mean(yary)
    xinc = xary[1] - xary[0]
    yinc = yary[1] - yary[0]
    X, Y = np.meshgrid(xary, yary)

    if series == 0:
        xx, xy, xz, yy, yz, zz = vertical_loading_kernel(X, Y, -depth, poisson)
        kernel = np.array([xx, xy, xz, yy, yz, zz])
        kernel = kernel.transpose(1, 2, 0)
        kernel = kernel * xinc * yinc
    elif series == 1:
        xx_x, xy_x, xz_x, yy_x, yz_x, zz_x = horizontal_loading_kernel(X, Y, -depth, poisson)
        yy_y, xy_y, yz_y, xx_y, xz_y, zz_y = horizontal_loading_kernel(Y, X, -depth, poisson)

        kernel_x = np.array([xx_x, xy_x, xz_x, yy_x, yz_x, zz_x])
        kernel_y = np.array([xx_y, xy_y, xz_y, yy_y, yz_y, zz_y])
        kernel_x = kernel_x.transpose(1, 2, 0) * xinc * yinc
        kernel_y = kernel_y.transpose(1, 2, 0) * xinc * yinc
        kernel = np.array([kernel_x, kernel_y])
    else:
        raise ValueError

    return lon_mesh, lat_mesh, kernel


def horizontal_force(
        topography_data,
        topo_tensor0_0,
        tect_tensor_0,
        reference_depth=0.1,
        density=2.6 * 1e3,
        g=9.8
):
    """

    Args:
        topography_data: topography data
        topo_tensor0_0: the stress result due to the vertical loading at the reference depth
        tect_tensor_0: the tectonic stress tensor at the reference depth
        reference_depth: the reference depth used to replace depth = 0
        density:
        g:

    Returns: horizontal loading due to the coupling of the topography and half-space

    """
    topo_xx0_0 = topo_tensor0_0[:, :, 0]
    topo_xy0_0 = topo_tensor0_0[:, :, 1]
    topo_yy0_0 = topo_tensor0_0[:, :, 3]

    tect_xx_0 = tect_tensor_0[0, 0]
    tect_xy_0 = tect_tensor_0[0, 1]
    tect_yy_0 = tect_tensor_0[1, 1]

    h_array = topography_data[:, :, 2].T + reference_depth
    x_gradient, y_gradient = topo_gradient(topography_data)

    x_force = -(-density * g * h_array + topo_xx0_0 + tect_xx_0)[1:-1, 1:-1] * x_gradient[1:-1, :] - \
              (topo_xy0_0 + tect_xy_0)[1:-1, 1:-1] * y_gradient[:, 1:-1]
    y_force = -(-density * g * h_array + topo_yy0_0 + tect_yy_0)[1:-1, 1:-1] * y_gradient[:, 1:-1] - \
              (topo_xy0_0 + tect_xy_0)[1:-1, 1:-1] * x_gradient[1:-1, :]

    return x_force, y_force


def gen_topo_stress_tensor(
        topography_data,
        depth,
        reference_depth=0.0,
        series=0,
        tect_tensor_0=None,
        topo_tensor0_0=None,
        poisson=0.3,
        density=2.6 * 1e3
):
    """

    Args:
        topography_data: the topography data
        depth: target depth to calculate the topographic stress tensor
        reference_depth: the reference depth used to replace depth = 0
        series: '0' calculate the stress caused by vertical loading of the topography; '1' calculate the stress
        caused by the coupling of the topography and the half-space; '-1' calculate the summary of '0' and '-1'.
        tect_tensor_0: the tectonic stress tensor at the reference depth
        topo_tensor0_0: the stress result due to the vertical loading at the reference depth
        poisson: poisson ratio
        density:

    Returns: topographic stress tensor at a target depth

    """
    if series == 0:
        P = topography_data[:, :, 2] * density * 9.8
        P = P.T
        lon_mesh, lat_mesh, kernel = gen_kernel(
            topography_data,
            depth,
            reference_depth=0.0,
            poisson=poisson
        )

        topo_tensor = np.zeros(np.shape(kernel))
        for i in range(6):
            topo_tensor[:, :, i] = signal.fftconvolve(
                P, kernel[:, :, i], mode='same')
    elif series == 1:
        x_force, y_force = horizontal_force(
            topography_data,
            topo_tensor0_0,
            tect_tensor_0,
            reference_depth=reference_depth,
            density=density
        )
        lon_mesh, lat_mesh, kernel_gradient = gen_kernel(
            topography_data,
            depth,
            reference_depth=reference_depth,
            series=1
        )
        x_kernel, y_kernel = kernel_gradient

        topo_tensor = np.zeros(np.shape(x_kernel[1:-1, 1:-1, :]))
        for i in range(6):
            topo_tensor[:, :, i] = \
                signal.fftconvolve(x_force, x_kernel[1:-1, 1:-1, i], mode='same') + \
                signal.fftconvolve(y_force, y_kernel[1:-1, 1:-1, i], mode='same')

        lon_mesh = lon_mesh[1:-1, 1:-1]
        lat_mesh = lat_mesh[1:-1, 1:-1]

    else:
        raise ValueError

    return lon_mesh, lat_mesh, topo_tensor


def interp(lon_mesh, lat_mesh, tensor, tar_lon, tar_lat):
    """
    Interpolate a tensor to the target location.
    Args:
        lon_mesh: mesh-grid longitude of the topography data
        lat_mesh: mesh-grid latitude of the topography data
        tensor: a tensor needed to be interpolated
        tar_lon: longitude of the target location
        tar_lat: latitude of the target location

    Returns: Interpolated tensor at the target location

    """
    tar_tensor = np.zeros([len(tar_lon), 6])
    for i in range(6):
        interp_f = interp2d(lon_mesh[0, :], lat_mesh[:, 0], tensor[:, :, i])
        tar_tensor[:, i] = [interp_f(tar_lon[j], tar_lat[j])[0]
                            for j in range(len(tar_lon))]

    return tar_tensor


def plotting(topography, ax=None, **kwargs):
    """
    Plot the topography data.

    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    lon_mesh = topography[:, :, 0].T
    lat_mesh = topography[:, :, 1].T
    height_mesh = topography[:, :, 2].T

    lon_mesh = lon_mesh - (lon_mesh[0, 1] - lon_mesh[0, 0]) / 2.0
    lon_right = lon_mesh[:, -1] + (lon_mesh[0, -1] - lon_mesh[0, -2])
    lon_mesh = np.hstack((lon_mesh, lon_right.reshape([len(lon_right), 1])))
    lon_mesh = np.vstack((lon_mesh, lon_mesh[0, :]))

    lat_mesh = lat_mesh - (lat_mesh[1, 0] - lat_mesh[0, 0]) / 2.0
    lat_up = lat_mesh[-1, :] + (lat_mesh[-1, 0] - lat_mesh[-2, 0])
    lat_mesh = np.vstack((lat_mesh, lat_up))
    lat_mesh = np.hstack(
        (lat_mesh, lat_mesh[:, 0].reshape([len(lat_mesh[:, 0]), 1])))

    p = ax.pcolor(lon_mesh, lat_mesh, height_mesh, **kwargs)
    return ax, p


def tensor_plotting(lon_mesh, lat_mesh, tensor, lon_range, lat_range):
    """
    Plot the tensor.

    """
    fig = plt.figure(figsize=[7, 6])
    ax_loc = [[0.03, 0.70, 0.46, 0.25],
              [0.40, 0.70, 0.46, 0.25],
              [0.03, 0.39, 0.46, 0.25],
              [0.40, 0.39, 0.46, 0.25],
              [0.03, 0.08, 0.46, 0.25],
              [0.40, 0.08, 0.46, 0.25]]
    component_name = ['xx', 'xy', 'xz', 'yy', 'yz', 'zz']
    for i in range(len(component_name)):
        component = tensor[:, :, i]
        ax = fig.add_axes(ax_loc[i])
        bar_lim = np.max(abs(component))
        # p = ax.pcolormesh(lon_mesh, lat_mesh, component, cmap='coolwarm')
        p = ax.pcolormesh(
            lon_mesh,
            lat_mesh,
            component,
            cmap='coolwarm',
            vmin=-bar_lim,
            vmax=bar_lim)
        ax.set_title(component_name[i])
        ax.set_xlim(lon_range)
        ax.set_ylim(lat_range)
        ax.set_aspect('equal')
        cbar = plt.colorbar(p, ax=ax)
        cbar.set_label('Pa')
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()

        if i + 1 in [1, 2, 3, 4]:
            ax.set_xticks([])
        if i + 1 in [2, 4, 6]:
            ax.set_yticks([])
        if i + 1 in [1, 3, 5]:
            ax.set_ylabel('Latitude')
        if i + 1 in [5, 6]:
            ax.set_xlabel('Longitude')

    return fig
