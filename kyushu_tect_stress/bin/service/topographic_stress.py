#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: yunnaidan
@time: 2020/12/02
@file: land_seafloor_comparison.py
"""
import os
import numpy as np
import pandas as pd
from scipy import io
import matplotlib.pyplot as plt

from stress import topography


def from_ffm(
        ffm_file=None,
        topography_file=None,
        topo_lon=None,
        topo_lat=None,
        kernel_figure_path=None,
        tensor_figure_path=None,
        output_file=None,
        series=0,
        topo_tensor0_0=None,
        tect_tensor_0=None,
        reference_depth=0.1,
        density=2.6e3,
        kernel=False,
        show=False
):
    ffm_df = pd.read_csv(ffm_file)
    tar_depth = sorted(list(set(ffm_df['dep'])))
    component_name = ['xx', 'xy', 'xz', 'yy', 'yz', 'zz']
    ffm_tensor = np.zeros([len(ffm_df), 6])

    topography_data = np.load(topography_file)
    topography_data = topography_data.transpose(2, 1, 0)

    if series in [1, -1]:
        if topo_tensor0_0 is None:
            lon_mesh, lat_mesh, topo_tensor0_0 = topography.gen_topo_stress_tensor(
                topography_data, reference_depth, series=0, density=density)
            # topography.tensor_plotting(lon_mesh, lat_mesh, topo_tensor0_0,
            #                            [topo_lon[0], topo_lon[1]],
            #                            [topo_lat[0], topo_lat[1]])
            # plt.show()
    elif series == 0:
        topo_tensor0_0 = None
    else:
        raise ValueError

    for depth in tar_depth:  # kilometer
        print(depth)

        # Calculate the zero-order

        if series in [0, -1]:

            if kernel:
                lon_mesh, lat_mesh, kernel0 = topography.gen_kernel(
                    topography_data, depth, series=0)
                topography.tensor_plotting(lon_mesh, lat_mesh, kernel0,
                                           [topo_lon[0], topo_lon[1]],
                                           [topo_lat[0], topo_lat[1]])
                if kernel_figure_path is not None:
                    kernel_figure_file = os.path.join(kernel_figure_path,
                                                      'depth.' + '%.4f' % depth + '.png')
                    plt.savefig(kernel_figure_file)
                plt.show()

            lon_mesh, lat_mesh, topo_tensor0 = topography.gen_topo_stress_tensor(
                topography_data, depth, series=0, density=density)

            if show:
                topography.tensor_plotting(lon_mesh, lat_mesh, topo_tensor0,
                                           [topo_lon[0], topo_lon[1]],
                                           [topo_lat[0], topo_lat[1]])
                if tensor_figure_path is not None:
                    tensor_figure_file = os.path.join(tensor_figure_path,
                                                      'depth.' + '%.4f' % depth + '.png')
                    plt.savefig(tensor_figure_file)
                plt.show()

        # Calculate the first-order

        if series in [1, -1]:

            # x_gradient, y_gradient = topography.topo_gradient(
            #     topography_data,
            #     show=True,
            #     lon_range=[topo_lon[0], topo_lon[1]],
            #     lat_range=[topo_lat[0], topo_lat[1]]
            # )
            #
            # x_force, y_force = topography.horizontal_force(
            #     topography_data,
            #     topo_tensor0_0,
            #     tect_tensor_0,
            #     reference_depth=reference_depth,
            #     show=True,
            #     lon_range=[topo_lon[0], topo_lon[1]],
            #     lat_range=[topo_lat[0], topo_lat[1]]
            # )

            if kernel:
                lon_mesh, lat_mesh, kernel1 = topography.gen_kernel(
                    topography_data, depth, reference_depth=reference_depth, series=1)
                topography.tensor_plotting(lon_mesh, lat_mesh, kernel1[0],
                                           [topo_lon[0], topo_lon[1]],
                                           [topo_lat[0], topo_lat[1]])
                if kernel_figure_path is not None:
                    kernel_figure_file = os.path.join(kernel_figure_path,
                                                      'depth.' + '%.4f' % depth + '_x_gradient.png')
                    plt.savefig(kernel_figure_file)
                topography.tensor_plotting(lon_mesh, lat_mesh, kernel1[1],
                                           [topo_lon[0], topo_lon[1]],
                                           [topo_lat[0], topo_lat[1]])
                if kernel_figure_path is not None:
                    kernel_figure_file = os.path.join(kernel_figure_path,
                                                      'depth.' + '%.4f' % depth + '_y_gradient.png')
                    plt.savefig(kernel_figure_file)
                plt.show()

            lon_mesh, lat_mesh, topo_tensor1 = topography.gen_topo_stress_tensor(
                topography_data,
                depth,
                reference_depth=reference_depth,
                series=1,
                tect_tensor_0=tect_tensor_0,
                topo_tensor0_0=topo_tensor0_0,
                density=density
            )

            if show:
                topography.tensor_plotting(lon_mesh, lat_mesh, topo_tensor1,
                                           [topo_lon[0], topo_lon[1]],
                                           [topo_lat[0], topo_lat[1]])
                if tensor_figure_path is not None:
                    tensor_figure_file = os.path.join(tensor_figure_path,
                                                      'depth.' + '%.4f' % depth + '_gradient.png')
                    plt.savefig(tensor_figure_file)
                plt.show()

        if series == 0:
            output_tensor = topo_tensor0
        elif series == 1:
            output_tensor = topo_tensor1
        elif series == -1:
            output_tensor = topo_tensor0[1:-1, 1:-1, :] + topo_tensor1
        else:
            raise ValueError

        # Interp and save result
        tar_index = np.where(ffm_df['dep'] == depth)
        tar_lon = ffm_df['lon'].values[tar_index]
        tar_lat = ffm_df['lat'].values[tar_index]
        tar_tensor = topography.interp(
            lon_mesh, lat_mesh, output_tensor, tar_lon, tar_lat)
        for i in range(6):
            ffm_tensor[tar_index, i] = tar_tensor[:, i]

    for j in range(6):
        ffm_df[component_name[j]] = ffm_tensor[:, j]

    ffm_df.to_csv(output_file, index=False)

    return None
