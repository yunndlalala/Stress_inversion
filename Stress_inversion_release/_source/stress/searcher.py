#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: yunnaidan
@time: 2020/10/10
@file: searcher.py
"""
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
from matplotlib.ticker import LinearLocator

from stress import smoothing
from stress import utils as stu
from stress.likelihood import Likelihood


def _filter_none(para_list):
    not_none_para_list = []
    for para in para_list:
        if None not in para:
            not_none_para_list.append(para)
    return not_none_para_list


def _grid_prepare(para_info):
    para_n = len(para_info)
    paras = []
    steps = []
    indexs = []
    nums = []
    for para_i in range(para_n):
        min_value, max_value, num_value = para_info[para_i]
        para = np.linspace(
            min_value,
            max_value,
            num=num_value)
        step = (max_value - min_value) / (num_value - 1)
        mesh_index = np.linspace(
            min_value - 0.5 * step,
            max_value + 0.5 * step,
            num_value + 1)

        paras.append(para)
        steps.append(step)
        indexs.append(mesh_index)
        nums.append(num_value)

    if para_n == 2:
        iters = product(range(nums[0]), range(nums[1]))
        meshes = np.meshgrid(
            indexs[0], indexs[1])
        L_array = np.zeros([nums[1], nums[0]])
        saving_data = paras + meshes
    elif para_n == 3:
        iters = product(range(nums[0]), range(nums[1]), range(nums[2]))
        L_array = np.zeros([nums[2], nums[0], nums[1]])
        saving_data = [paras[2], paras[0], paras[1]]
    else:
        raise ValueError

    total_iters = np.prod(np.array(nums))
    norm_factor = np.prod(np.array(steps))

    return paras, iters, total_iters, L_array, norm_factor, saving_data


def _load_topo_tensors(
        direction=None,
        shmin=None,
        topo_tensors=None,
        density=None
):
    topo_file = '../../data/ipw_combined_topo_stress_python/FFM_and_topo' + topo_tensors + '_seg1' + \
                '_' + '%.1f' % direction + \
                '_' + '%.2f' % shmin + \
                '_' + '%.1f' % density + \
                '_new.csv'
    ffm_df = pd.read_csv(topo_file)
    tar_df = ffm_df[ffm_df['slip_all'] != 0.0]
    tar_topo_tensors = stu.list_tensors2array_tensors(
        tar_df[['xx', 'xy', 'xz', 'yy', 'yz', 'zz']].values)

    return tar_topo_tensors


def grid(FFM_df=None,
         topo_tensors=None,
         density=None,
         angle_min=-90,
         angle_max=90,
         angle_num=10,
         max_mag_min=None,
         max_mag_max=None,
         max_mag_num=None,
         min_mag_min=None,
         min_mag_max=None,
         min_mag_num=None,
         s_info=None,
         linear=False,
         linear2=False,
         angle_list=None,
         type_list=None,
         data_sigma=1.0,
         likelihood_mode='common',
         stress_log=True,
         pure_shear=False,
         both_pull=False,
         rock=False,
         smooth=False,
         smoothing_mode='rake',
         smoothing_sigma=None,
         row_n=12,
         column_n=21,
         output_file=None):

    likelihood_instance = \
        Likelihood(rake_observed=FFM_df['rake'].values,
                   rake_sigma=data_sigma,
                   slip=FFM_df['slip_all'].values,
                   mode=likelihood_mode)

    if rock:
        rock_h = FFM_df['rock_h'].values
        rock_v = FFM_df['rock_v'].values
        rock_tensors = np.array(
            [
                [
                    [rock_h[i], 0.0, 0.0],
                    [0.0, rock_h[i], 0.0],
                    [0.0, 0.0, rock_v[i]]
                ]
                for i in range(len(FFM_df))
            ]
        )
        known_tensors = topo_tensors + rock_tensors
    else:
        known_tensors = topo_tensors

    if s_info is None:
        s_info = [[None]]
    para_info = [
        [max_mag_min, max_mag_max, max_mag_num],
        [min_mag_min, min_mag_max, min_mag_num],
        [angle_min, angle_max, angle_num]
                ] \
                + s_info
    para_info = _filter_none(para_info)

    paras, iters, total_iters, L_array, norm_factor, saving_data = \
        _grid_prepare(para_info)

    for iter in tqdm(iters, total=total_iters):
        if max_mag_num is not None and min_mag_num is None:
            m_index, a_index = iter
            if stress_log:
                shmin = -np.inf
            else:
                shmin = 0.0
            tectonic_tensor = \
                stu.gen_tectonic_tensor(
                    shmax=paras[0][m_index],
                    direction=paras[1][a_index],
                    shmin=shmin,
                    stress_log=stress_log,
                    pure_shear=pure_shear,
                    both_pull=both_pull,
                )
            L_index = (a_index, m_index)
        elif max_mag_num is None and min_mag_num is not None:
            if linear:
                m_index, a_index = iter
                tectonic_tensor = stu.gen_depth_dependent_tectonic_tensor(
                    depths=FFM_df['dep'].values * 1000,
                    shmax_g=0.0,
                    shmin_g=-paras[0][m_index],
                    shmax_direction=paras[1][a_index],
                    density=density
                )
                L_index = (a_index, m_index)
            elif linear2:
                coe_index, con_index = iter
                tectonic_tensor = stu.gen_depth_dependent_tectonic_tensor(
                    depths=FFM_df['dep'].values * 1000,
                    shmax_g=0.0,
                    shmin_g=-paras[0][coe_index],
                    shmax_direction=6.5,
                    constant=paras[1][con_index]
                )
                L_index = (con_index, coe_index)
            else:
                m_index, a_index = iter
                # a_index = 0
                # m_index = 1
                if stress_log:
                    shmax = -np.inf
                else:
                    shmax = 0.0
                tectonic_tensor = \
                    stu.gen_tectonic_tensor(
                        shmax=shmax,
                        direction=paras[1][a_index],
                        shmin=paras[0][m_index],
                        stress_log=stress_log,
                        pure_shear=pure_shear,
                        both_pull=both_pull,
                    )

                if topo_tensors in ['1', '-1']:
                    known_tensors = _load_topo_tensors(
                        direction=paras[1][a_index],
                        shmin=paras[0][m_index],
                        topo_tensors=topo_tensors,
                        density=density
                    )

                L_index = (a_index, m_index)

        elif max_mag_num is not None and min_mag_num is not None:
            max_m_index, min_m_index, a_index = iter
            tectonic_tensor = \
                stu.gen_tectonic_tensor(
                    shmax=paras[0][max_m_index],
                    direction=paras[2][a_index],
                    shmin=paras[1][min_m_index],
                    stress_log=stress_log,
                    pure_shear=pure_shear,
                    both_pull=both_pull,
                )
            L_index = (a_index, max_m_index, min_m_index)
        elif s_info is not None:
            s1_index, s2_index = iter
            tectonic_tensor = \
                stu.summarize_2tensors(
                    s_list=[paras[0][s1_index], paras[1][s2_index]],
                    a_list=angle_list,
                    type_list=type_list,
                    stress_log=stress_log,
                )
            L_index = (s2_index, s1_index)

        else:
            raise ValueError

        full_tensor = known_tensors + tectonic_tensor

        slip_predicted, rake_predicted = stu.get_max_shear_stress(
            strike=FFM_df['strike'].values,
            dip=FFM_df['dip'].values,
            stress_tensor=full_tensor,
            angle='degrees'
        )

        # Smoothing the predicted rakes if necessary.
        if smooth:
            smoothing_sigma = np.array([1.0, 1.0]) \
                if smoothing_sigma is None else smoothing_sigma
            if smoothing_mode == 'rake':
                smoothed_rake_predicted = smoothing._smoothing_rake(
                    rakes=rake_predicted,
                    sigma=smoothing_sigma,
                    row_n=row_n,
                    column_n=column_n)
                L_array[L_index] = \
                    np.e ** (likelihood_instance.pdf_func(smoothed_rake_predicted))
            elif smoothing_mode == 'slip':
                smoothed_rake_predicted, smoothed_slips = smoothing._smoothing_slip(
                    rakes=rake_predicted,
                    slips=FFM_df['raw_slip'].values,
                    sigma=smoothing_sigma,
                    row_n=row_n,
                    column_n=column_n)
                L_array[L_index] = \
                    np.e ** (likelihood_instance.smoothed_weight2(
                        smoothed_rake_predicted, smoothed_slips))

            else:
                raise ValueError('No this smoothing mode!')
        else:
            L_array[L_index] = \
                np.e**(likelihood_instance.pdf_func(rake_predicted))

    saving_data.append(L_array)

    normalization = np.sum(L_array) * norm_factor
    # All zero values will be changed to all nan values.
    L_array = L_array / normalization
    saving_data.append(L_array)

    np.save(output_file, np.array(saving_data))

    return None


def show_grid_2D(
        grid_search_file=None,
        axes=None,
        vmin=None,
        vmax=None,
        para_label=None,
        cut=None,
        limitation_file=None
):
    if axes is None:
        left, width_mid = 0.12, 0.55
        bottom, height_mid = 0.1, 0.55
        height_up, width_right = 0.3, 0.3
        spacing = 0.005

        rect_scatter = [left, bottom, width_mid, height_mid]
        rect_histx = [
            left,
            bottom +
            height_mid +
            spacing,
            width_mid,
            height_up]
        rect_histy = [
            left +
            width_mid +
            spacing,
            bottom,
            width_right,
            height_mid]
        grid_bar = [0.8, 0.7, 0.02, 0.2]

        fig = plt.figure(figsize=(6, 6))
        ax_scatter = plt.axes(rect_scatter)
        ax_scatter.tick_params(
            direction='in',
            top=True,
            right=True,
            which='both')
        ax_histx = plt.axes(rect_histx)
        ax_histx.tick_params(
            direction='in',
            labelbottom=False,
            which='both')
        ax_histy = plt.axes(rect_histy)
        ax_histy.tick_params(direction='in', labelleft=False, which='both')
        grid_bar_ax = plt.axes(grid_bar)

        axes = [ax_scatter, ax_histx, ax_histy, grid_bar_ax]
    elif len(axes) == 3:
        axes.append(plt.axes([0.9, 0.7, 0.02, 0.2]))
        ax_scatter = axes[0]
        ax_histx = axes[1]
        ax_histy = axes[2]

    try:
        magnitudes, angle, magnitude_mesh, angle_mesh, _, L_array = np.load(
            grid_search_file, allow_pickle=True)
    except:
        magnitudes, angle, magnitude_mesh, angle_mesh, L_array = np.load(
            grid_search_file, allow_pickle=True)

    # Dara preparing
    magnitude_step = magnitudes[1] - magnitudes[0]
    angle_step = angle[1] - angle[0]

    if limitation_file is not None:
        # Plot upper limitation
        reasonable_parameters = np.load(limitation_file)
        searching_angles = np.unique(reasonable_parameters[:, 0])
        upper_limitation_magnitudes = []
        for searching_angle in searching_angles:
            tar_index = np.where(reasonable_parameters[:, 0] == searching_angle)
            upper_limitation_magnitude = np.max(reasonable_parameters[tar_index, 1])
            upper_limitation_magnitudes.append(upper_limitation_magnitude)

            tar_L_array_angle_index = np.where(angle == searching_angle)[0]
            tar_L_array_magnitude_index = np.where(magnitudes > upper_limitation_magnitude)[0]
            L_array[tar_L_array_angle_index, tar_L_array_magnitude_index] = np.nan
        normalization = np.nansum(L_array) * magnitude_step * angle_step
        # All zero values will be changed to all nan values.
        L_array = L_array / normalization

        upper_limitation_magnitudes = np.array(upper_limitation_magnitudes)

    # Plot joint distribution
    if vmin is not None and vmax is not None:
        grid = axes[0].pcolormesh(
            magnitude_mesh,
            angle_mesh,
            L_array,
            cmap='coolwarm',
            vmin=vmin,
            vmax=vmax,
            edgecolors='',
        )
    else:
        grid = axes[0].pcolormesh(
            magnitude_mesh,
            angle_mesh,
            L_array,
            cmap='coolwarm',
            vmin=np.nanmin(L_array),
            vmax=np.nanmax(L_array),
            edgecolors='',
        )
    axes[0].set_xlabel(para_label[1])
    axes[0].set_ylabel(para_label[0])
    major_index_x = np.linspace(0, len(magnitudes) - 1, 4)[:-1].astype(int)
    major_index_y = np.linspace(0, len(angle) - 1, 4)[:-1].astype(int)
    axes[0].set_xticks(magnitudes[major_index_x], minor=False)
    axes[0].set_yticks(angle[major_index_y], minor=False)
    cbar = plt.colorbar(grid, cax=axes[3])
    cbar.set_label('Probability')

    # Plot marginal distribution of magnitude
    magnitudes_prob = np.nansum(L_array, axis=0) * angle_step
    axes[1].plot(magnitudes,
                 magnitudes_prob,
                 'k-o',
                 markersize=3,
                 linewidth=1)
    axes[1].set_xlim(axes[0].get_xlim())
    axes[1].set_xticks(axes[0].get_xticks(), minor=False)
    axes[1].set_ylabel('Probability')

    if limitation_file is None:
        magnitude_min_index, magnitude_max_index = turning_point(
            pdf=magnitudes_prob,
            bin=magnitude_step,
            cut=cut
        )
        if magnitude_min_index is not None:
            magnitude_min = magnitudes[magnitude_min_index]
            axes1_ymin, axes1_ymax = axes[1].get_ylim()
            axes[1].plot([magnitude_min, magnitude_min], [axes1_ymin, axes1_ymax], 'k--')
            print('Min magnitude: ', magnitude_min)
        else:
            magnitude_min = np.nan
        if magnitude_max_index is not None:
            magnitude_max = magnitudes[magnitude_max_index]
            axes[1].plot([magnitude_max, magnitude_max], [axes1_ymin, axes1_ymax], 'k--')
            print('Max magnitude: ', magnitude_max)
        else:
            magnitude_max = np.nan
    else:
        magnitude_min, magnitude_max = confidence_level_value(
            x_point=magnitudes,
            pdf=magnitudes_prob,
            bin=magnitude_step,
            cl=0.68
        )
        axes1_ymin, axes1_ymax = axes[1].get_ylim()
        axes[1].plot([magnitude_min, magnitude_min], [axes1_ymin, axes1_ymax], 'k--')
        axes[1].plot([magnitude_max, magnitude_max], [axes1_ymin, axes1_ymax], 'k--')
        print('Min magnitude: ', magnitude_min)
        print('Max magnitude: ', magnitude_max)

    # Plot marginal distribution of angle
    angles_prob = np.nansum(L_array, axis=1) * magnitude_step
    axes[2].plot(angles_prob,
                 angle,
                 'k-o',
                 markersize=3,
                 linewidth=1)
    axes[2].set_yticks(axes[0].get_yticks(), minor=False)
    axes[2].set_ylim(axes[0].get_ylim())
    axes[2].set_xlabel('Probability')

    if limitation_file is None:
        angle_min_index, angle_max_index = turning_point(
            pdf=angles_prob,
            bin=angle_step,
            cut=cut
        )
        if angle_min_index is not None:
            angle_min = angle[angle_min_index]
            axes[2].plot(axes[2].get_xlim(), [angle_min, angle_min], 'k--')
            print('Min angle: ', angle_min)
        else:
            angle_min = np.nan
        if angle_max_index is not None:
            angle_max = angle[angle_max_index]
            axes[2].plot(axes[2].get_xlim(), [angle_max, angle_max], 'k--')
            print('Max angle: ', angle_max)
        else:
            angle_max = np.nan
    else:
        angle_min, angle_max = confidence_level_value(
            x_point=angle,
            pdf=angles_prob,
            bin=angle_step,
            cl=0.68
        )
        axes[2].plot(axes[2].get_xlim(), [angle_min, angle_min], 'k--')
        axes[2].plot(axes[2].get_xlim(), [angle_max, angle_max], 'k--')
        print('Min angle: ', angle_min)
        print('Max angle: ', angle_max)

    # Plot optimal points.
    L_array_roughly = L_array
    # L_array_roughly = np.around(L_array, decimals=12)
    max_a_index, max_m_index = np.where(
        L_array_roughly == np.nanmax(L_array_roughly))
    optimal_magnitude = magnitudes[max_m_index]
    print('Optimal magnitude: ', optimal_magnitude)
    optimal_angle = angle[max_a_index]
    print('Optimal angle: ', optimal_angle)
    axes[0].scatter(magnitudes[max_m_index],
                    angle[max_a_index],
                    marker='*',
                    edgecolor='k',
                    facecolor='',
                    s=100.0)
    axes[1].scatter(magnitudes[max_m_index],
                    magnitudes_prob[max_m_index],
                    marker='*',
                    edgecolor='r',
                    facecolor='',
                    s=200.0)
    axes[2].scatter(angles_prob[max_a_index],
                    angle[max_a_index],
                    marker='*',
                    edgecolor='r',
                    facecolor='',
                    s=200.0)

    # Plot physical bound
    if limitation_file is not None:
        axes[0].plot(upper_limitation_magnitudes, searching_angles, 'k-')
        optimal_angle_index = np.where((searching_angles >= angle_min) & (searching_angles <= angle_max))
        print('average upper limitation: %f' % np.mean(upper_limitation_magnitudes[optimal_angle_index[0]]))

    # Prepare return result
    parameter_results = [optimal_magnitude, magnitude_min, magnitude_max, optimal_angle, angle_min, angle_max]
    plotting_results = [ax_scatter, ax_histx, ax_histy]
    return parameter_results, plotting_results


def show_grid(
        grid_search_file=None,
        para_label=None,
        uncertainty=True,
        fig=None,
        axes=None,
        bar_axes=None
):
    search_data = np.load(
        grid_search_file, allow_pickle=True)
    L_array = search_data[-1]
    para_num = len(np.shape(L_array))
    para_step = np.array([para[1] - para[0]
                          for para in search_data[:para_num]])

    # Plot optimal points.
    # L_array_roughly = L_array
    L_array_roughly = np.around(L_array, decimals=12)
    op_para_indexs = np.where(
        L_array_roughly == np.max(L_array_roughly))
    for i in range(para_num):
        print(para_label[i], search_data[i][op_para_indexs[i]])

    if axes is None:
        fig = plt.figure(figsize=[12, 10])
        axes = fig.subplots(para_num, para_num)
        plt.subplots_adjust(left=0.1, bottom=0.5, wspace=0.001, hspace=0.001)

        bar_axes = [
            plt.axes([0.24, 0.64, 0.08, 0.01]),
            plt.axes([0.24, 0.32, 0.08, 0.01]),
            plt.axes([0.56, 0.32, 0.08, 0.01])
        ]

    axes_para_dic = {0: 2, 1: 0, 2: 1}

    for para_indexs in product(range(para_num), range(para_num)):
        index1, index2 = para_indexs
        if index1 == index2:
            index = index1
            para = search_data[index]
            tar_L_array = L_array.copy()
            sum_index = np.delete(np.arange(para_num), index)
            tar_L_array = np.sum(tar_L_array, axis=tuple(sum_index)) \
                          * np.prod(para_step[sum_index])

            if uncertainty:
                p_min, p_max = turning_point(
                    pdf=tar_L_array,
                    bin=para_step[index]
                )
            else:
                p_min = None
                p_max = None

            ax = axes[axes_para_dic[index], axes_para_dic[index]]
            if axes_para_dic[index] == 2:
                ax.plot(tar_L_array,
                        para,
                        'k-o',
                        markersize=3,
                        linewidth=1)
                for op_i in op_para_indexs[index]:
                    # op_index = op_para_indexs[index][0]
                    ax.scatter(tar_L_array[op_i], para[op_i],
                               marker='*', s=200, c='r')

                x_min, x_max = ax.get_xlim()
                if p_min is not None:
                    ax.plot([x_min, x_max], [para[p_min], para[p_min]], 'k--')
                    print('Min: ', para[p_min])
                if p_max is not None:
                    ax.plot([x_min, x_max], [para[p_max], para[p_max]], 'k--')
                    print('Max: ', para[p_max])

                ax.set_ylabel(para_label[index])
                ax.set_xlabel('Probability')
                ax.set_ylim([para[0] - 0.5 * para_step[index],
                             para[-1] + 0.5 * para_step[index]])
                ax.set_yticks(para, minor=False)
                ax.yaxis.set_major_locator(LinearLocator(4))
            else:
                if axes_para_dic[index] == 1:
                    ax.set_yticks([])
                    ax = ax.twinx()
                    axes[axes_para_dic[index], axes_para_dic[index]] = ax

                ax.plot(para,
                        tar_L_array,
                        'k-o',
                        markersize=3,
                        linewidth=1)
                for op_i in op_para_indexs[index]:
                    # op_index = op_para_indexs[index][0]
                    ax.scatter(para[op_i], tar_L_array[op_i],
                               marker='*', s=200, c='r')

                y_min, y_max = ax.get_ylim()
                if p_min is not None:
                    ax.plot([para[p_min], para[p_min]], [y_min, y_max], 'k--')
                    print('Min: ', para[p_min])
                if p_max is not None:
                    ax.plot([para[p_max], para[p_max]], [y_min, y_max], 'k--')
                    print('Max: ', para[p_max])

                ax.set_xlabel(para_label[index])
                ax.set_ylabel('Probability')
                ax.set_xlim([para[0] - 0.5 * para_step[index],
                             para[-1] + 0.5 * para_step[index]])
                ax.set_xticks(para, minor=False)
                ax.xaxis.set_major_locator(LinearLocator(4))

        elif index1 < index2:
            para1 = search_data[index1]
            para2 = search_data[index2]
            para1_mesh_index = np.linspace(
                para1[0] - 0.5 * para_step[index1],
                para1[-1] + 0.5 * para_step[index1],
                len(para1) + 1)
            para2_mesh_index = np.linspace(
                para2[0] - 0.5 * para_step[index2],
                para2[-1] + 0.5 * para_step[index2],
                len(para2) + 1)
            para1_mesh, para2_mesh = np.meshgrid(para1_mesh_index, para2_mesh_index)

            tar_L_array = L_array.copy()
            sum_index = tuple(np.delete(np.arange(para_num), para_indexs))
            tar_L_array = np.sum(tar_L_array, axis=sum_index) \
                          * np.prod(para_step[sum_index])

            new_index_1 = axes_para_dic[index1]
            new_index_2 = axes_para_dic[index2]
            if new_index_1 < new_index_2:
                ax_x = new_index_2
                ax_y = new_index_1
                new_para1 = para1
                new_para2 = para2
                new_para1_mesh = para1_mesh
                new_para2_mesh = para2_mesh
                showing_L_array = tar_L_array.T
                x_label = para_label[index1]
                y_label = para_label[index2]
            else:
                ax_x = new_index_1
                ax_y = new_index_2
                new_para1 = para2
                new_para2 = para1
                new_para1_mesh = para2_mesh.T
                new_para2_mesh = para1_mesh.T
                showing_L_array = tar_L_array
                x_label = para_label[index2]
                y_label = para_label[index1]
            ax = axes[ax_x, ax_y]
            grid = ax.pcolormesh(
                new_para1_mesh,
                new_para2_mesh,
                showing_L_array,
                cmap='coolwarm',
                # vmin=np.min(L_array),
                # vmax=np.max(L_array),
                edgecolors='',
            )
            cbar = plt.colorbar(
                grid,
                cax=bar_axes[index1 + index2 - 1],
                orientation='horizontal')
            font = {'color': 'white',
                    'size': 10,
                    }
            cbar.ax.tick_params(labelsize=10, colors='white')
            cbar.set_label('Probability', fontdict=font)

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_xlim([new_para1_mesh[0, 0],
                         new_para1_mesh[0, -1]])
            ax.set_ylim([new_para2_mesh[0, 0],
                         new_para2_mesh[-1, 0]])
            ax.set_xticks(new_para1, minor=False)
            ax.set_yticks(new_para2, minor=False)
            ax.xaxis.set_major_locator(LinearLocator(4))
            ax.yaxis.set_major_locator(LinearLocator(4))
        else:
            axes[index2, index1].spines['top'].set_visible(False)
            axes[index2, index1].spines['left'].set_visible(False)
            axes[index2, index1].spines['right'].set_visible(False)
            axes[index2, index1].spines['bottom'].set_visible(False)
            axes[index2, index1].set_xticks([])
            axes[index2, index1].set_yticks([])

    # norm = mpl.colors.Normalize(vmin=np.min(L_array), vmax=np.max(L_array))
    # map = mpl.cm.ScalarMappable(norm=norm, cmap="coolwarm")
    # cbar = plt.colorbar(map, cax=axes[0, 2])
    # cbar.set_label('Probability')

    return fig, axes, bar_axes


def statistic_likelihood(
        grid_files=None,
        k=None,
        n=None,
        labels=None,
        colors=None,
        ax=None,
        **kwargs,
):
    if ax is None:
        fig, ax = plt.subplots()
    tar_l_mean_list = []
    for i, file in enumerate(grid_files):
        search_data = np.load(
            file, allow_pickle=True)
        l_array = search_data[-2]
        l_array = l_array.flatten()
        l_mean = np.log(l_array)/n[i]
        print('Largest mean L: ', np.max(l_mean))
        lower_limit = np.max(l_mean) - 0.5 * abs(np.max(l_mean))
        tar_l_mean = l_mean[np.where(l_mean >= lower_limit)]
        tar_l_mean_list.append(tar_l_mean)
        print('# of target mean L: ', len(tar_l_mean))
        if k is not None:
            bic = k[i] * np.log(n[i]) - 2 * np.log()
            ax.hist(tar_l_mean, density=True,
                    alpha=0.4, label=labels[i] + ' %.4f' % np.max(l_mean), color=colors[i])

    ax.boxplot(tar_l_mean_list, **kwargs)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Average likelihood')
    return ax


def statistic_variation(
        grid_files=None,
        labels=None,
        ax=None,
        **kwargs,
):
    if ax is None:
        fig, ax = plt.subplots()
    tar_min_mag_list = []
    for i, file in enumerate(grid_files):
        search_data = np.load(
            file, allow_pickle=True)
        L_array = search_data[-1]
        print('Largest mean L: ', np.max(L_array))
        lower_limit = np.max(L_array) - 0.5 * abs(np.max(L_array))
        tar_L_mean_index = np.where(L_array >= lower_limit)

        min_mag = search_data[0]
        tar_min_mag = min_mag[tar_L_mean_index[1]]
        tar_min_mag_list.append(tar_min_mag)

    ax.boxplot(tar_min_mag_list, **kwargs)
    ax.set_xticklabels(labels)
    ax.set_ylabel('log($S_{hmin}$) (Pa)')
    return ax


def confidence_level_value(
        x_point=None,
        pdf=None,
        cl=None,
):

    fine_x = np.linspace(x_point[0], x_point[-1], 10 * len(x_point))
    fine_bin = fine_x[1] - fine_x[0]
    interpolated_pdf = interpolate.interp1d(x_point, pdf, kind='linear')
    fine_pdf = interpolated_pdf(fine_x)
    fine_cdf = np.cumsum(fine_pdf) * fine_bin

    cl_left_cdf = abs(fine_cdf - (0.5 - cl / 2))
    cl_right_cdf = abs(fine_cdf - (0.5 + cl / 2))

    x_min_index = np.where(cl_left_cdf == np.min(cl_left_cdf))
    x_max_index = np.where(cl_right_cdf == np.min(cl_right_cdf))

    x_min = fine_x[x_min_index]
    x_max = fine_x[x_max_index]

    return x_min, x_max


def turning_point(
        pdf=None,
        bin=None,
        cut=None,
):
    k = (pdf[2:] - pdf[:-2]) / (2 * bin)
    kk = np.zeros(len(pdf))
    kk[1:-1] = k
    max_index = np.where(pdf == np.nanmax(pdf))[0][0]
    kk_left = kk[:max_index]
    if cut is None:
        kk_right = abs(kk[max_index:])
    else:
        kk_right = abs(kk[max_index:cut])

    kk_left_max_index = np.where(kk_left == np.max(kk_left))
    min_point = kk_left_max_index

    kk_right_max_index = np.where(kk_right == np.max(kk_right))
    max_point = kk_right_max_index

    if len(min_point[0]) == 0:
        min_turn_index = None
    else:
        min_turn_index = min_point[0][0]
    if len(max_point[0]) == 0:
        max_turn_index = None
    else:
        max_turn_index = max_point[0][0] + max_index
    return min_turn_index, max_turn_index


def likelihood_distribution(
        grid_files=None,
        boundaries=None,
        n=None,
        labels=None,
        colors=None,
        ax=None,
        **kwargs,
):
    if ax is None:
        fig = plt.figure()
        ax = fig.subplots(1, 1)

    for i, file in enumerate(grid_files):
        boundary = boundaries[labels[i]]
        search_data = np.load(
            file, allow_pickle=True)
        likelihood = np.log(search_data[-2])/n[i]
        l_array = search_data[-1]

        para_num = len(np.shape(l_array))

        if para_num == 2:
            indexes = []
            for p in range(para_num):
                para = search_data[1 - p]
                index = np.where((para >= boundary[p][0]) & (para <= boundary[p][1]))
                indexes.append([index[0][0], index[0][-1]])
            tar_samples = likelihood[
                          indexes[0][0]:indexes[0][1] + 1,
                          indexes[1][0]:indexes[1][1] + 1
                          ]
        if para_num == 3:
            indexes = []
            for p in range(para_num):
                para = search_data[1 - p]
                index = np.where((para >= boundary[p][0]) & (para <= boundary[p][0]))
                indexes.append(index)
            tar_samples = likelihood[
                indexes[0],
                indexes[1],
                indexes[2],
                          ]
        else:
            raise ValueError

        tar_samples = tar_samples.flatten()

        samples = tar_samples.copy()
        print('Largest: ', np.max(samples))
        print('# : ', len(samples))

        ax.hist(
            samples,
            density=True,
            bins=20,
            orientation='horizontal',
            color=colors[i],
            alpha=0.5,
            label=labels[i],
            **kwargs
        )
        mean = np.mean(samples)
        ax.plot([0, 80], [mean, mean], '--', color=colors[i])
        ax.text(80.0, -0.28 - 0.01 * i,
                'Max: %.2f Mean: %.2f' % (np.max(samples), mean),
                color=colors[i]
                )

        ax.set_xlabel('Density')

        ax.set_ylabel('Average likelihood')

        ax.grid()
        ax.legend()

    return ax



