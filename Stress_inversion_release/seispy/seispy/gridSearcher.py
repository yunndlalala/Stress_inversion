#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: yunnaidan
@time: 2022/03/23
@file: gridSearcher.py
"""
import itertools
import numpy as np
from itertools import product

import pandas as pd
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt


def _show_grid(
        parameters=None,
        joint_pdf=None,
        para_label=None,
        axes=None,
        **kwargs
):
    """
    parameters: mxn, n is the number of the searched parameters.
                m is the number of the parameter groups.
    joint_pdf: m

    """

    if 'line' in kwargs.keys():
        line_kwargs = kwargs['line']
    else:
        line_kwargs = {}
    if 'optimal' in kwargs.keys():
        optimal_kwargs = kwargs['optimal']
    else:
        optimal_kwargs = {}
    if 'mesh' in kwargs.keys():
        mesh_kwargs = kwargs['mesh']
    else:
        mesh_kwargs = {}

    sorted_indexes = np.lexsort((parameters[:, 2], parameters[:, 1], parameters[:, 0]))
    parameters = parameters[sorted_indexes]
    joint_pdf = joint_pdf[sorted_indexes]

    para_num = len(parameters[0])
    para_step = np.zeros(para_num)
    para_points = np.zeros(para_num)
    for i in range(para_num):
        unique_para = sorted(np.unique(parameters[:, i]))
        para_step[i] = unique_para[1] - unique_para[0]
        para_points[i] = len(unique_para)

    joint_pdf_roughly = np.around(joint_pdf, decimals=12)
    op_para_indexs = np.where(
        joint_pdf_roughly == np.max(joint_pdf_roughly))[0]
    print('The optimal parameters: ')
    for index in op_para_indexs:
        print(str(para_label) + ': ' + str(parameters[index]))

    joint_pdf = joint_pdf.reshape(para_points.astype(int))

    if axes is None:
        fig = plt.figure()
        axes = fig.subplots(para_num, para_num)

    bar_axes = np.array([
        plt.axes([
            ax.get_position().x1 + 0.01,
            ax.get_position().y0,
            0.05 * (ax.get_position().x1 - ax.get_position().x0),
            ax.get_position().height
        ])
        for ax in axes.flatten()
    ])
    bar_axes = bar_axes.reshape(np.shape(axes))

    for para_indexs in product(range(para_num), range(para_num)):
        # para_indexs = (0, 2)
        index1, index2 = para_indexs
        if index1 == index2:
            index = index1
            para = np.array(sorted(np.unique(parameters[:, index])))
            sum_index = np.delete(np.arange(para_num), index)
            tar_pdf = np.sum(joint_pdf, axis=tuple(sum_index)) \
                          * np.prod(para_step[sum_index])

            ax = axes[index1, index2]
            ax.plot(
                para,
                tar_pdf,
                **line_kwargs
            )

            optimal_tar_para = np.unique(parameters[op_para_indexs][:, index])
            optimal_tar_pdf = tar_pdf[[
                i for i in range(len(para)) if para[i] in optimal_tar_para
            ]]
            ax.scatter(
                optimal_tar_para, optimal_tar_pdf,
                **optimal_kwargs
            )

            ax.set_xlabel(para_label[index])
            ax.set_ylabel('Probability')
            ax.set_xlim([
                para[0] - 0.5 * para_step[index],
                para[-1] + 0.5 * para_step[index]
            ])
            ax.set_xticks(para[np.linspace(0, len(para) - 1, 4).astype(int)], minor=False)

            bar_axes[index1, index2].axis('off')

        elif index1 < index2:
            para1 = np.array(sorted(np.unique(parameters[:, index1])))
            para2 = np.array(sorted(np.unique(parameters[:, index2])))
            para1_mesh_index = np.linspace(
                para1[0] - 0.5 * para_step[index1],
                para1[-1] + 0.5 * para_step[index1],
                len(para1) + 1
            )
            para2_mesh_index = np.linspace(
                para2[0] - 0.5 * para_step[index2],
                para2[-1] + 0.5 * para_step[index2],
                len(para2) + 1
            )
            para1_mesh, para2_mesh = np.meshgrid(para1_mesh_index, para2_mesh_index)

            sum_index = tuple(np.delete(np.arange(para_num), para_indexs))
            tar_pdf = np.sum(joint_pdf, axis=sum_index) \
                          * np.prod(para_step[sum_index])

            if index1 < index2:
                ax_x = index2
                ax_y = index1
                new_para1 = para1
                new_para2 = para2
                new_para1_mesh = para1_mesh
                new_para2_mesh = para2_mesh
                showing_pdf = tar_pdf.T
                x_label = para_label[index1]
                y_label = para_label[index2]
            else:
                ax_x = index1
                ax_y = index2
                new_para1 = para2
                new_para2 = para1
                new_para1_mesh = para2_mesh.T
                new_para2_mesh = para1_mesh.T
                showing_pdf = tar_pdf
                x_label = para_label[index2]
                y_label = para_label[index1]
            ax = axes[ax_x, ax_y]
            grid = ax.pcolormesh(
                new_para1_mesh,
                new_para2_mesh,
                showing_pdf,
                **mesh_kwargs
            )
            cbar = plt.colorbar(
                grid,
                cax=bar_axes[ax_x, ax_y]
            )
            cbar.set_label('Probability')

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_xlim([new_para1_mesh[0, 0],
                         new_para1_mesh[0, -1]])
            ax.set_ylim([new_para2_mesh[0, 0],
                         new_para2_mesh[-1, 0]])
            ax.set_xticks(
                new_para1[np.linspace(0, len(new_para1) - 1, 4).astype(int)],
                minor=False
            )
            ax.set_yticks(
                new_para2[np.linspace(0, len(new_para2) - 1, 4).astype(int)],
                minor=False
            )

        else:
            axes[index2, index1].spines['top'].set_visible(False)
            axes[index2, index1].spines['left'].set_visible(False)
            axes[index2, index1].spines['right'].set_visible(False)
            axes[index2, index1].spines['bottom'].set_visible(False)
            axes[index2, index1].set_xticks([])
            axes[index2, index1].set_yticks([])
            bar_axes[index2, index1].axis('off')

    return axes, bar_axes


class GridSearcher:
    def __init__(self, parameters=None, searcher=None):
        self.parameters = parameters
        self.searcher = searcher
        self.result = None
        self.result_file = None

    def search_run(
            self,
            parameters=None,
            searcher=None,
            hyperparameters=None,
            core=1
    ):
        if parameters is not None:
            self.parameters = parameters
        if searcher is not None:
            self.searcher = searcher

        if core == 1:
            for p in parameters:
                self.searcher()

    def search_show(self, result=None, result_file=None, axes=None, **kwargs):
        if self.result is None:
            if self.result_file is None:
                if result is None:
                    self.result = pd.read_csv(result_file)
                else:
                    self.result = result
            else:
                self.result = pd.read_csv(self.result_file)

        parameters = result.iloc[:, :-1].values
        joint_pdf = result.iloc[:, -1].values
        para_label = result.columns.values[:-1]
        axes, bar_axes = _show_grid(
            parameters=parameters,
            joint_pdf=joint_pdf,
            para_label=para_label,
            axes=axes,
            **kwargs
        )

        return axes, bar_axes









