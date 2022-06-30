#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: yunnaidan
@time: 2022/02/10
@file: points.py
"""


class Points:
    def __init__(self, point_data):
        self.point_data = point_data

    def colored_plot(self, ax=None, **kwargs):
        event_mappable = ax.scatter(
            self.point_data[:, 0], self.point_data[:, 1], c=self.point_data[:, 2],
            **kwargs
        )

        return event_mappable


    def texted_plot(self, ax=None, **kwargs):
        for point in self.point_data:
            ax.scatter(float(point[0]), float(point[1]), **kwargs['point_attributes'])
            ax.text(
                float(point[0]), float(point[1]), point[2], **kwargs['text_attributes']
            )
        return None
