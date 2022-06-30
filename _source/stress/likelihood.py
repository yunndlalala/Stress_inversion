#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: yunnaidan
@time: 2020/01/06
@file: likelihood.py
"""
import numpy as np
from scipy.ndimage import filters
from scipy.special import expit


class Likelihood(object):
    def __init__(
            self,
            strike_observed=None,
            dip_observed=None,
            rake_observed=None,
            slip=None,
            fore_fms=None,
            post_fms=None,
            sigmoid_scale=None,
            ruptured='step',
            unruptured=None,
            rake_sigma=None,
            mode='common'
    ):
        # For inverting using FFM and topography

        self.strike_observed = strike_observed
        self.dip_observed = dip_observed
        self.rake_observed = rake_observed
        self.slip = slip

        # For inverting using FM

        self.fore_fms = fore_fms
        self.post_fms = post_fms

        self.ruptured = ruptured
        self.unruptured = unruptured
        self.sigmoid_scale = sigmoid_scale
        self.normalized_sigmoid_scale = None

        # For all inversion

        self.rake_sigma = rake_sigma

        likehood_func = {
            'common': self.log_likelihood_angle,
            'weight': self.log_likelihood_angle_slip_weight,
            'weight2': self.log_likelihood_angle_slip_weight2,
            'coulomb': self.log_coulomb
        }
        self.pdf_func = likehood_func[mode]

    def log_likelihood_angle(self, rake_predicted):
        rake_diff = np.fmin(abs(self.rake_observed - rake_predicted),
                            360 - abs(self.rake_observed - rake_predicted))

        log_likelihood = - 0.5 * \
            np.sum(rake_diff ** 2.0 / (self.rake_sigma ** 2.0))
        return log_likelihood

    def log_likelihood_angle_slip_weight(self, rake_predicted):
        rake_diff = np.fmin(abs(self.rake_observed - rake_predicted),
                            360 - abs(self.rake_observed - rake_predicted))
        normalized_slip = abs(self.slip) / np.mean(self.slip)
        data_sigma_weight = np.nan_to_num(np.array([self.rake_sigma]).repeat(
            len(rake_diff), axis=0) / np.sqrt(normalized_slip))
        log_likelihood = - 0.5 * \
            np.sum(rake_diff ** 2.0 / (data_sigma_weight ** 2.0))
        return log_likelihood

    def log_likelihood_angle_slip_weight2(self, rake_predicted):
        slip_x_observed = np.array([self.slip[i] * np.cos(np.deg2rad(self.rake_observed[i]))
                                    for i in range(len(self.slip))])
        slip_y_observed = np.array([self.slip[i] * np.sin(np.deg2rad(self.rake_observed[i]))
                                    for i in range(len(self.slip))])
        slip_x_predicted = np.array([self.slip[i] * np.cos(np.deg2rad(rake_predicted[i]))
                                     for i in range(len(self.slip))])
        slip_y_predicted = np.array([self.slip[i] * np.sin(np.deg2rad(rake_predicted[i]))
                                     for i in range(len(self.slip))])
        data_sigma = np.array([self.rake_sigma]).repeat(len(self.slip), axis=0)
        log_likelihood_1 = - 0.5 * \
            np.sum((slip_x_predicted - slip_x_observed) ** 2.0 / (data_sigma ** 2.0))
        log_likelihood_2 = - 0.5 * \
            np.sum((slip_y_predicted - slip_y_observed) ** 2.0 / (data_sigma ** 2.0))
        log_likelihood = log_likelihood_1 + log_likelihood_2

        return log_likelihood

    def smoothed_weight2(self, rake_predicted, smoothed_slips):
        slip_x_observed = np.array([self.slip[i] * np.cos(np.deg2rad(self.rake_observed[i]))
                                    for i in range(len(self.slip))])
        slip_y_observed = np.array([self.slip[i] * np.sin(np.deg2rad(self.rake_observed[i]))
                                    for i in range(len(self.slip))])
        slip_x_predicted = np.array([smoothed_slips[i] * np.cos(np.deg2rad(rake_predicted[i]))
                                     for i in range(len(self.slip))])
        slip_y_predicted = np.array([smoothed_slips[i] * np.sin(np.deg2rad(rake_predicted[i]))
                                     for i in range(len(self.slip))])
        data_sigma = np.array([self.rake_sigma]).repeat(len(self.slip), axis=0)
        log_likelihood_1 = - 0.5 * \
            np.sum((slip_x_predicted - slip_x_observed) ** 2.0 / (data_sigma ** 2.0))
        log_likelihood_2 = - 0.5 * \
            np.sum((slip_y_predicted - slip_y_observed) ** 2.0 / (data_sigma ** 2.0))
        log_likelihood = log_likelihood_1 + log_likelihood_2

        return log_likelihood

    def log_coulomb(
            self,
            obs_fms_type='fore',
            pred_friction=None,
            pos_pred_shear_stresses=None,
            pos_pred_norm_stresses=None,
            pos_pred_rakes=None,
            neg_pred_shear_stresses=None,
            neg_pred_norm_stresses=None,
    ):
        if obs_fms_type == 'fore':
            obs_fms = self.fore_fms
        elif obs_fms_type == 'post':
            obs_fms = self.post_fms
        else:
            raise ValueError

        rake_diff = np.around(
            np.fmin(
                abs(obs_fms['pos_obs_rakes'] - pos_pred_rakes),
                360 - abs(obs_fms['pos_obs_rakes'] - pos_pred_rakes)
            ),
            10
        )
        rake_log_likelihood = - 0.5 * np.sum(rake_diff ** 2.0 / (self.rake_sigma ** 2.0))

        pos_coloumb_stresses = np.around(pos_pred_shear_stresses - pred_friction * pos_pred_norm_stresses, 10)
        neg_coloumb_stresses = np.around(neg_pred_shear_stresses - pred_friction * neg_pred_norm_stresses, 10)

        if self.ruptured == 'step':
            ruptured_log_likelihood = np.sum(log_step(
                pos_coloumb_stresses
            ))
        # elif self.ruptured == 'step_&_open':
        #     ruptured_log_likelihood = \
        #         np.sum(log_step(pos_coloumb_stresses)) + np.sum(log_step(pos_pred_norm_stresses))
        elif self.ruptured == 'sigmoid':
            ruptured_log_likelihood = np.sum(log_sigmoid(
                pos_coloumb_stresses,
                scale=self.normalized_sigmoid_scale
            ))
        else:
            raise ValueError

        if self.unruptured == 'None':
            unruptured_log_likelihood = 0.0
        # elif self.unruptured == 'open':
        #     unruptured_log_likelihood = np.sum(log_step(neg_pred_norm_stresses))
        elif self.unruptured == 'step':
            unruptured_log_likelihood = np.sum(log_step(
                -neg_coloumb_stresses
            ))
        elif self.unruptured == 'sigmoid':
            unruptured_log_likelihood = np.sum(log_sigmoid(
                -neg_coloumb_stresses,
                scale=self.normalized_sigmoid_scale
            ))
        else:
            raise ValueError

        log_likelihood = rake_log_likelihood + ruptured_log_likelihood + unruptured_log_likelihood

        return log_likelihood


def log_step(
        x,
        invert=False,
        contain0=True
):
    if invert is True:
        x = - x

    p = np.zeros(np.shape(x))

    if contain0:
        p[np.where(x < 0.0)] = -np.inf

    else:
        p[np.where(x <= 0.0)] = -np.inf

    return p


def log_sigmoid(
        x,
        scale=0.1,
        invert=False
):
    if invert is True:
        x = -x

    return np.log(expit(x / scale))





