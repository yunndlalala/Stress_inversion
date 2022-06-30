#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: yunnaidan
@time: 2020/01/05
@file: utils.py
"""
import numpy as np


def get_norm_vector(
        strike=None,
        dip=None,
        angle='degrees'
):
    """
    Coordinates: X,Y,Z = East, North, Up.
    Args:
        strike: strike angle of the fault plane
        dip: dip angle of the fault plane
        angle: 'degrees' or 'radians'

    Returns: The unit norm vector of the fault plane of the hanging wall.
             If strike and dip are floats, return (3,1) numpy matrix: [n_E, n_N, n_U].
             If strike and dip are arrays, return (n, 1, 3) where n is the length of strike and dip.

    """

    if angle == 'degrees':
        strike_rad = np.deg2rad(strike)
        dip_rad = np.deg2rad(dip)
    elif angle == 'radians':
        strike_rad = strike
        dip_rad = dip

    nE = -np.sin(dip_rad) * np.cos(strike_rad)
    nN = np.sin(dip_rad) * np.sin(strike_rad)
    nD = -np.cos(dip_rad)

    if not isinstance(strike, np.ndarray) and not isinstance(strike, list):
        norm_vector = np.array([nE, nN, nD])
    else:
        norm_vector = np.array([nE, nN, nD])
        norm_vector = np.transpose(np.expand_dims(norm_vector, axis=0), [2, 0, 1])

    return norm_vector.astype(float)


def get_strike_vector(
        strike=None,
        angle='degrees'
):
    """
    Coordinates: X,Y,Z = East, North, Up.
    Args:
        strike: strike angle of the fault plane
        angle: 'degrees' or 'radians'

    Returns: The vector on the fault plane of the hanging wall points in the direction of strike.
             If strike is float, return (3,1) numpy matrix: [n_E, n_N, n_U].
             If strike is array, return (n, 1, 3) where n is the length of strike.

    """

    if angle == 'degrees':
        strike_rad = np.deg2rad(strike)
    elif angle == 'radians':
        strike_rad = strike

    if not isinstance(strike, np.ndarray) and not isinstance(strike, list):
        strike_vector = np.array([np.sin(strike_rad), np.cos(strike_rad), 0])
    else:
        strike_vector = np.array(
            [np.sin(strike_rad), np.cos(strike_rad), np.zeros(len(strike))])
        strike_vector = np.transpose(np.expand_dims(strike_vector, axis=0), [2, 0, 1])

    return strike_vector.astype(float)


def get_dip_vector(
        strike=None,
        dip=None,
        angle='degrees'
):
    """
    Coordinates: X,Y,Z = East, North, Up.
    Args:
        strike: strike angle of the fault plane
        dip: dip angle of the fault plane
        angle: 'degrees' or 'radians'

    Returns: The vector on the fault plane of the hanging wall points down dip.
             If strike and dip are floats, return (3,1) numpy matrix: [n_E, n_N, n_U].
             If strike and dip are arrays, return (n, 1, 3) where n is the length of strike and dip.

    """

    N = get_norm_vector(strike, dip, angle=angle)
    S = get_strike_vector(strike, angle=angle)

    return np.cross(N, S).astype(float)


def get_rake_from_shear_components(
        strike_shear=None,
        dip_shear=None,
        angle='degrees'
):
    """

    Args:
        strike_shear: shear stress in the strike direction
        dip_shear: shear stress in the down-dip direction
        angle: type of output, 'degrees' or 'radians'

    Returns: rake angle; 0 means the direction of strike.

    """

    rake = np.arctan2(-dip_shear, strike_shear)  # rad
    if angle == 'degrees':
        rake = np.degrees(rake)

    return rake


def get_norm_stress(
        strike=None,
        dip=None,
        stress_tensor=None,
        angle='degrees'
):
    """
    Coordinates: X,Y,Z = East, North, Up.
    Args:
        strike: strike angle of the fault plane
        dip: dip angle of the fault plane
        stress_tensor: stress tensors; (3, 3) or (n, 3, 3) where n is the number of stress tensors and also the lengths
        of the strike and dip.
        angle: 'degrees' or 'radians'

    Returns: Stress in the norm vector direction of the hanging wall.
             Positive stress means compressive.
    """

    N = get_norm_vector(strike, dip, angle)

    if len(np.shape(N)) == 1:
        norm_stress = N.dot(stress_tensor).dot(N.T).astype(float)
    elif len(np.shape(N)) == 3:
        plane_stress = np.array(list(map(np.dot, N, stress_tensor)))
        norm_stress = np.array(
            list(map(np.dot, plane_stress, np.transpose(N, [0, 2, 1]))))
        norm_stress = np.squeeze(norm_stress.astype(float))
    else:
        raise ValueError

    return norm_stress


def get_dip_shear_stress(
        strike=None,
        dip=None,
        stress_tensor=None,
        angle='degrees'
):
    """
    Coordinates: X,Y,Z = East, North, Up.
    Args:
        strike: strike angle of the fault plane
        dip: dip angle of the fault plane
        stress_tensor: stress tensors; (3, 3) or (n, 3, 3) where n is the number of stress tensors and also the lengths
        of the strike and dip.
        angle: 'degrees' or 'radians'

    Returns: Stress in the down_dip direction on the hanging wall.
             Positive shear stress means the hanging wall moves down, i.e. normal-sense fault.
    """

    N = get_norm_vector(strike, dip, angle)
    D = get_dip_vector(strike, dip, angle)

    if len(np.shape(N)) == 1:
        dip_stress = N.dot(stress_tensor).dot(D.T).astype(float)
    elif len(np.shape(N)) == 3:
        plane_stress = np.array(list(map(np.dot, N, stress_tensor)))
        dip_stress = np.array(
            list(map(np.dot, plane_stress, np.transpose(D, [0, 2, 1]))))
        dip_stress = np.squeeze(dip_stress.astype(float))
    else:
        raise ValueError

    return dip_stress


def get_strike_shear_stress(
        strike=None,
        dip=None,
        stress_tensor=None,
        angle='degrees'
):
    """
    Coordinates: X,Y,Z = East, North, Up.
    Args:
        strike: strike angle of the fault plane
        dip: dip angle of the fault plane
        stress_tensor: stress tensors; (3, 3) or (n, 3, 3) where n is the number of stress tensors and also the lengths
        of the strike and dip.
        angle: 'degrees' or 'radians'

    Returns: Stress in the strike direction on the hanging wall.
             Positive shear stress means the hanging wall moves in the direction of strike, i.e. left-lateral fault.
    """

    N = get_norm_vector(strike, dip, angle)
    S = get_strike_vector(strike, angle)

    if len(np.shape(N)) == 1:
        strike_stress = N.dot(stress_tensor).dot(S.T).astype(float)
    elif len(np.shape(N)) == 3:
        plane_stress = np.array(list(map(np.dot, N, stress_tensor)))
        strike_stress = np.array(
            list(map(np.dot, plane_stress, np.transpose(S, [0, 2, 1]))))
        strike_stress = np.squeeze(strike_stress.astype(float))
    else:
        raise ValueError

    return strike_stress


def get_max_shear_stress(
        tau_ss=None,
        tau_dd=None,
        strike=None,
        dip=None,
        stress_tensor=None,
        angle='degrees'
):
    """
    Coordinates: X,Y,Z = East, North, Up.
    Args:
        tau_ss: strike shear stress
        tau_dd: dip shear stress
        strike: strike angle of the fault plane
        dip: dip angle of the fault plane
        stress_tensor: stress tensors; (3, 3) or (n, 3, 3) where n is the number of stress tensors and also the lengths
        of the strike and dip.
        angle: 'degrees' or 'radians'

    Returns: the maximum shear stress on the plane, as well as the rake of the maximum shear stress value.
    """

    if tau_ss is None:
        tau_ss = get_strike_shear_stress(
            strike=strike,
            dip=dip,
            stress_tensor=stress_tensor,
            angle=angle)
    if tau_dd is None:
        tau_dd = get_dip_shear_stress(
            strike=strike,
            dip=dip,
            stress_tensor=stress_tensor,
            angle=angle)

    tau_max = (tau_ss ** 2 + tau_dd ** 2) ** 0.5

    tau_rake = get_rake_from_shear_components(
        strike_shear=tau_ss, dip_shear=tau_dd, angle=angle
    )
    return tau_max, tau_rake


def stress_angle2tensor(
        x_stress=None,
        y_stress=None,
        stress_angle=None,
        angle='degrees'
):
    """
    Construct plane tectonic stress.
    Args:
        x_stress: float or array; amount of stress in x axis.
        y_stress: float or array; amount of stress in y axis
        stress_angle: float or array; azimuth of x_stress, i.e. between x_stress and East (-90~90)
        angle: 'degrees' or 'radians'

    Returns: (3,3) or (n, 3, 3) where n is the length of inputs.

    """
    stress_angle = -stress_angle
    if angle == 'degrees':
        max_stress_angle_rad = np.deg2rad(stress_angle)
    else:
        max_stress_angle_rad = stress_angle

    A = np.array([
        [np.cos(max_stress_angle_rad), np.sin(max_stress_angle_rad), 0.0],
        [-np.sin(max_stress_angle_rad), np.cos(max_stress_angle_rad), 0.0],
        [0.0, 0.0, 1.0]
    ])

    if type(x_stress) in [np.float64, float, int]:

        max_T = np.array(
            [[x_stress, 0.0, 0.0],
             [0.0, y_stress, 0.0],
             [0.0, 0.0, 0.0]])

        T = A.dot(max_T).dot(A.T)

    else:

        max_T = np.zeros([len(x_stress), 3, 3])
        max_T[:, 0, 0] = x_stress
        max_T[:, 1, 1] = y_stress

        As = np.expand_dims(A, 0). \
            repeat(len(x_stress), axis=0)
        T_tmp = np.array(
            list(map(np.dot, As, max_T)))
        As_T = np.expand_dims(A.T, 0). \
            repeat(len(x_stress), axis=0)
        T = np.array(
            list(map(np.dot, T_tmp, As_T)))

    return T


def tensor_rotate(
        tensor=None,
        angles=None,
        angle='degrees'
):

    angle_x, angle_y, angle_z = angles

    if angle == 'degrees':
        angle_x = np.deg2rad(angle_x)
        angle_y = np.deg2rad(angle_y)
        angle_z = np.deg2rad(angle_z)

    rotate_z = np.array([
        [np.cos(angle_z), -np.sin(angle_z), 0.0],
        [np.sin(angle_z), np.cos(angle_z), 0.0],
        [0.0, 0.0, 1.0]
    ])
    rotate_y = np.array([
        [np.cos(angle_y), 0.0, -np.sin(angle_y)],
        [0.0, 1.0, 0.0],
        [np.sin(angle_y), 0.0, np.cos(angle_y)]

    ])
    rotate_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(angle_x), -np.sin(angle_x)],
        [0.0, np.sin(angle_x), np.cos(angle_x)]

    ])

    rotate_all = rotate_x.dot(rotate_y).dot(rotate_z)

    output_tensor = rotate_all.dot(tensor).dot(rotate_all.T)

    return output_tensor


def list_tensors2array_tensors(tensors):
    """
    Translate tensor format from [xx, xy, xz, yy, yz, zz] to
    [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]].
    Input can be the array of some tensors.
    """
    tensor_n = len(tensors)
    out_tensors = np.zeros([tensor_n, 3, 3])
    out_tensors[:, 0, 0] = tensors[:, 0]
    out_tensors[:, 0, 1] = tensors[:, 1]
    out_tensors[:, 0, 2] = tensors[:, 2]
    out_tensors[:, 1, 0] = tensors[:, 1]
    out_tensors[:, 1, 1] = tensors[:, 3]
    out_tensors[:, 1, 2] = tensors[:, 4]
    out_tensors[:, 2, 0] = tensors[:, 2]
    out_tensors[:, 2, 1] = tensors[:, 4]
    out_tensors[:, 2, 2] = tensors[:, 5]

    return out_tensors


def gen_tectonic_tensor(
        shmax=5.0,
        direction=45.0,
        shmin=0.0,
        stress_log=False,
        pure_shear=False,
        both_pull=False
):

    if stress_log:
        shmax = 10 ** shmax
        shmin = -1.0 * 10 ** shmin

    if pure_shear:
        shmin = -1.0 * shmax

    if both_pull:
        shmax = -1.0 * shmax

    tectonic_tensor = stress_angle2tensor(
        x_stress=shmax,
        y_stress=shmin,
        stress_angle=direction,
        angle='degrees'
    )

    return tectonic_tensor


def gen_depth_dependent_tectonic_tensor(
        depths=None,
        shmax_g=None,
        shmin_g=None,
        shmax_direction=None,
        constant=0.0,
        density=3.3e3,
        g=9.8
):
    if constant != 0.0:
        constant = 10 ** constant

    shmax = shmax_g * density * g * depths
    shmin = shmin_g * density * g * depths - constant

    tectonic_tensor = stress_angle2tensor(
        x_stress=shmax,
        y_stress=shmin,
        stress_angle=shmax_direction,
        angle='degrees'
    )
    return tectonic_tensor


def summarize_2tensors(
        s_list=None,
        a_list=None,
        type_list=None,
        stress_log=False,
):
    tectonic_tensor = np.zeros([3, 3])
    for s_index, s in enumerate(s_list):
        a = a_list[s_index]
        type = type_list[s_index]
        if type == 'dilatation':
            shmax = -np.inf
            shmin = s
            pure_shear = False
        elif type == 'pure_shear':
            shmax = s
            shmin = -np.inf
            pure_shear = True
        tectonic_tensor += gen_tectonic_tensor(
            shmax=shmax,
            direction=a,
            shmin=shmin,
            stress_log=stress_log,
            pure_shear=pure_shear,
        )

    return tectonic_tensor


def gen_rock_tensor(depth=None,
                    density=2.6e3,
                    g=9.8,
                    poisson=2.5
                    ):
    lateral_c = poisson / (1.0 - poisson)
    rock_v = density * g * depth
    rock_h = rock_v * lateral_c

    return rock_h, rock_v


if __name__ == '__main__':
    a = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])
    print(tensor_rotate(a, [30, 0, 0]))
    # print(stress_angle2tensor(max_stress_magnitude=1.0, min_stress_magnitude=0, max_stress_angle=30))
    # v = get_norm_vector(
    #     strike=[1,2, 3],
    #     dip=[1,2, 3],
    #     angle='degrees')
    # s=stress_angle2tensor(max_stress_magnitude=1,
    #                     min_stress_magnitude=0,
    #                     max_stress_angle=-25,
    #                     angle='degrees')

    # norm_v = get_norm_vector(strike=180, dip=45, angle='degrees')
    # strike_v = get_strike_vector(strike=180, angle='degrees')
    # dip_v = get_dip_vector(strike=180, dip=45, angle='degrees')
    # rake = get_rake_from_shear_components(strike_shear=-1, dip_shear=-0.1, angle='degrees')
    # n0 = np.array([1, 0, 0])
    # t0 = np.array([[1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    # print(n0.dot(t0).dot(n0.T))
    #
    # max_stress = 1
    # max_stress_angle = 45
    # t = stress_angle2tensor(max_stress, max_stress_angle)
    # n = np.array([np.sqrt(2) / 2, np.sqrt(2) / 2, 0])
    # print(n.dot(t).dot(n.T))
    # print(get_dip_vector(strike=180, dip=90, angle='degrees'))

    # print(get_strike_shear_stress(strike=135,
    #                                 dip=45,
    #                                 stress_tensor=np.array([[1, 0, 0],
    #                                                         [0, -1, 1],
    #                                                         [0, 1, 1]]),
    #                           angle='degrees'))

    pass
