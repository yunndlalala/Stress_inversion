#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: yunnaidan
@time: 2020/09/09
@file: utils.py
"""
import os
import math
import numpy as np


def download_by_tar_str(usr_name=None,
                        local_id=None,
                        server_path=None,
                        tar_str=None,
                        local_path=None):

    if local_id is None or \
            usr_name is None or \
            server_path is None or \
            tar_str is None or \
            local_path is None:
        raise ValueError('Missing input!')

    for dir, folders, files in os.walk(server_path):
        for file in files:
            if tar_str in file:
                file_abs_path = os.path.join(server_path, file)
                output_path = os.join(local_path, dir.replace(server_path, ''))
                if not os.path.exists(output_path):
                    os.mkdirs(output_path)
                print(
                    'scp %s %s@%s:%s ' %
                    (file_abs_path, usr_name, local_id, output_path))
                os.system('scp %s %s@%s:%s ' %
                          (file_abs_path, usr_name, local_id, output_path))
    return None


def get_bearing(lat1, lon1, lat2, lon2):
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)

    dlon = lon2 - lon1

    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)

    bearing = math.atan2(y, x)
    bearing = np.degrees(bearing)
    bearing = (bearing + 360) % 360

    return bearing


# if __name__ == '__main__':
#     print(get_bearing(0, 0, 0, 10))
#     pass

