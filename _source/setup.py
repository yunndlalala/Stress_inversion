#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: yunnaidan
@time: 2020/04/08
@file: setup.py
"""
import os
from distutils.sysconfig import get_python_lib

# Set path
print('>>> Set path...')

pwd = os.getcwd()
lib_path = get_python_lib()
pth_file = os.path.join(lib_path, 'my_package.pth')
with open(pth_file, 'w') as f:
    f.write(pwd)