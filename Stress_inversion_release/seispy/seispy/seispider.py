#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: yunnaidan
@time: 2019/11/22
@file: seispider.py
"""
import re
import requests
import logging

# IncompleteRead error occur sometimes.
# A violent solution is as follows. (I haven't understand.)
# import httplib
# httplib.HTTPConnection._http_vsn = 10
# httplib.HTTPConnection._http_vsn_str = 'HTTP/1.0'


class FDSNWS(object):

    def __init__(self,
                 home_url='http://service.iris.edu',
                 log_file='log.txt'):
        self.home_url = home_url
        logging.basicConfig(
            filename=log_file,
            format='%(asctime)s-%(name)s-%(levelname)s-%(module)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S %p',
            level=logging.INFO,
            filemode='w')

    def request(self,
                task='dataselect',
                **kwargs
                ):
        # Build data url.
        parameters = [str(k) + '=' + str(v)
                      for k, v in kwargs.items() if v is not None]
        sub_url = '/fdsnws/' + task + '/1/query?' + '&'.join(parameters)
        url = self.home_url + sub_url
        # Request the url.
        try:
            r = requests.get(url, timeout=10, stream=True)
            if r.status_code in [204, 400, 404]:
                print('<Response[%i]>' % r.status_code)
                print(r.text)
                logging.warning('\n' + url +
                                '\n<Response[%i]>' % r.status_code +
                                '\n' + r.text)
                output = None
            else:
                output = r

        except Exception as err_msg:
            print('ERROR: %s' % err_msg)
            logging.warning('ERROR: %s' % err_msg)
            output = None

        return output

    def download(self,
                 out_file='test.mseed',
                 progress_bar=True,
                 r=None,
                 task='dataselect',
                 **kwargs):
        if r is None:
            r = self.request(
                task=task,
                **kwargs
            )
        if r is not None:
            print('<Response[%i]>' % r.status_code)
            with open(out_file, "wb") as f:
                chunk_n = 0
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        chunk_n += 1
                        if progress_bar:
                            print(
                                '\r Finished: %.1fMB' %
                                (chunk_n / 1024), end='')
                    else:
                        raise ValueError('Chunk error!')
                print('\n')
        return None


class IRISWS(object):

    def __init__(self):
        self.home_url = 'http://service.iris.edu'

    def sacpz(self,
              net='IU',
              sta='ANMO',
              cha='00',
              loc='BHZ',
              start='2010-02-27T06:30:00.000',
              end='2010-02-27T10:30:00.000',
              out_file='example.PZ'):
        sub_url = '/irisws/sacpz/1/query?net={0}&sta={1}&cha={2}&loc={3}&start={4}&end={5}'.format(
            net, sta, cha, loc, start, end)
        url = self.home_url + sub_url
        for times in range(10):
            r = requests.get(url)
            if r:
                break
        if not r:
            raise ValueError('Request error!')

        text = r.text
        lines = text.split('\n')

        node_index = [-1]
        for l_index, line in enumerate(lines):
            first_str = re.split('\s+', line)[0]
            if first_str == 'CONSTANT':
                node_index.append(l_index)

        for i in range(len(node_index) - 1):
            out_file_i = out_file + '_' + str(i)
            with open(out_file_i, 'w') as f:
                for l in range(node_index[i] + 1, node_index[i + 1] + 1):
                    f.writelines(lines[l] + '\n')
        return None


class SCEDCWS(object):
    def __init__(self):
        self.home_url = 'http://service.scedc.caltech.edu'

    def sacpz(self,
              net='CI',
              sta='ADO',
              loc='--',
              cha='BHN',
              start='2009-01-01T00:00:00',
              end='2019-06-30T23:59:59',
              out_file='example.pz',
              nodata='404'):
        sub_url = '/scedcws/sacpz/1/query?net={0}&sta={1}&cha={2}&loc={3}&start={4}&end={5}&nodata={6}'.format(
            net, sta, cha, loc, start, end, nodata)
        url = self.home_url + sub_url
        for times in range(10):
            r = requests.get(url)
            if r:
                break
        if not r:
            raise ValueError('Request error!')

        text = r.text
        lines = text.split('\n')

        node_index = [-1]
        for l_index, line in enumerate(lines):
            first_str = re.split('\s+', line)[0]
            if first_str == 'CONSTANT':
                node_index.append(l_index)

        for i in range(len(node_index) - 1):
            out_file_i = out_file + '_' + str(i)
            with open(out_file_i, 'w') as f:
                for l in range(node_index[i] + 1, node_index[i + 1] + 1):
                    f.writelines(lines[l] + '\n')
        return None


class NCEDCWS(object):
    def __init__(self):
        self.home_url = 'http://service.ncedc.org'

    def sacpz(self,
              net='CI',
              sta='ADO',
              loc='--',
              cha='BHN',
              start='2009-01-01T00:00:00',
              end='2019-06-30T23:59:59',
              out_file='example.pz',
              nodata='404'):
        sub_url = '/ncedcws/sacpz/1/query?net={0}&sta={1}&cha={2}&loc={3}&start={4}&end={5}&nodata={6}'.format(
            net, sta, cha, loc, start, end, nodata)
        url = self.home_url + sub_url
        for times in range(10):
            r = requests.get(url)
            if r:
                text = r.text
                lines = text.split('\n')

                node_index = [-1]
                for l_index, line in enumerate(lines):
                    first_str = re.split('\s+', line)[0]
                    if first_str == 'CONSTANT':
                        node_index.append(l_index)

                for i in range(len(node_index) - 1):
                    out_file_i = out_file + '_' + str(i)
                    with open(out_file_i, 'w') as f:
                        for l in range(
                                node_index[i] + 1, node_index[i + 1] + 1):
                            f.writelines(lines[l] + '\n')

                break

        if not r:
            if r.status_code == 404:
                print('No data!')
                logging.warning('No data!')
            else:
                print('Request error!')
                logging.warning('Request error!')

        return None


class UNAVCO(object):
    def __init__(self):
        self.home_url = 'https://web-services.unavco.org'

    def request(self,
                task='metadata',
                station=None,
                type='application/json',
                **kwargs):
        # Build data url.
        parameters = [str(k) + '=' + str(v)
                      for k, v in kwargs.items() if v is not None]
        if task == 'metadata':
            sub_url_head = '/gps/metadata/sites/v1?'
        elif task == 'position':
            sub_url_head = '/gps/data/position/' + station + '/v3?'
        sub_url = sub_url_head + '&'.join(parameters)
        url = self.home_url + sub_url
        # Request the url.
        try:
            header = {'accept': type}
            r = requests.get(url, timeout=10, stream=True, headers=header)
            if r.status_code in [400, 404, 406, 500, 502, 503]:
                print('<Response[%i]>' % r.status_code)
                print(r.text)
                logging.warning(r.text)
                output = None
            else:
                output = r

        except Exception as err_msg:
            print('ERROR: %s' % err_msg)
            logging.warning('ERROR: %s' % err_msg)
            output = None

        return output

    def download(self,
                 out_file='gps_station.json',
                 progress_bar=True,
                 r=None,
                 task='metadata',
                 station=None,
                 type='application/json',
                 **kwargs):
        if r is None:
            r = self.request(
                task=task,
                station=station,
                type=type,
                **kwargs
            )
        if r is not None:
            print('<Response[%i]>' % r.status_code)
            with open(out_file, "wb") as f:
                chunk_n = 0
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        chunk_n += 1
                        if progress_bar:
                            print(
                                '\r Finished: %.1fMB' %
                                (chunk_n / 1024), end='')
                    else:
                        raise ValueError('Chunk error!')
                print('\n')
        return None

