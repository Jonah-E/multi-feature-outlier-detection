#!/usr/bin/env python

import datetime as dt

from .pandasdataset import PandasDataset
from .spedaswrapper import SpedasWrapper


_DATASET_DAYSIDE = [
    {'trange': ('2017-12-14/16:00:00', '2017-12-15/05:00:00'),
        'roi': None,
        'tag': 'dayside' },
    {'trange': ('2017-12-17/16:00:00', '2017-12-17/22:00:00'),
        'roi': [('2017-12-17/17:52:00', '2017-12-17/17:54:00')],
        'tag': 'dayside' },
    {'trange': ('2018-01-12/00:50:00', '2018-01-12/06:00:00'),
        'roi': [('2018-01-12/01:50:00', '2018-01-12/01:52:30')],
        'tag': 'dayside' },
    {'trange': ('2018-12-14/02:00:00', '2018-12-14/07:20:00'),
        'roi': [('2018-12-14/04:21:00', '2018-12-14/04:22:00')],
        'tag': 'Out-bound' },
    {'trange': ('2018-12-10/04:00:00', '2018-12-10/11:00:00'),
        'roi': [('2018-12-10/04:40:00', '2018-12-10/04:42:00'),
                ('2018-12-10/05:12:00', '2018-12-10/05:25:00'),
                ('2018-12-10/06:27:00', '2018-12-10/06:31:00')],
        'tag': 'dayside' },
    {'trange': ('2019-01-05/16:00:00', '2019-01-05/19:00:00'),
        'roi': [('2019-01-05/17:38:00', '2019-01-05/17:41:00')],
        'tag': 'dayside' },
    {'trange': ('2021-01-12/00:00:00', '2021-01-12/06:00:00'),
        'roi': [('2021-01-12/01:18:00', '2021-01-12/01:21:00')],
        'tag': 'dayside' },
    {'trange': ('2021-02-13/10:00:00', '2021-02-13/16:00:00'),
        'roi': [('2021-02-13/11:05:00', '2021-02-13/11:06:10')],
        'tag': 'dayside' },
    {'trange': ('2022-05-02/18:00:00', '2022-05-02/22:00:00'),
        'roi': [('2022-05-02/18:23:00', '2022-05-02/18:25:00')],
        'tag': 'dayside' },
    {'trange': ('2022-11-24/02:00:00', '2022-11-24/09:00:00'),
        'roi': [('2022-11-24/04:16:00', '2022-11-24/04:18:00')],
        'tag': 'dayside' },
    {'trange': ('2023-01-16/06:00:00', '2023-01-16/11:00:00'),
        'roi': [('2023-01-16/08:21:00', '2023-01-16/08:24:00')],
        'tag': 'dayside' },
]

_DATASET_NIGHTSIDE = [
    {'trange': ('2017-07-23/12:00:00', '2017-07-23/18:00:00'),
        'roi': [('2017-07-23/16:55:00', '2017-07-23/16:56:00')],
        'tag': 'nightside' },
    {'trange': ('2021-08-14/16:00:00', '2021-08-15/06:00:00'),
        'roi': [('2021-08-15/01:23:00', '2021-08-15/01:25:00')],
        'tag': 'nightside' },
]

def get_dataset(selection = None):
    dataset_prototype = _DATASET_DAYSIDE + _DATASET_NIGHTSIDE
    if selection == 'day':
        dataset_prototype = _DATASET_DAYSIDE
    elif selection == 'night':
        dataset_prototype = _DATASET_NIGHTSIDE

    dataset = []
    for i, v in enumerate(dataset_prototype):
        t1_s = v['trange'][0]
        t2_s = v['trange'][1]

        roi = None
        if v['roi']:
            roi = [(dt.datetime.strptime(d[0], '%Y-%m-%d/%H:%M:%S'),
                    dt.datetime.strptime(d[1], '%Y-%m-%d/%H:%M:%S'))
                        for d in v['roi']]

        tag = None
        if 'tag' in v:
            tag = v['tag']
        dataset.append({'name' : f'data/{t1_s[:10]}_resampled.feather',
                        'trange' : ( t1_s, t2_s),
                        'roi' : roi,
                        'tag': tag})
    return dataset


if __name__ == '__main__':
    datasets = get_dataset(None)
