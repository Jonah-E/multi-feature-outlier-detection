from torch.utils.data import Dataset
import pytplot
import pandas as pd
import numpy as np

class SpedasWrapper(Dataset):
    """
    Wrapper for loading varibles from pyspedas (tplot) into a pytorch dataset. The loading
    is a beast effort and might not work for some varibles and missions. The varibles are
    assumed to already be loaded into tplot in the correct timerange.

    Parameters
    ----------
    tplot_vars : list of strings
        The varibles to load.
    dropna : bool (default True)
        Drop times where one of the varibles have value NAN. May result in removal of all
        data if tplot_vars are sampled at different times and resample is not set.
    resample : string (default None)
        Time interval for resampling, follows the pandas style for resample. No resampling
        is done if set to None.
    transform : callable
        Transform to apply to each sample.

    """
    def __init__(self, tplot_vars, dropna = True, resample = None, transform = None):
        self.dataset = None
        self.features = []
        feature_cnt = 0
        for var in tplot_vars:
            pre = None
            if len(var) > 1:
                var, pre = var
            data = pytplot.get_data(var)
            if len(data) == 2:
                time, data = data
            elif len(data) == 3:
                time, data, a = data

            names = pytplot.get_data(var, metadata = True)['CDF']['LABELS']
            if names is None:
                names = ['']
            else:
                names = [f'_{n}' for n in names]

            if data.ndim > 2:
                raise ValueError(f'Cannot handle {data.ndim} dimentions!')
            elif data.ndim == 2:
                if data.shape[1] != len(names):
                    names = [f'_{i:02}' for i in range(data.shape[1])]

            if pre is None:
                names = [f'{var}{n}' for n in names]
            else:
                names = [f'{pre}{n}' for n in names]

            self.features.append((feature_cnt, feature_cnt+len(names)))
            feature_cnt += len(names)
            if data.ndim > 1:
                tmp = pd.DataFrame({k: data[:,i] for i,k in enumerate(names)},
                    index = pd.to_datetime(time, unit='s'))
            else:
                tmp = pd.DataFrame({k: data[:] for k in names},
                    index = pd.to_datetime(time, unit='s'))
            if self.dataset is None:
                self.dataset = tmp
            else:
                self.dataset = self.dataset.join(tmp, how = 'outer')

        self.resampled = False
        if resample:
            self.resampled = True
            self.dataset = self.dataset.resample(resample).mean()

        if dropna:
            self.dataset = self.dataset.dropna(axis='columns', how='all')
            self.dataset = self.dataset.dropna(axis='index', how='any')

        self.transform = transform
        self.length = len(self.dataset.index)

    def __len__(self):
        return self.length

    def __getitem__(self,idx):
        if not isinstance(idx, int):
            raise ValueError('Expected idx to be an integer value')
        data = np.array([d for d in self.dataset.iloc[idx]])

        if self.transform:
            data = self.transform(data)

        return (data,)

    def get_dataframe(self):
        """
        Get the full pandas DataFrame

        Returns
        -------
        dataset : pandas DataFrame
            The full loaded data.
        """
        return self.dataset
