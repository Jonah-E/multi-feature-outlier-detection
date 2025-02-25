"""
Module containing the FeatureCoupledMinMaxScaler and OutlierDetector
"""
from sklearn.decomposition import IncrementalPCA
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

def _flag(x):
    """
    Helper function to create a flag for marking samples as outlier or
    calibration.
    """
    if x['Calibration'] == 1:
        return 2
    #Holdout samples are marked as outliers
    if x['Calibration'] == -1:
        return 1
    if x['Outlier'] == True:
        return 1
    return 0
def _flag_str(x, col):
    """
    Helper function to create a string version of the flag.
    """
    if x[col] == 2:
        return '2: Calibrating'
    if x[col] == 1:
        return '1: Outlier'
    return '0: No activity'


class FeatureCoupledMinMaxScaler():
    """
    Scaling the data according to Min and Max values for that feature.

    Parameters
    ----------
    fetaures : list of tuples [start, end)
        List of tuples with the ranges where each feature type can be found.
    feature_range : tuple [min, max]
        The range to scale between.

    Examples
    --------
    >>> from detector import FeatureCoupledMinMaxScaler
    >>> features = [(0,3),(3,6)]
    >>> scaler = FeatureCoupledMinMaxScaler(features)
    >>> scaler.fit(x)
    >>> x = scaler.transform(x)

    """
    def __init__(self, features, feature_range = (0,1)):
        self.features = features
        self._max = [0 for _ in features]
        self._min = [0 for _ in features]
        self._scale = [0 for _ in features]
        self.feature_range = feature_range


    def fit(self, x):
        """
        Fit scaler to data.

        Parameters
        ----------

            X : ndarray of shape (n_samples, n_features)
                The data to fit.

        Returns
        _______
            self : object
                The fitted scaler.
        """
        for i, (i1, i2) in enumerate(self.features):
            self._min[i] = np.min(x[:,i1:i2])
            self._max[i] = np.max(x[:,i1:i2])
            data_range = self._max[i] - self._min[i]
            self._scale[i] = (self.feature_range[1] - self.feature_range[0]) / data_range
            self._min[i] = self.feature_range[0] - self._min[i] * self._scale[i]

        return self

    def transform(self, x):
        """
        Scale data according to scaler.

        Parameters
        ----------
        X : Array of shape (n_samples, n_features)
            The data to scale

        Returns
        -------
        X_tr : ndarray of shape (n_samples, n_features)
            The scaled version of X.
        """
        for i, (i1, i2) in enumerate(self.features):
            x[:,i1:i2] =  (x[:,i1:i2] * self._scale[i]) + self._min[i]
        return x

class OutlierDetector():
    """
    Detector for finding outliers in streaming data.

    Parameters
    ----------
    calib_batch_size : integer
        The number of samples required to initiate a calibration.
    mean_window : integer
        The number of reconstruction error from samples to use for calculating the current
        mean and standard deviation, used in setting the current threshold for outliers.
    n_components : integer
        The number of components to calculate in the PCA.
    error_threshold : float
        The threshold for labaling a sample as an outlier.
    outlier_limit : integer
        The number of samples allowed to be outliers before they start to be counted towards
        a new calibration.
    store_components : bool (default True)
        Store the PCA components in the history data.

    """
    def __init__(self, outlier_limit = None, error_threshold = 3, mean_window = None,
                 n_components = 1, calib_batch_size = None, store_components = True):

        self.n_components = n_components

        self.calib_batch_size = calib_batch_size

        self.error_threshold = error_threshold
        self.outlier_limit = outlier_limit
        self._mean_buffer_filled = False

        self.mean_window = mean_window
        self._calib_mode = 'Init'
        self._calib_buf = None
        self._calib_buf_ptr = 0
        self._mean_ptr = 0
        self._mean_buffer_reset = False
        self._mean = None

        self.store_components = store_components
        self._history_index = 0
        self._history = pd.DataFrame({
                 'Error' : [],
                 'Threshold' : [],
                 'Threshold Alt.' : [],
                 'Time' : [],
                 'Mean' : [],
                 'Std' : [],
                 'Outlier' : [],
                 'Calibration': [],})

    def _calib(self, sample):
        """
        Internal function for updating the PCA
        """
        self.pca.partial_fit(sample)

    def _setup(self, samples, features):
        """
        Internal function for setuing up buffers and unset parameters.
        """

        if self.calib_batch_size is None:
            self.calib_batch_size = samples

        if self._calib_buf is None:
            self._calib_buf = np.zeros((self.calib_batch_size + self.outlier_limit, features))
        self._features = features

        if self.mean_window == None:
            self.mean_window = self.calib_batch_size
        self._mean_buffer = np.zeros((self.mean_window,))

        if self.n_components is None:
            self.n_components = self.calib_batch_size
        self.pca = IncrementalPCA(n_components = self.n_components)

        if self.outlier_limit is None:
            self.outlier_limit = int(round(self.calib_batch_size/2,0))

    def _first_calib(self,sample):
        """
        Internal function for creating the first PCA.
        """
        calib_range = None
        self._setup(*sample.shape)
        ptr = self._calib_buf_ptr + sample.shape[0]
        if ptr <= self.calib_batch_size:
            self._calib_buf[self._calib_buf_ptr:ptr,:] = sample
            self._calib_buf_ptr = ptr
            ptr = sample.shape[0]
        else:
            ptr = self.calib_batch_size - self._calib_buf_ptr
            self._calib_buf[self.calib_buf_ptr:,:] = sample[:ptr,:]
            self._calib_buf_ptr += ptr

            sample = sample[ptr:,:]

        if self._calib_buf_ptr == self.calib_batch_size:
            self._calib(self._calib_buf[:self.calib_batch_size,:])
            self._history.loc[:(self.calib_batch_size-1), 'Calibration'] = 1 #Pandas slicing is end-inclusive
            self._history['Components'] = [np.zeros(self.pca.components_.shape) for _ in range(self.calib_batch_size) ]

            self._calib_mode = 'Check'
            self._calib_buf_ptr = 0
            calib_range = (ptr - self.calib_batch_size, ptr)

        elif not sample.shape[0] == 0:
            raise Exception('Loosing data')

        return calib_range

    def _mean_n_std(self, x):
        """
        Internal function for calculating the current mean and standard deviation.
        """
        mean, std = (0,0)
        if self._mean_buffer_filled:
            mean = np.mean(self._mean_buffer)
            std = np.std(self._mean_buffer)
        elif self._mean_ptr > 0:
            mean = np.mean(self._mean_buffer[:self._mean_ptr])
            std = np.std(self._mean_buffer[:self._mean_ptr])

        return mean, std

    def _add_to_mean_buffer(self, x):
        """
        Internal function for adding a sample error to the mean buffer.
        """

        self._mean_buffer[self._mean_ptr] = x
        self._mean_ptr += 1

        if self._mean_ptr >= self.mean_window:
            self._mean_buffer_filled = True
            self._mean_ptr = 0


    def _error(self, sample):
        """
        Internal function for calculating the current reconstruction error.
        """
        p = self.pca.transform(sample)
        return np.linalg.norm(sample - self.pca.inverse_transform(p), axis = 1)

    def _outlier_cond(self, error, ptr):
        """
        Internal function for checking for outliers.
        """
        nr_errors = error.shape[0]
        outlier = np.zeros((nr_errors))
        err_thres = np.zeros((nr_errors))
        mean = np.zeros((nr_errors))
        std = np.zeros((nr_errors))

        for i, e in enumerate(error):
            mean[i], std[i] = self._mean_n_std(e)

            # If the mean buffer where reset add the first samples to the mean buffer
            if (mean[i] == 0 or std[i] == 0) and self._mean_buffer_reset:
                self._add_to_mean_buffer(e)
                continue
            self._mean_buffer_reset = False

            err_thres[i] = mean[i] + self.error_threshold * std[i]

            # Check for outlier
            if e > err_thres[i]:
                outlier[i] = True

            ## Only mark as outliers if the mean buffer is filled
            if not self._mean_buffer_filled:
                outlier[i] = False
                self._add_to_mean_buffer(e)
            elif outlier[i] == False:
                #Only add non outliers to mean buffer
                self._add_to_mean_buffer(e)

        # Update history
        start = self._history_index + ptr
        end = start + nr_errors -1
        self._history.loc[start:end, 'Threshold'] = err_thres[:]
        self._history.loc[start:end, 'Mean'] = mean
        self._history.loc[start:end, 'Std'] = std

        return outlier

    def _calib_cond(self, sample, outlier):
        """
        Internal function to check if the calibration condition is met.
        """
        ptr = 0
        for i,d in enumerate(outlier):
            if not d:
                self._calib_buf_ptr = 0
                continue

            #The outlier limit is checked by simply adding everythin to the buffer
            self._calib_buf[self._calib_buf_ptr] = sample[i]
            self._calib_buf_ptr += 1

            if self._calib_buf_ptr == (self.calib_batch_size + self.outlier_limit):
                ptr = i+1

                #Mark samples counting towards the outlier limit
                start = self._history_index + (ptr - self._calib_buf_ptr)
                end = start + self.outlier_limit - 1
                self._history.loc[start:end, 'Calibration'] = -1

                #Mark calibration samples
                start = end + 1
                end = start + self.calib_batch_size - 1
                self._history.loc[start:end, 'Calibration'] = 1

                self._calib_mode = 'Calib'

                return ptr, sample[ptr:,:].shape[0] > 0

        return ptr, False

    def get_history(self):
        """
        Get the detector history as a pandas dataframe.

        Returns
        -------
        history : pandas DataFrame
            The history of the detector.
        """
        tmp = self._history
        tmp['Outlier'] = tmp['Outlier'].astype(bool)
        tmp['Flag'] = tmp[['Outlier', 'Calibration']].apply(_flag, axis = 1)
        tmp['Flag '] = tmp[['Flag']].apply(lambda x: _flag_str(x,'Flag'), axis = 1)
        return tmp

    def reset_mean(self):
        """
        Reset the mean buffer. Removes all the current samples from the mean buffer, causing
        all samples to be labeled as not outliers untill the buffer is filled again.
        """
        self._mean_ptr = 0
        self._mean_buffer_filled = False
        self._mean_buffer_reset = True

    def reset_history(self):
        """
        Reset the detector history.
        """
        self._history = pd.DataFrame({
                 'Error' : [],
                 'Threshold' : [],
                 'Time' : [],
                 'Outlier' : [],
                 'Calibration': [],})

    def __call__(self, sample, time = None):
        """
        Check samples for outliers. It is recommended not to call with more samples than can fit in the calibration buffer.

        Parameters
        ----------
        samples : ndarray of shape (n_samples, n_features)
            The samples to check for outliers. It is recomended to have
            n_samples <= calib_batch_size
        time : (optional) ndarray of size n_samples
            The time values correspondig to the samples, only added to history.

        """
        calib_range = None
        nr_samples = sample.shape[0]
        tmp = pd.DataFrame({
                     'Error' : np.zeros((nr_samples)),
                     'Threshold' : np.zeros((nr_samples)),
                     'Threshold Alt.' : np.zeros((nr_samples)),
                     'Mean' : np.zeros((nr_samples)),
                     'Std' : np.zeros((nr_samples)),
                     'Outlier' : np.zeros((nr_samples)),
                     'Calibration': np.zeros((nr_samples))
            })
        if time:
            tmp['Time'] = time[:]
        if self.store_components and self._calib_mode != 'Init':
            tmp['Components'] = [self.pca.components_ for _ in range(nr_samples)]

        tmp.index += self._history_index
        self._history = pd.concat([self._history, tmp])

        check = True
        if self._calib_mode == 'Init':
            check = self._first_calib(sample)

        error = np.zeros((nr_samples))
        outlier = np.zeros((nr_samples))
        ptr = 0

        while check:
            if self._calib_mode == 'Check':

                #Calculate the reconstruction error.
                error[ptr:] = self._error(sample[ptr:,:])

                #Check the outlier condition
                outlier[ptr:] = self._outlier_cond(error[ptr:], ptr)

                ptr, check = self._calib_cond(sample[ptr:,:], outlier[ptr:])


            if self._calib_mode == 'Calib':
                self._calib(self._calib_buf[self.outlier_limit:])
                if self.store_components:
                    start = self._history_index + ptr
                    self._history.loc[start:, 'Components'] = [self.pca.components_ for _ in range(len(sample[ptr:,:]))]
                self._calib_mode = 'Check'
                self._calib_buf_ptr = 0

        self._history.loc[self._history_index:, 'Error'] = error
        self._history.loc[self._history_index:, 'Outlier'] = outlier
        self._history_index += nr_samples

        return outlier

def run_detector(detector, dataset, batch_size, verbose = False):
    """
    Run the detector for a torch dataset.
    """
    time = dataset.dataset.index

    iterator = enumerate(DataLoader(dataset, batch_size=batch_size))
    if verbose:
        dataloader = DataLoader(dataset, batch_size=batch_size)
        iterator = tqdm(enumerate(dataloader), total = len(dataloader))

    for i, (x, ) in iterator:
        x = x.numpy()
        _ = detector(x)

    tmp = detector.get_history()
    tmp['Time'] = time

    detector.reset_history()
    return tmp

