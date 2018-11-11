import numpy as np
import pandas as pd
import time
import multiprocessing
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters, ComprehensiveFCParameters

from src.utils.logger import log
from src.feature_engineering.feature_extractors.base import FeatureExtractorBase


class GlobalFeatureExtractor(FeatureExtractorBase):

    def __init__(self, params):

        assert isinstance(params, dict)
        self._params = params
        self._feature_calculator_to_params = self._params.get('feature_calculator_to_params')
        self._parallel = self._params.get('parallel')
        assert isinstance(self._parallel, (bool, type(None)))
        if self._parallel is None:
            self._parallel = False
        n_cores_available = multiprocessing.cpu_count()
        self._num_of_cores_to_use = n_cores_available if self._parallel else 1
        super().__init__()

    def extract(self, data):

        assert isinstance(data, pd.DataFrame)
        # assert that data have no missing values
        assert not pd.isnull(data).values.any(), 'data should not contain missing values.'

        log.debug('Running Global feature extractor ..')
        gfe_start_time = time.time()

        # setting time series features to extract or use default
        # fc_parameters = MinimalFCParameters()
        # fc_parameters = EfficientFCParameters()
        # fc_parameters = ComprehensiveFCParameters()

        # feature extraction
        design_matrix = extract_features(data,
                                         default_fc_parameters=self._feature_calculator_to_params,
                                         column_id='batch_id',
                                         column_sort='end_time_stamp',
                                         column_kind='metric_id',
                                         column_value='sensor_value',
                                         n_jobs=self._num_of_cores_to_use)

        # impute: use a builtin tsfresh method that replaces NaN with median and -inf
        # [+inf] with min [max] in a columnwise fashion (and in place)
        # If the column does not contain finite values at all, it is filled with zeros
        # Also, all columns will be guaranteed to be of type np.float64
        # (can also be done by passing impute_function=impute) to extract_features())
        impute(design_matrix)
        # TODO: assert that none cf the columns was filled with zeros
        print(design_matrix.info())

        gfe_end_time = time.time()
        gfe_duration = round((gfe_end_time - gfe_start_time) / 60, 2)

        log.debug('Done running Global feature extractor [Total time: {} mins.].'.format(gfe_duration))

        return design_matrix