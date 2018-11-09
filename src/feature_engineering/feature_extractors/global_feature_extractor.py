import numpy as np
import pandas as pd
import time
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters, ComprehensiveFCParameters

from src.utils.logger import log
from src.feature_engineering.feature_extractors.base import FeatureExtractorBase


class GlobalFeatureExtractor(FeatureExtractorBase):

    def __init__(self, params):

        assert isinstance(params, dict)
        self._params = params
        self._feature_calculator_to_params = self._params.get('feature_calculator_to_params')
        super().__init__()

    def extract(self, data):

        assert isinstance(data, pd.DataFrame)
        # assert that data have no Naassert not pd.isnull(data).values.any(), 'data should not contain missing values or -+infinity.'N, Inf, -Inf values
        data = data.replace([np.inf, -np.inf], np.nan, inplace=False)
        assert not pd.isnull(data).values.any(), 'data should not contain missing values or -+infinity.'

        log.debug('Running Global feature extractor ..')
        gfe_start_time = time.time()

        # setting time series features to extract or use default
        # fc_parameters = MinimalFCParameters()
        # fc_parameters = EfficientFCParameters()
        # fc_parameters = ComprehensiveFCParameters()

        design_matrix = extract_features(data,
                                         default_fc_parameters=self._feature_calculator_to_params,
                                         column_id='batch_id',
                                         column_sort='end_time_stamp',
                                         column_kind='metric_id',
                                         column_value='sensor_value')

        gfe_end_time = time.time()
        gfe_duration = round((gfe_end_time - gfe_start_time) / 60, 2)
        log.debug('Done running Global feature extractor [Total time: {} mins.].'.format(gfe_duration))

        return design_matrix