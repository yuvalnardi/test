import numpy as np
import pandas as pd
import time

from src.utils.logger import log
from src.feature_engineering.feature_extractors.base import FeatureExtractorBase


class TemporalFeatureExtractor(FeatureExtractorBase):

    def __init__(self):
        super().__init__()

    def extract(self, data):

        assert isinstance(data, pd.DataFrame)
        # assert that data have no missing values
        assert not pd.isnull(data).values.any(), 'data should not contain missing values.'

        log.debug('Running Temporal feature extractor ..')
        tfe_start_time = time.time()

        design_matrix = data.groupby(['batch_id', 'metric_id'])['end_time_stamp'].aggregate(self._timespan)

        tfe_end_time = time.time()
        tfe_duration = round((tfe_end_time - tfe_start_time) / 60, 2)
        log.debug('Done running Temporal feature extractor [Total time: {} mins.].'.format(tfe_duration))

        return design_matrix

    def _timespan(self, ts):
        # TODO
        return ts.min()

