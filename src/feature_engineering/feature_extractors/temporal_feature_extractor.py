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

        log.debug('Running Temporal feature extractor ..')
        tfe_start_time = time.time()

        design_matrix = pd.DataFrame(np.random.normal(10, 1, [5, 3]), columns=['Z1', 'Z2', 'Z3'])

        tfe_end_time = time.time()
        tfe_duration = round((tfe_end_time - tfe_start_time) / 60, 2)
        log.debug('Done running Temporal feature extractor [Total time: {} mins.].'.format(tfe_duration))

        return design_matrix
