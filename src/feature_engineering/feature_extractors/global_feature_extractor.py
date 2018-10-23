import numpy as np
import pandas as pd
import time

from src.utils.logger import log
from src.feature_engineering.feature_extractors.base import FeatureExtractorBase


class GlobalFeatureExtractor(FeatureExtractorBase):

    def __init__(self):
        super().__init__()

    def extract(self, data):

        assert isinstance(data, pd.DataFrame)

        log.debug('Running Global feature extractor ..')
        gfe_start_time = time.time()

        design_matrix = pd.DataFrame(np.random.normal(0, 1, [5, 2]), columns=['X1', 'X2'])

        gfe_end_time = time.time()
        gfe_duration = round((gfe_end_time - gfe_start_time) / 60, 2)
        log.debug('Done running Global feature extractor [Total time: {} mins.].'.format(gfe_duration))

        return design_matrix
