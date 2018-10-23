import pandas as pd
import time

from src.utils.logger import log
from src.feature_engineering.feature_extractors.base import FeatureExtractorBase


class CompositeFeatureExtractor(FeatureExtractorBase):

    def __init__(self, feature_extractors):
        super().__init__()
        self._feature_extractors = feature_extractors

    def extract(self, data):
        assert isinstance(data, pd.DataFrame)

        log.debug('Running Composite feature extractor ..')
        cfe_start_time = time.time()

        matrices_lst = []
        for fe in self._feature_extractors:
            matrices_lst.append(fe.extractFeatures(data))

        design_matrix = pd.concat(matrices_lst, axis=1)

        cfe_end_time = time.time()
        cfe_duration = round((cfe_end_time - cfe_start_time) / 60, 2)
        log.debug('Done running Composite feature extractor [Total time: {} mins.].'.format(cfe_duration))

        return design_matrix
