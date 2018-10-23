import pandas as pd

from src.feature_engineering.feature_extractors.base import FeatureExtractorBase


class CompositeFeatureExtractor(FeatureExtractorBase):

    def __init__(self, feature_extractors):
        super().__init__()
        self._feature_extractors = feature_extractors

    def extract(self, data):

        assert isinstance(data, pd.DataFrame)
        matrices_lst = []
        for fe in self._feature_extractors:
            matrices_lst.append(fe.extractFeatures(data))

        design_matrix = pd.concat(matrices_lst, axis=0)

        return design_matrix
