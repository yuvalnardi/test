import pandas as pd

from src.feature_engineering.feature_extractors.base import FeatureExtractorBase


class GlobalFeatureExtractor(FeatureExtractorBase):

    def __init__(self):
        super().__init__()

    def extract(self, data):
        design_matrix = pd.DataFrame({})

        return design_matrix
