import numpy as np
import pandas as pd

from src.feature_engineering.feature_extractors.base import FeatureExtractorBase


class TemporalFeatureExtractor(FeatureExtractorBase):

    def __init__(self):
        super().__init__()

    def extract(self, data):
        design_matrix = pd.DataFrame(np.random.normal(10, 1, [5, 3]), columns=['Z1', 'Z2', 'Z3'])

        return design_matrix
