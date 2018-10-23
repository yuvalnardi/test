import numpy as np
import pandas as pd

from src.feature_engineering.feature_extractors.base import FeatureExtractorBase


class GlobalFeatureExtractor(FeatureExtractorBase):

    def __init__(self):
        super().__init__()

    def extract(self, data):
        design_matrix = pd.DataFrame(np.random.normal(0, 1, [5, 2]), columns=['X1', 'X2'])

        return design_matrix
