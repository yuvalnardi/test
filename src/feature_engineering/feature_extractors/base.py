from src.utils.logger import log


class FeatureExtractorBase(object):

    def __init__(self):
        pass

    def extract_features(self, data):
        design_matrix = self.extract(data)

        return design_matrix

    def extract(self, data):
        log.warn('Warning: Empty implementation at top level Model class. Must be overriden by a subclass.')
        return None