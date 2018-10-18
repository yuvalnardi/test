import pandas as pd

from src.utils.logger import log
from src.feature_engineering.main.feature_engineering_params import FeatureEngineeringParams


class FeatureEngineeringRunner(object):

    def __init__(self, config_file_full_path):
        assert isinstance(config_file_full_path, str)
        feature_engineering_params = FeatureEngineeringParams(config_file_full_path)

        meta_params = feature_engineering_params.get_meta_params()
        self._client_name = meta_params.get_client_name()
        self._number_of_batches = meta_params.get_number_of_batches()
        self._run_in_parallel = meta_params.get_run_in_parallel()
        self._raw_data_dir = meta_params.get_raw_data_dir()
        self._design_matrix_dir = meta_params.get_design_matrix_dir()

        self._time_series_features_enricher = feature_engineering_params.get_time_series_features_enricher()
        self._get_feature_extractors = feature_engineering_params.get_feature_extractors()

    def run(self):
        log.debug('Running feature engineering ..')

        feature_engineering_main_output = pd.DataFrame({})

        log.debug('Done running feature engineering.')

        return feature_engineering_main_output


if __name__ == '__main__':
    config_file_full_path = '/Users/yuval/Desktop/test/resources/config/feature_engineering_config.yml'

    feature_engineering_runner = FeatureEngineeringRunner(config_file_full_path)
    feature_engineering_main_output = feature_engineering_runner.run()
