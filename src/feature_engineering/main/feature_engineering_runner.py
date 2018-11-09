import numpy as np
import pandas as pd
import time

from src.utils.logger import log
from src.feature_engineering.main.feature_engineering_params import FeatureEngineeringParams
from src.feature_engineering.feature_extractors.composite_feature_extractor import CompositeFeatureExtractor
from src.feature_engineering.feature_extractors.global_feature_extractor import GlobalFeatureExtractor
from src.feature_engineering.feature_extractors.temporal_feature_extractor import TemporalFeatureExtractor

# pd.set_option('display.max_rows', None, 'display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

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
        self._feature_extractor_names = feature_engineering_params.get_feature_extractor_names()

    def run(self):

        log.debug('Running feature engineering ..')
        fe_start_time = time.time()

        # load data
        # data = pd.DataFrame(np.random.normal(0, 1, [10, 2]), columns=['A', 'B'])
        dir = u'D:\\FAMILY\\Yuval\\Work\\Seebo\\'
        file = u'Yuval_TS_Table.csv'
        path = dir + file
        # date_cols = ['end_time_stamp', 'start_time', 'end_time']
        # data = pd.read_csv(path, parse_dates=date_cols, infer_datetime_format=True)
        data = pd.read_csv(path)
        sensors_per_batch = data.groupby('batch_id')['metric_id'].apply(lambda ts: ts.unique())
        if sensors_per_batch.shape[0] > 1:
            # TODO: validate every batch has the same sensors data. Make more elegant
            first_batch_sensors = sensors_per_batch.iloc[0]
            for i in range(1, len(sensors_per_batch.shape[0])):
                assert np.equal(first_batch_sensors, sensors_per_batch.iloc[i]), \
                    'all batches should have the same sensors'

        # TODO: sort by batch_id, metric_id, value
        data = data.sort_values(by=['batch_id', 'metric_id', 'sensor_value'], inplace=False)

        # impute missing values
        # TODO: replace this stupid imputation method
        data['sensor_value'] = data['sensor_value'].fillna(0.0, inplace=False)

        # instantiate composite feature extractor
        # TODO: map self._feature_extractor_names to self._feature_extractor_objects
        gfe = GlobalFeatureExtractor()
        tfe = TemporalFeatureExtractor()
        cfe = CompositeFeatureExtractor([gfe, tfe])
        design_matrix = cfe.extract(data)

        feature_engineering_main_output = design_matrix

        fe_end_time = time.time()
        fe_duration = round((fe_end_time - fe_start_time)/60, 2)
        log.debug('Done running feature engineering [Total time: {} mins.].'.format(fe_duration))

        return feature_engineering_main_output


if __name__ == '__main__':
    # config_file_full_path = '/Users/yuval/Desktop/test/resources/config/feature_engineering_config.yml'
    config_file_full_path = 'D:\\FAMILY\\Yuval\\Work\\Seebo\\test\\resources\\config\\feature_engineering_config.yml'

    feature_engineering_runner = FeatureEngineeringRunner(config_file_full_path)
    feature_engineering_main_output = feature_engineering_runner.run()
