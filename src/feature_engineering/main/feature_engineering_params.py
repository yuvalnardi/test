from src.feature_engineering.handlers.fe_params_handler import FeatureEngineeringParamsHandler


class MetaParams(object):

    def __init__(self, client_name, number_of_batches, run_in_parallel,
                 raw_data_dir, design_matrix_dir):
        assert isinstance(client_name, str)
        assert isinstance(number_of_batches, (int, type(None)))
        assert isinstance(run_in_parallel, (bool, type(None)))
        assert isinstance(raw_data_dir, (str, type(None)))
        assert isinstance(design_matrix_dir, (str, type(None)))

        self._client_name = client_name
        self._number_of_batches = number_of_batches
        self._run_in_parallel = run_in_parallel
        self._raw_data_dir = raw_data_dir
        self._design_matrix_dir = design_matrix_dir

    def get_client_name(self):
        return self._client_name

    def get_number_of_batches(self):
        return self._number_of_batches

    def get_run_in_parallel(self):
        return self._run_in_parallel

    def get_raw_data_dir(self):
        return self._raw_data_dir

    def get_design_matrix_dir(self):
        return self._design_matrix_dir


class FeatureEngineeringParams(object):

    def __init__(self, params_full_path):

        # config file main sections
        self._meta_params = None
        self._time_series_features_enricher = None
        self._feature_extractors = None

        self.params_full_path = params_full_path
        self._load_params(params_full_path)

    def get_meta_params(self):
        return self._meta_params

    def get_time_series_features_enricher(self):
        return self._time_series_features_enricher

    def get_feature_extractors(self):
        return self._feature_extractors

    def _load_params(self, params_full_path):

        feature_engineering_params_handler = FeatureEngineeringParamsHandler()
        fe_object_to_params = feature_engineering_params_handler.get_params(params_full_path)

        # MetaParams
        section_name = 'MetaParams'
        meta_params = fe_object_to_params.get(section_name)

        # client_name
        name = 'client_name'
        client_name = meta_params.get(name)

        # number_of_batches
        name = 'number_of_batches'
        number_of_batches = meta_params.get(name)

        # run_in_parallel
        name = 'run_in_parallel'
        if meta_params.get(name) is None:
            # set default value
            run_in_parallel = True
        else:
            run_in_parallel = meta_params.get(name)

        # raw_data_dir
        name = 'raw_data_dir'
        raw_data_dir = meta_params.get(name)

        # design_matrix_dir
        name = 'design_matrix_dir'
        design_matrix_dir = meta_params.get(name)

        self._meta_params = MetaParams(client_name, number_of_batches, run_in_parallel,
                                       raw_data_dir, design_matrix_dir)

        # TimeSeriesFeaturesEnricher
        section_name = 'TimeSeriesFeaturesEnricher'
        if fe_object_to_params.get(section_name) is None:
            self._time_series_features_enricher = {}
        else:
            self._time_series_features_enricher = fe_object_to_params[section_name]

        # FeatureExtractors
        section_name = 'FeatureExtractors'
        self._feature_extractors = fe_object_to_params[section_name]