import yaml

from src.utils.logger import log

class FeatureEngineeringParamsHandler(object):

    def __init__(self):
        pass

    @staticmethod
    def get_params(params_file_path):
        try:
            with open(params_file_path, 'r') as ymlfile:
                params_config = yaml.load(ymlfile)
                return params_config
        except FileNotFoundError as er:
            log.error('File {} does not exist\n'.format(params_file_path) + str(er))
            raise
        except Exception:
            log.error('Unknown general error!')
            raise