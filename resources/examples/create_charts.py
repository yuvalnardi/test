import pandas as pd
import random
import matplotlib.pyplot as plt

from src.utils.logger import log

# pd.set_option('display.max_rows', None, 'display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


def create_prospect_chart(data, batch_id, sensor_id):
    log.debug('Creating prospect charts (Forward View) for batch {} and sensor {}.'.format(
        batch_id, sensor_id))

    assert isinstance(data, pd.DataFrame)
    expected_columns = ['batch_id', 'sensor_id', 'timestamp', 'value', 'batch_label']
    assert set(data.columns) == set(expected_columns)
    s = 'batch {} has no records for sensor {}'.format(batch_id, sensor_id)
    assert sensor_id in data.loc[data['batch_id'] == batch_id, 'sensor_id'].unique(), s

    # TODO:
    x = data.loc[(data['batch_id'] == batch_id) & (data['sensor_id'] == sensor_id), 'timestamp']
    minutes_since_start = [(x - x.iloc[0]).iloc[i].total_seconds()/60 for i in range(len(x))]
    y = data.loc[(data['batch_id'] == batch_id) & (data['sensor_id'] == sensor_id), 'value']
    y_average_normal = y + 2

    lines = plt.plot(minutes_since_start, y, 'k-', minutes_since_start, y_average_normal, 'b--')
    plt.show()

    log.debug('Done creating prospect charts (Forward View) for batch {} and sensor {}.'.format(
        batch_id, sensor_id))


def create_retrospect_chart(data, batch_id, sensor_id):
    log.debug('Creating retrospect charts (Backward View) for batch {} and sensor {}.'.format(
        batch_id, sensor_id))

    assert isinstance(data, pd.DataFrame)
    expected_columns = ['batch_id', 'sensor_id', 'timestamp', 'value', 'batch_label']
    assert set(data.columns) == set(expected_columns)
    s = 'batch {} has no records for sensor {}'.format(batch_id, sensor_id)
    assert sensor_id in data.loc[data['batch_id'] == batch_id, 'sensor_id'].unique(), s

    # TODO:

    log.debug('Done creating retrospect charts (Forward View) for batch {} and sensor {}.'.format(
        batch_id, sensor_id))


if __name__ == '__main__':
    fullpath = '/Users/yuval/Desktop/mock_data.csv'
    data = pd.read_csv(fullpath, parse_dates=['timestamp'], infer_datetime_format=True)

    batch_id = random.choice(data['batch_id'].unique())
    sensor_id = random.choice(data['sensor_id'].unique())

    create_prospect_chart(data, batch_id, sensor_id)
    #create_retrospect_chart(data, batch_id, sensor_id)
