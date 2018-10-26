import pandas as pd
import random
import matplotlib.pyplot as plt

from src.utils.logger import log

# pd.set_option('display.max_rows', None, 'display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


def _prepare_data_for_chart(data, batch_id, sensor_id):
    # batch data
    batch_data = data.loc[(data['batch_id'] == batch_id) & (data['sensor_id'] == sensor_id)].copy()
    batch_timestamps = batch_data['timestamp'].copy()
    batch_duration_in_minutes = [(batch_timestamps - batch_timestamps.iloc[0]).iloc[i].total_seconds() / 60
                                 for i in range(len(batch_timestamps))]

    # normal batches data

    batch_duration_in_minutes = pd.Series({})
    normal_batches_duration_in_minutes = pd.Series({})
    batch_values = pd.Series({})
    normal_batch_lower_values = pd.Series({})
    normal_batch_upper_values = pd.Series({})

    data_in_chart_format = dict()
    data_in_chart_format['batch_duration_in_minutes'] = batch_duration_in_minutes
    data_in_chart_format['normal_batches_duration_in_minutes'] = normal_batches_duration_in_minutes
    data_in_chart_format['batch_values'] = batch_values
    data_in_chart_format['normal_batch_lower_values'] = normal_batch_lower_values
    data_in_chart_format['normal_batch_upper_values'] = normal_batch_upper_values

    return data_in_chart_format


def create_prospect_chart(data, batch_id, sensor_id):
    log.debug('Creating prospect charts (Forward View) for batch {} and sensor {}.'.format(
        batch_id, sensor_id))

    assert isinstance(data, pd.DataFrame)
    assert isinstance(batch_id, str)
    assert isinstance(sensor_id, str)
    expected_columns = ['batch_id', 'sensor_id', 'timestamp', 'value', 'batch_label']
    assert set(data.columns) == set(expected_columns)
    s = 'batch {} has no records for sensor {}'.format(batch_id, sensor_id)
    assert sensor_id in data.loc[data['batch_id'] == batch_id, 'sensor_id'].unique(), s

    # TODO: FINISH _prepare_data_for_chart
    data_in_chart_format = _prepare_data_for_chart(data, batch_id, sensor_id)
    batch_duration_in_minutes = data_in_chart_format.get('batch_duration_in_minutes')
    normal_batches_duration_in_minutes = data_in_chart_format.get('normal_batches_duration_in_minutes')
    batch_values = data_in_chart_format.get('batch_values')
    normal_batch_lower_values = data_in_chart_format.get('normal_batch_lower_values')
    normal_batch_upper_values = data_in_chart_format.get('normal_batch_upper_values')

    x = data.loc[(data['batch_id'] == batch_id) & (data['sensor_id'] == sensor_id), 'timestamp']
    batch_duration_in_minutes = [(x - x.iloc[0]).iloc[i].total_seconds() / 60 for i in range(len(x))]
    normal_batches_duration_in_minutes = batch_duration_in_minutes
    y = data.loc[(data['batch_id'] == batch_id) & (data['sensor_id'] == sensor_id), 'value']
    y_average_normal = y + 2
    normal_batch_lower_values = y_average_normal - 0.4
    normal_batch_upper_values = y_average_normal + 0.4

    plt.plot(batch_duration_in_minutes, y, marker='', color='red', label='Batch id: {}'.format(batch_id))
    # plt.plot(minutes_since_start, y_average_normal, marker='', color='lightgreen', linewidth=10, label='Normal Batches')
    plt.fill_between(normal_batches_duration_in_minutes, normal_batch_lower_values, normal_batch_upper_values,
                     color='lightgreen', alpha='0.5')
    plt.title('Prospect (Forward) View: Sensor id {}'.format(sensor_id), fontsize=15)
    plt.xlabel('Minutes (since start)')
    plt.legend()
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

    abnormal_batch_ids = data.loc[data['batch_label'] == 1, 'batch_id'].unique()
    assert len(abnormal_batch_ids) > 0, 'There is no abnormal batch in the data.'
    batch_id = random.choice(abnormal_batch_ids)
    sensor_id = random.choice(data['sensor_id'].unique())

    create_prospect_chart(data, batch_id, sensor_id)
    # create_retrospect_chart(data, batch_id, sensor_id)
