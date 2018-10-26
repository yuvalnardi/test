import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from src.utils.logger import log

# pd.set_option('display.max_rows', None, 'display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


def _timestamp2duration_in_minutes(timestamp):
    assert isinstance(timestamp, pd.Series)

    duration_in_minutes = [(timestamp - timestamp.iloc[0]).iloc[i].total_seconds() / 60 for i in
                           range(len(timestamp))]

    duration_in_minutes = pd.Series(duration_in_minutes)
    return duration_in_minutes


def _prepare_data_for_chart(data, batch_id, sensor_id):
    # batch data
    batch_data = data.loc[(data['batch_id'] == batch_id) & (data['sensor_id'] == sensor_id)].copy()
    batch_timestamps = batch_data['timestamp'].copy()
    batch_timestamps = batch_timestamps.reset_index(drop=True)
    batch_duration_in_minutes = _timestamp2duration_in_minutes(batch_timestamps)
    batch_values = batch_data['value'].copy()
    batch_values = batch_values.reset_index(drop=True)

    # normal batches data
    normal_batches_data = data.loc[
        (data['batch_id'] != batch_id) & (data['sensor_id'] == sensor_id) & (data['batch_label'] == 0)].copy()
    assert not normal_batches_data.empty, 'There are no normal batches for the sensor id: {}.'.format(sensor_id)
    normal_batches_data['duration_in_minutes'] = normal_batches_data.groupby('batch_id')['timestamp'].transform(
        _timestamp2duration_in_minutes)
    normal_batches_average_values = normal_batches_data.groupby('duration_in_minutes')['value'].aggregate(
        [np.mean, np.std, len]).reset_index()

    normal_batches_duration_in_minutes = normal_batches_average_values['duration_in_minutes']
    normal_batch_averages = normal_batches_average_values['mean']
    # TODO: need to revise weights
    # width depending on number of batches participating in the average calculation
    normal_batch_lengths = normal_batches_average_values['len']
    weights = normal_batch_lengths / normal_batch_lengths.max()
    normal_batch_lower_values = normal_batch_averages - weights
    normal_batch_upper_values = normal_batch_averages + weights
    # width depending of +- 3 std
    normal_batch_stds = normal_batches_average_values['std']
    normal_batch_lower_values = normal_batch_averages - 3 * (normal_batch_stds / np.sqrt(normal_batch_lengths))
    normal_batch_upper_values = normal_batch_averages + 3 * (normal_batch_stds / np.sqrt(normal_batch_lengths))

    data_for_chart = dict()
    data_for_chart['batch_duration_in_minutes'] = batch_duration_in_minutes
    data_for_chart['batch_values'] = batch_values
    data_for_chart['normal_batches_duration_in_minutes'] = normal_batches_duration_in_minutes
    data_for_chart['normal_batch_averages'] = normal_batch_averages
    data_for_chart['normal_batch_lower_values'] = normal_batch_lower_values
    data_for_chart['normal_batch_upper_values'] = normal_batch_upper_values

    return data_for_chart


def create_prospect_chart(data, batch_id, sensor_id):
    log.debug('Creating prospect charts (Forward View) for batch {} and sensor {}.'.format(
        batch_id, sensor_id))

    assert isinstance(data, pd.DataFrame)
    assert not pd.isnull(data).any().any(), 'Data have missing values. Please check.'
    assert isinstance(batch_id, str)
    assert isinstance(sensor_id, str)
    expected_columns = ['batch_id', 'sensor_id', 'timestamp', 'value', 'batch_label']
    assert set(data.columns) == set(expected_columns)
    s = 'batch {} has no records for sensor {}'.format(batch_id, sensor_id)
    assert sensor_id in data.loc[data['batch_id'] == batch_id, 'sensor_id'].unique(), s

    # TODO: FINISH _prepare_data_for_chart
    data_for_chart = _prepare_data_for_chart(data, batch_id, sensor_id)
    batch_duration_in_minutes = data_for_chart.get('batch_duration_in_minutes')
    batch_values = data_for_chart.get('batch_values')
    normal_batches_duration_in_minutes = data_for_chart.get('normal_batches_duration_in_minutes')
    normal_batch_averages = data_for_chart.get('normal_batch_averages')
    normal_batch_lower_values = data_for_chart.get('normal_batch_lower_values')
    normal_batch_upper_values = data_for_chart.get('normal_batch_upper_values')

    plt.plot(batch_duration_in_minutes, batch_values,
             marker='', color='red', label='Batch id: {}'.format(batch_id))
    plt.plot(normal_batches_duration_in_minutes, normal_batch_averages,
             marker='', color='green', linewidth=3, label='Normal Batches (avg.)')
    plt.fill_between(normal_batches_duration_in_minutes, normal_batch_lower_values, normal_batch_upper_values,
                     color='lightgreen', alpha='0.2')
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
    # TODO: add sorting by batch_id, sensor_id, timestamp

    abnormal_batch_ids = data.loc[data['batch_label'] == 1, 'batch_id'].unique()
    assert len(abnormal_batch_ids) > 0, 'There is no abnormal batch in the data.'
    batch_id = random.choice(abnormal_batch_ids)
    sensor_id = random.choice(data['sensor_id'].unique())

    create_prospect_chart(data, batch_id, sensor_id)
    # create_retrospect_chart(data, batch_id, sensor_id)
