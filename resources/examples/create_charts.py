import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from src.utils.logger import log

# pd.set_option('display.max_rows', None, 'display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


def _timestamp2duration_in_minutes_forward_view(timestamp):
    assert isinstance(timestamp, pd.Series)

    duration_in_minutes = [(timestamp - timestamp.iloc[0]).iloc[i].total_seconds() / 60 for i in
                           range(len(timestamp))]

    duration_in_minutes = pd.Series(duration_in_minutes)
    return duration_in_minutes


def _timestamp2duration_in_minutes_backward_view(timestamp):
    assert isinstance(timestamp, pd.Series)

    duration_in_minutes = [(timestamp - timestamp.iloc[-1]).iloc[i].total_seconds() / 60 for i in
                           range(len(timestamp))]

    duration_in_minutes = pd.Series(duration_in_minutes)
    return duration_in_minutes


def _prepare_data_for_chart(data, batch_id, sensor_id):
    # batch data
    batch_data = data.loc[(data['batch_id'] == batch_id) & (data['sensor_id'] == sensor_id)].copy()
    batch_timestamps = batch_data['timestamp'].copy()
    batch_timestamps = batch_timestamps.reset_index(drop=True)
    batch_duration_in_minutes_forward_view = _timestamp2duration_in_minutes_forward_view(batch_timestamps)
    batch_duration_in_minutes_backward_view = _timestamp2duration_in_minutes_backward_view(batch_timestamps)
    batch_values = batch_data['value'].copy()
    batch_values = batch_values.reset_index(drop=True)

    # normal batches data
    normal_batches_data = data.loc[
        (data['batch_id'] != batch_id) & (data['sensor_id'] == sensor_id) & (data['batch_label'] == 0)].copy()
    assert not normal_batches_data.empty, 'There are no normal batches for the sensor id: {}.'.format(sensor_id)

    normal_batches_data['duration_in_minutes_forward_view'] = normal_batches_data.groupby('batch_id')[
        'timestamp'].transform(_timestamp2duration_in_minutes_forward_view)
    normal_batches_data['duration_in_minutes_backward_view'] = normal_batches_data.groupby('batch_id')[
        'timestamp'].transform(_timestamp2duration_in_minutes_backward_view)

    normal_batches_average_values_forward_view = normal_batches_data.groupby('duration_in_minutes_forward_view')[
        'value'].aggregate(
        [np.mean, np.std, len]).reset_index()

    normal_batches_duration_in_minutes_forward_view = normal_batches_average_values_forward_view[
        'duration_in_minutes_forward_view']
    normal_batch_averages_forward_view = normal_batches_average_values_forward_view['mean']
    # TODO: need to revise weights
    # width depending on number of batches participating in the average calculation
    normal_batch_lengths = normal_batches_average_values_forward_view['len']
    weights = normal_batch_lengths / normal_batch_lengths.max()
    normal_batch_lower_values_forward_view = normal_batch_averages_forward_view - weights
    normal_batch_upper_values_forward_view = normal_batch_averages_forward_view + weights
    # width depending of +- 3 std
    normal_batch_stds = normal_batches_average_values_forward_view['std']
    normal_batch_lower_values_forward_view = normal_batch_averages_forward_view - 3 * (
            normal_batch_stds / np.sqrt(normal_batch_lengths))
    normal_batch_upper_values_forward_view = normal_batch_averages_forward_view + 3 * (
            normal_batch_stds / np.sqrt(normal_batch_lengths))

    normal_batches_average_values_backward_view = normal_batches_data.groupby('duration_in_minutes_backward_view')[
        'value'].aggregate([np.mean, np.std, len]).reset_index()

    normal_batches_duration_in_minutes_backward_view = normal_batches_average_values_backward_view[
        'duration_in_minutes_backward_view']
    normal_batch_averages_backward_view = normal_batches_average_values_backward_view['mean']
    # TODO: need to revise weights
    # width depending on number of batches participating in the average calculation
    normal_batch_lengths = normal_batches_average_values_backward_view['len']
    weights = normal_batch_lengths / normal_batch_lengths.max()
    normal_batch_lower_values_backward_view = normal_batch_averages_backward_view - weights
    normal_batch_lower_upper_backward_view = normal_batch_averages_backward_view + weights
    # width depending of +- 3 std
    normal_batch_stds = normal_batches_average_values_backward_view['std']
    normal_batch_lower_values_backward_view = normal_batch_averages_backward_view - 3 * (
            normal_batch_stds / np.sqrt(normal_batch_lengths))
    normal_batch_upper_values_backward_view = normal_batch_averages_backward_view + 3 * (
            normal_batch_stds / np.sqrt(normal_batch_lengths))

    data_for_chart = dict()
    data_for_chart['batch_duration_in_minutes_forward_view'] = batch_duration_in_minutes_forward_view
    data_for_chart['batch_duration_in_minutes_backward_view'] = batch_duration_in_minutes_backward_view
    data_for_chart['batch_values'] = batch_values
    data_for_chart['normal_batches_duration_in_minutes_forward_view'] = normal_batches_duration_in_minutes_forward_view
    data_for_chart[
        'normal_batches_duration_in_minutes_backward_view'] = normal_batches_duration_in_minutes_backward_view
    data_for_chart['normal_batch_averages_forward_view'] = normal_batch_averages_forward_view
    data_for_chart['normal_batch_lower_values_forward_view'] = normal_batch_lower_values_forward_view
    data_for_chart['normal_batch_upper_values_forward_view'] = normal_batch_upper_values_forward_view

    data_for_chart['normal_batch_averages_backward_view'] = normal_batch_averages_backward_view
    data_for_chart['normal_batch_lower_values_backward_view'] = normal_batch_lower_values_backward_view
    data_for_chart['normal_batch_upper_values_backward_view'] = normal_batch_upper_values_backward_view

    return data_for_chart


def create_chart(data, batch_id, sensor_id, dir=None):
    log.debug('Creating prospect/retrospect charts (Forward/Backward View) for batch {} and sensor {}.'.format(
        batch_id, sensor_id))

    assert isinstance(data, pd.DataFrame)
    assert not pd.isnull(data).any().any(), 'Data have missing values. Please check.'
    assert isinstance(batch_id, str)
    assert isinstance(sensor_id, str)
    expected_columns = ['batch_id', 'sensor_id', 'timestamp', 'value', 'batch_label']
    assert set(data.columns) == set(expected_columns)
    s = 'batch {} has no records for sensor {}'.format(batch_id, sensor_id)
    assert sensor_id in data.loc[data['batch_id'] == batch_id, 'sensor_id'].unique(), s

    data_for_chart = _prepare_data_for_chart(data, batch_id, sensor_id)

    # forward view
    batch_duration_in_minutes_forward_view = data_for_chart.get('batch_duration_in_minutes_forward_view')
    batch_values = data_for_chart.get('batch_values')
    normal_batches_duration_in_minutes_forward_view = data_for_chart.get(
        'normal_batches_duration_in_minutes_forward_view')
    normal_batch_averages_forward_view = data_for_chart.get('normal_batch_averages_forward_view')
    normal_batch_lower_values_forward_view = data_for_chart.get('normal_batch_lower_values_forward_view')
    normal_batch_upper_values_forward_view = data_for_chart.get('normal_batch_upper_values_forward_view')

    # backward view
    batch_duration_in_minutes_backward_view = data_for_chart.get('batch_duration_in_minutes_backward_view')
    batch_values = data_for_chart.get('batch_values')
    normal_batches_duration_in_minutes_backward_view = data_for_chart.get(
        'normal_batches_duration_in_minutes_backward_view')
    normal_batch_averages_backward_view = data_for_chart.get('normal_batch_averages_backward_view')
    normal_batch_lower_values_backward_view = data_for_chart.get('normal_batch_lower_values_backward_view')
    normal_batch_upper_values_backward_view = data_for_chart.get('normal_batch_upper_values_backward_view')

    fig, ax = plt.subplots(2, 1)

    ax[0].plot(batch_duration_in_minutes_forward_view, batch_values,
               marker='', color='red', label='Batch id: {}'.format(batch_id))
    ax[0].plot(normal_batches_duration_in_minutes_forward_view, normal_batch_averages_forward_view,
               marker='', color='green', linewidth=3, label='Normal Batches (avg.)')
    ax[0].fill_between(normal_batches_duration_in_minutes_forward_view, normal_batch_lower_values_forward_view,
                       normal_batch_upper_values_forward_view,
                       color='lightgreen', alpha='0.2')
    # ax[0].title('Prospect (Forward) View: Sensor id {}'.format(sensor_id), fontsize=12)
    ax[0].set_title('Prospect (Forward) View', size=12)
    # ax[0].xlabel('Minutes (since start)')
    ax[0].set_xlabel('Minutes (since start)')
    ax[0].legend()

    ax[1].plot(batch_duration_in_minutes_backward_view, batch_values,
               marker='', color='red', label='Batch id: {}'.format(batch_id))
    ax[1].plot(normal_batches_duration_in_minutes_backward_view, normal_batch_averages_backward_view,
               marker='', color='green', linewidth=3, label='Normal Batches (avg.)')
    ax[1].fill_between(normal_batches_duration_in_minutes_backward_view, normal_batch_lower_values_backward_view,
                       normal_batch_upper_values_backward_view,
                       color='lightgreen', alpha='0.2')
    # ax[1].title('Retrospect (Backward) View: Sensor id {}'.format(sensor_id), fontsize=12)
    ax[1].set_title('Retrospect (Backward) View', size=12)
    # ax[1].xlabel('Minutes (prior to end)')
    ax[1].set_xlabel('Minutes (prior to end)')
    ax[1].legend()

    fig.suptitle('Anomaly Chart for batch id: {} and sensor id: {}'.format(batch_id, sensor_id), size=15)
    #fig.show()

    if dir is not None:
        file_name = 'anomaly_chart_' + 'batch_id_' + batch_id + 'sensor_id_' + sensor_id + '.pdf'
        full_path = dir + file_name

        fig.set_size_inches(10, 10)
        fig.savefig(full_path, dpi=100)

    log.debug('Done creating prospect/retrospect charts (Forward/Backward View) for batch {} and sensor {}.'.format(
        batch_id, sensor_id))


if __name__ == '__main__':
    fullpath = '/Users/yuval/Desktop/mock_data.csv'
    data = pd.read_csv(fullpath, parse_dates=['timestamp'], infer_datetime_format=True)
    # TODO: add sorting by batch_id, sensor_id, timestamp

    abnormal_batch_ids = data.loc[data['batch_label'] == 1, 'batch_id'].unique()
    assert len(abnormal_batch_ids) > 0, 'There is no abnormal batch in the data.'
    batch_id = random.choice(abnormal_batch_ids)
    sensor_id = random.choice(data['sensor_id'].unique())

    #create_chart(data, batch_id, sensor_id, dir=None)
    create_chart(data, batch_id, sensor_id, dir='/Users/yuval/Desktop/')
