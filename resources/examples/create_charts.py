import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from plotly.offline import plot
import plotly.graph_objs as go

from src.utils.logger import log

# pd.set_option('display.max_rows', None, 'display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


def _timestamp2duration_in_minutes_forward_view(timestamps):
    assert isinstance(timestamps, pd.Series)

    duration_in_minutes = [(timestamps - timestamps.iloc[0]).iloc[i].total_seconds() / 60 for i in
                           range(len(timestamps))]

    duration_in_minutes = pd.Series(duration_in_minutes)
    return duration_in_minutes


def _timestamp2duration_in_minutes_backward_view(timestamps):
    assert isinstance(timestamps, pd.Series)

    duration_in_minutes = [(timestamps - timestamps.iloc[-1]).iloc[i].total_seconds() / 60 for i in
                           range(len(timestamps))]

    duration_in_minutes = pd.Series(duration_in_minutes)
    return duration_in_minutes


def _prepare_data_for_chart(data, anomalous_batch_id, sensor_id):
    # batch data
    batch_data = data.loc[(data['batch_id'] == anomalous_batch_id) & (data['sensor_id'] == sensor_id)].copy()
    batch_timestamps = batch_data['timestamp'].copy()
    batch_timestamps = batch_timestamps.reset_index(drop=True)
    batch_duration_in_minutes_forward_view = _timestamp2duration_in_minutes_forward_view(batch_timestamps)
    batch_duration_in_minutes_backward_view = _timestamp2duration_in_minutes_backward_view(batch_timestamps)
    batch_values = batch_data['value'].copy()
    batch_values = batch_values.reset_index(drop=True)

    # normal batches data
    normal_batches_data = data.loc[(data['batch_label'] == 0) & (data['sensor_id'] == sensor_id)].copy()
    assert not normal_batches_data.empty, 'There are no normal batches for the sensor id: {}.'.format(sensor_id)

    normal_batches_data['duration_in_minutes_forward_view'] = normal_batches_data.groupby('batch_id')[
        'timestamp'].transform(_timestamp2duration_in_minutes_forward_view)
    normal_batches_data['duration_in_minutes_backward_view'] = normal_batches_data.groupby('batch_id')[
        'timestamp'].transform(_timestamp2duration_in_minutes_backward_view)

    normal_batches_average_values_forward_view = normal_batches_data.groupby('duration_in_minutes_forward_view')[
        'value'].aggregate(
        [np.mean, np.std, len]).reset_index()

    # normal batches - forward view
    normal_batches_duration_in_minutes_forward_view = normal_batches_average_values_forward_view[
        'duration_in_minutes_forward_view']
    normal_batches_averages_forward_view = normal_batches_average_values_forward_view['mean']
    # TODO: need to revise weights
    # width depending on number of batches participating in the average calculation
    normal_batches_lengths = normal_batches_average_values_forward_view['len']
    weights = normal_batches_lengths / normal_batches_lengths.max()
    normal_batches_lower_values_forward_view = normal_batches_averages_forward_view - weights
    normal_batches_upper_values_forward_view = normal_batches_averages_forward_view + weights
    # width depending of +- 3 std
    normal_batches_stds = normal_batches_average_values_forward_view['std']
    normal_batches_lower_values_forward_view = normal_batches_averages_forward_view - 3 * (
            normal_batches_stds / np.sqrt(normal_batches_lengths))
    normal_batches_upper_values_forward_view = normal_batches_averages_forward_view + 3 * (
            normal_batches_stds / np.sqrt(normal_batches_lengths))

    # normal batches - backward view
    normal_batches_average_values_backward_view = normal_batches_data.groupby('duration_in_minutes_backward_view')[
        'value'].aggregate([np.mean, np.std, len]).reset_index()

    normal_batches_duration_in_minutes_backward_view = normal_batches_average_values_backward_view[
        'duration_in_minutes_backward_view']
    normal_batches_averages_backward_view = normal_batches_average_values_backward_view['mean']
    # TODO: need to revise weights
    # width depending on number of batches participating in the average calculation
    normal_batches_lengths = normal_batches_average_values_backward_view['len']
    weights = normal_batches_lengths / normal_batches_lengths.max()
    normal_batches_lower_values_backward_view = normal_batches_averages_backward_view - weights
    normal_batches_lower_upper_backward_view = normal_batches_averages_backward_view + weights
    # width depending of +- 3 std
    normal_batches_stds = normal_batches_average_values_backward_view['std']
    normal_batches_lower_values_backward_view = normal_batches_averages_backward_view - 3 * (
            normal_batches_stds / np.sqrt(normal_batches_lengths))
    normal_batches_upper_values_backward_view = normal_batches_averages_backward_view + 3 * (
            normal_batches_stds / np.sqrt(normal_batches_lengths))

    data_for_chart = dict()
    data_for_chart['batch_duration_in_minutes_forward_view'] = batch_duration_in_minutes_forward_view
    data_for_chart['batch_duration_in_minutes_backward_view'] = batch_duration_in_minutes_backward_view
    data_for_chart['batch_values'] = batch_values

    data_for_chart['normal_batches_duration_in_minutes_forward_view'] = normal_batches_duration_in_minutes_forward_view
    data_for_chart[
        'normal_batches_duration_in_minutes_backward_view'] = normal_batches_duration_in_minutes_backward_view

    data_for_chart['normal_batches_averages_forward_view'] = normal_batches_averages_forward_view
    data_for_chart['normal_batches_lower_values_forward_view'] = normal_batches_lower_values_forward_view
    data_for_chart['normal_batches_upper_values_forward_view'] = normal_batches_upper_values_forward_view

    data_for_chart['normal_batches_averages_backward_view'] = normal_batches_averages_backward_view
    data_for_chart['normal_batches_lower_values_backward_view'] = normal_batches_lower_values_backward_view
    data_for_chart['normal_batches_upper_values_backward_view'] = normal_batches_upper_values_backward_view

    return data_for_chart


def _validate_and_sort_data_prior_to_charting(data, anomalous_batch_id, sensor_id):
    assert not pd.isnull(data).any().any(), 'Data have missing values. Please check.'
    expected_columns = ['batch_id', 'sensor_id', 'timestamp', 'value', 'batch_label']
    assert set(data.columns) == set(expected_columns)
    s = 'batch {} has no records for sensor {}'.format(anomalous_batch_id, sensor_id)
    assert sensor_id in data.loc[data['batch_id'] == anomalous_batch_id, 'sensor_id'].unique(), s

    assert set(data['batch_label'].unique()) == set(np.array([0, 1]))

    assert (data.loc[data['batch_id'] == anomalous_batch_id,
                     'batch_label'] == 1).all(), 'batch_id should be an abnormal batch.'

    normal_batches = data.loc[data['batch_label'] == 0].copy()
    number_of_normal_batches = len(normal_batches['batch_id'].unique())
    assert not normal_batches.empty, 'There are no normal batches.'
    s = 'At least one normal batch has no records for sensor: {}.'.format(sensor_id)
    assert normal_batches.groupby('batch_id')['sensor_id'].aggregate(
        lambda ts: sensor_id in ts.unique()).sum() == number_of_normal_batches, s

    # sort data by (batch_id, sensor_id, timestamp)
    data = data.sort_values(by=['batch_id', 'sensor_id', 'timestamp'], inplace=False)

    log.debug('Done validating and sorting data by (batch_id, sensor_id, timestamp) prior to charting.')

    return data


def create_anomalous_charts(data, anomalous_batch_id, sensor_id, dir=None, show=True, plotly=False):
    log.debug('Creating prospect/retrospect charts (Forward/Backward View) for batch {} and sensor {}.'.format(
        anomalous_batch_id, sensor_id))

    assert isinstance(data, pd.DataFrame)
    assert isinstance(anomalous_batch_id, str)
    assert isinstance(sensor_id, str)
    assert isinstance(dir, (str, type(None)))
    assert isinstance(show, bool)

    data = _validate_and_sort_data_prior_to_charting(data, anomalous_batch_id, sensor_id)

    data_for_chart = _prepare_data_for_chart(data, anomalous_batch_id, sensor_id)

    batch_values = data_for_chart.get('batch_values')
    # forward view
    batch_duration_in_minutes_forward_view = data_for_chart.get('batch_duration_in_minutes_forward_view')
    normal_batches_duration_in_minutes_forward_view = data_for_chart.get(
        'normal_batches_duration_in_minutes_forward_view')
    normal_batches_averages_forward_view = data_for_chart.get('normal_batches_averages_forward_view')
    normal_batches_lower_values_forward_view = data_for_chart.get('normal_batches_lower_values_forward_view')
    normal_batches_upper_values_forward_view = data_for_chart.get('normal_batches_upper_values_forward_view')

    # backward view
    batch_duration_in_minutes_backward_view = data_for_chart.get('batch_duration_in_minutes_backward_view')
    normal_batches_duration_in_minutes_backward_view = data_for_chart.get(
        'normal_batches_duration_in_minutes_backward_view')
    normal_batches_averages_backward_view = data_for_chart.get('normal_batches_averages_backward_view')
    normal_batches_lower_values_backward_view = data_for_chart.get('normal_batches_lower_values_backward_view')
    normal_batches_upper_values_backward_view = data_for_chart.get('normal_batches_upper_values_backward_view')

    fig, ax = plt.subplots(2, 1)

    ax[0].plot(batch_duration_in_minutes_forward_view, batch_values,
               marker='', color='red', label='Batch id: {}'.format(anomalous_batch_id))
    ax[0].plot(normal_batches_duration_in_minutes_forward_view, normal_batches_averages_forward_view,
               marker='', color='green', linewidth=3, label='Normal Batches (avg.)')
    ax[0].fill_between(normal_batches_duration_in_minutes_forward_view, normal_batches_lower_values_forward_view,
                       normal_batches_upper_values_forward_view,
                       color='lightgreen', alpha='0.2')
    # ax[0].title('Prospect (Forward) View: Sensor id {}'.format(sensor_id), fontsize=12)
    ax[0].set_title('Prospect (Forward) View', size=12)
    # ax[0].xlabel('Minutes (since start)')
    ax[0].set_xlabel('Minutes (since start)')
    ax[0].legend()

    ax[1].plot(batch_duration_in_minutes_backward_view, batch_values,
               marker='', color='red', label='Batch id: {}'.format(anomalous_batch_id))
    ax[1].plot(normal_batches_duration_in_minutes_backward_view, normal_batches_averages_backward_view,
               marker='', color='green', linewidth=3, label='Normal Batches (avg.)')
    ax[1].fill_between(normal_batches_duration_in_minutes_backward_view, normal_batches_lower_values_backward_view,
                       normal_batches_upper_values_backward_view,
                       color='lightgreen', alpha='0.2')
    # ax[1].title('Retrospect (Backward) View: Sensor id {}'.format(sensor_id), fontsize=12)
    ax[1].set_title('Retrospect (Backward) View', size=12)
    # ax[1].xlabel('Minutes (prior to end)')
    ax[1].set_xlabel('Minutes (prior to end)')
    ax[1].legend()

    fig.suptitle('Anomaly Charts for batch id: {} and sensor id: {}'.format(anomalous_batch_id, sensor_id), size=15)

    if show:
        fig.show()

    if dir is not None:
        file_name = 'anomaly_chart_' + 'batch_id_' + anomalous_batch_id + 'sensor_id_' + sensor_id + '.pdf'
        full_path = dir + file_name

        fig.set_size_inches(10, 10)
        fig.savefig(full_path, dpi=100)

    if plotly:
        abnormal_batch = go.Scatter(
            x=batch_duration_in_minutes_forward_view,
            y=batch_values,
            name='Abnormal Batch',
            line=dict(color='red'))
        normal_batches_average = go.Scatter(
            x=normal_batches_duration_in_minutes_forward_view,
            y=normal_batches_averages_forward_view,
            name='Normal Batches',
            line=dict(color='green', width=4))
        normal_batches_lower = go.Scatter(
            x=normal_batches_duration_in_minutes_forward_view,
            y=normal_batches_lower_values_forward_view,
            name='lower',
            hoverinfo='skip',
            fill=None,
            mode='lines',
            line=dict(color='lightgreen'),
            showlegend=False)
        normal_batches_upper = go.Scatter(
            x=normal_batches_duration_in_minutes_forward_view,
            y=normal_batches_upper_values_forward_view,
            name='upper',
            hoverinfo='skip',
            fill='tonexty',
            fillcolor='lightgreen',
            mode='lines',
            opacity=0.005,
            line=dict(color='lightgreen'),
            showlegend=False)

        data = [normal_batches_lower, normal_batches_upper, abnormal_batch, normal_batches_average]
        layout = dict(title='Yuval Nardi')

        fig = dict(data=data, layout=layout)
        plot(fig)

    log.debug('Done creating prospect/retrospect charts (Forward/Backward View) for batch {} and sensor {}.'.format(
        anomalous_batch_id, sensor_id))


if __name__ == '__main__':
    fullpath = '/Users/yuval/Desktop/mock_data.csv'
    data = pd.read_csv(fullpath, parse_dates=['timestamp'], infer_datetime_format=True)

    abnormal_batch_ids = data.loc[data['batch_label'] == 1, 'batch_id'].unique()
    assert len(abnormal_batch_ids) > 0, 'There is no abnormal batch in the data.'
    anomalous_batch_id = random.choice(abnormal_batch_ids)
    sensor_id = random.choice(data['sensor_id'].unique())

    # create_anomalous_charts(data, batch_id, sensor_id, dir=None, show=True)
    # create_anomalous_charts(data, anomalous_batch_id, sensor_id, dir='/Users/yuval/Desktop/', show=True, plotly=True)
    create_anomalous_charts(data, anomalous_batch_id, sensor_id, dir=None, show=False, plotly=True)
