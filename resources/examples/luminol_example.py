import pandas as pd
import os
import matplotlib.pyplot as plt
from plotly import tools
from plotly.offline import plot
import plotly.graph_objs as go
from luminol.anomaly_detector import AnomalyDetector
from luminol.modules.time_series import TimeSeries

from src.utils.logger import log

# pd.set_option('display.max_rows', None, 'display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


def load_ts_data(path, timestamp_col=None, date_cols=None, epoch_col=None, set_index=False, n_rows=None):
    """ path should be a full path to a csv file with exactly one of columns
        timestamp or epoch
    """
    assert isinstance(path, str)
    assert isinstance(timestamp_col, (str, type(None)))
    assert isinstance(date_cols, (list, type(None))) # columns to be parsed as dates
    if date_cols is None:
        date_cols = False
    if timestamp_col is not None:
        assert timestamp_col in date_cols, 'timestamp_col should be one of date_cols.'
    assert isinstance(epoch_col, (str, type(None)))
    assert isinstance(set_index, bool)
    assert isinstance(n_rows, (int, type(None)))

    if timestamp_col is not None:
        ts = pd.read_csv(path, parse_dates=date_cols, infer_datetime_format=True)
    else:
        ts = pd.read_csv(path)
    assert isinstance(ts, pd.DataFrame)
    assert ((timestamp_col is None) and (epoch_col is not None)) or \
           ((timestamp_col is not None) and (
                   epoch_col is None)), 'ts should have either \'timestamp\' or \'epoch\' columns (but not both).'

    log.debug('ts shape: {}.'.format(ts.shape))
    log.debug('ts head: {}'.format(ts.head()))

    if timestamp_col is None:
        # ts has epoch column. rename and add timestamp column
        ts.rename(columns={epoch_col: 'epoch'}, inplace=True)
        # convert to timestamp
        ts['timestamp'] = pd.to_datetime(ts['epoch'], unit='s')

    if epoch_col is None:
        # ts has timestamp column. rename and add epoch column
        ts.rename(columns={timestamp_col: 'timestamp'}, inplace=True)
        # convert to epoch
        ts['epoch'] = ts['timestamp'].astype('int64') // 1e9
        ts['epoch'] = ts['epoch'].astype('int64')

    if set_index:
        ts = ts.set_index('timestamp')
    if n_rows is not None:
        ts = ts.iloc[0:n_rows]

    return ts


def plot_ts_and_anomalies(ts, value_col, anomalies, anomaly_scores, ts_only=False, dir=None, show=True, plotly=False):
    assert isinstance(ts, pd.DataFrame)
    assert isinstance(value_col, str)
    assert not pd.isnull(ts[value_col]).any(), 'value_col has missing data'
    assert isinstance(anomalies, list)
    assert isinstance(anomaly_scores, TimeSeries)
    assert isinstance(ts_only, bool)
    assert isinstance(dir, (str, type(None)))
    if dir is not None:
        assert dir[-1] == '/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    assert isinstance(show, bool)
    assert isinstance(plotly, bool)

    assert len(ts['timestamp'].unique()) == ts.shape[0], 'timestamp should not have duplicated values'

    if (len(anomalies) == 0) or ts_only:
        # plot ts only
        if len(anomalies) == 0:
            log.debug('Found no anomalies.')
        plt.plot_date(ts['timestamp'], ts[value_col], color='blue', fmt='-')
        plt.title('ts', size=12)

        if show:
            plt.show()

        # TODO: add plotly plot in this case as well

    else:
        # plot ts and anomalies
        log.debug('Found {} anomalies.'.format(len(anomalies)))

        scores = anomaly_scores.values

        fig, ax = plt.subplots(2, 1)

        # plot ts
        # ax[0].plot(ts['epoch'], ts[value_col], color='blue')
        ax[0].plot_date(ts['timestamp'], ts[value_col], color='blue', fmt='-')
        ax[0].set_title('ts', size=12)

        # plot anomalies on top of ts
        for anomaly in anomalies:
            anomay_time_window = anomaly.get_time_window()
            epoch_left = anomay_time_window[0]
            epoch_right = anomay_time_window[1]
            timestamp_left = ts.loc[ts['epoch'] == epoch_left, 'timestamp'].values[0]
            timestamp_right = ts.loc[ts['epoch'] == epoch_right, 'timestamp'].values[0]
            ax[0].axvspan(timestamp_left, timestamp_right, alpha=0.5, color='gray')

        # plot anomaly scores
        # ax[1].plot(ts['epoch'], scores, color='red')
        ax[1].plot_date(ts['timestamp'], scores, color='red', fmt='-')
        ax[1].set_title('scores', size=12)

        if show:
            fig.show()

        if dir is not None:
            file_name = 'ts_and_anomaly_scores.pdf'
            full_path = dir + file_name

            fig.set_size_inches(10, 10)
            fig.savefig(full_path, dpi=100)

        if plotly:
            if dir is not None:
                file_name = 'ts_and_anomaly_scores.html'
                full_path = dir + file_name
                time_series = go.Scatter(
                    x=ts['timestamp'],
                    y=ts[value_col],
                    name='ts',
                    mode='lines',
                    line=dict(color='blue'))
                anomaly_scores = go.Scatter(
                    x=ts['timestamp'],
                    y=scores,
                    name='scores',
                    line=dict(color='red'))

                fig = tools.make_subplots(rows=2, cols=1, specs=[[{}], [{}]],
                                          shared_xaxes=True, shared_yaxes=False)

                fig.append_trace(time_series, 1, 1)
                fig.append_trace(anomaly_scores, 2, 1)

                # fig['layout'].update(height=600, width=800, title='Time series and anomaly scores')
                fig['layout'].update(title='Time series and anomaly scores')
                plot(fig, filename=full_path)
            else:
                log.debug('Need to supply a dir in order to generate plotly chart.')


if __name__ == '__main__':

    example = 6
    if example == 1:
        # example 1 - has epoch_col
        path = '/Users/yuval/Dropbox/MyData/Misc/Seebo/data/g.csv'
        ts = load_ts_data(path, epoch_col='timestamp', n_rows=10000)
        value_col = 'value'
    elif example == 2:
        # example 2 - has timestamp_col
        path = '/Users/yuval/Dropbox/MyData/Misc/Seebo/data/art_noisy.csv'
        ts = load_ts_data(path, timestamp_col='timestamp', n_rows=1000)
        value_col = 'value'
    elif example == 3:
        # example 3
        path = '/Users/yuval/Dropbox/MyData/Misc/Seebo/data/cpu4.csv'
        ts = load_ts_data(path, epoch_col='timestamp', n_rows=30000)
        value_col = 'value'
    elif example == 4:
        # example 4 - has many columns. Does not run. Has duplicated timestamp values
        path = '/Users/yuval/Dropbox/MyData/Misc/Seebo/data/network.csv'
        ts = load_ts_data(path, timestamp_col='time')
        value_col = 'In Octets'
    elif example == 5:
        # example 5
        path = '/Users/yuval/Dropbox/MyData/Misc/Seebo/data/sample-1H.csv'
        ts = load_ts_data(path, timestamp_col='date')
        value_col = 'value'
        ts = ts.loc[ts['category'] == 'C'] # 'A', 'B', or 'C'
    elif example == 6:
        # example 6 - strauss, one batch_id
        path = '/Users/yuval/Desktop/Yuval_TS_Table.csv'
        date_cols = ['end_time_stamp', 'start_time', 'end_time']
        ts = load_ts_data(path, timestamp_col='end_time_stamp', date_cols=date_cols)
        ts = ts.loc[ts['stage_parallel'] == 'Sterilization #111'] # 'Puding Mixing #1', 'Sterilization #111', 'Storage tank #1'
        value_col = 'sensor_value'
    else:
        raise Exception('Unknown example.')

    # run anomaly detection algorithm
    keys = ts['epoch']
    values = ts[value_col]
    ts_dict = dict(zip(keys, values))

    algorithm_name = 'exp_avg_detector'

    anomaly_detector = AnomalyDetector(ts_dict, algorithm_name=algorithm_name)
    anomalies = anomaly_detector.get_anomalies()
    anomaly_scores = anomaly_detector.get_all_scores()

    # plot results
    plot_ts_and_anomalies(ts, value_col, anomalies, anomaly_scores,
                          ts_only=False, dir='/Users/yuval/Desktop/', show=True, plotly=True)
