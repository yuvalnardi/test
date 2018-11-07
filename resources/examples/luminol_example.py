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


def load_ts_data(path, timestamp_col=None, epoch_col=None, set_index=False, n_rows=None):
    """ path should be a full path to a csv file with exactly one of columns
        timestamp or epoch, and a value column
    """
    assert isinstance(path, str)
    assert isinstance(timestamp_col, (str, type(None)))
    assert isinstance(epoch_col, (str, type(None)))
    assert isinstance(set_index, bool)
    assert isinstance(n_rows, (int, type(None)))

    ts = pd.read_csv(path)  # , parse_dates=['timestamp'], infer_datetime_format=True)
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
        # TODO: make sure this is correct
        ts['epoch'] = ts['timestamp'].head().astype('int64') // 1e9

    if set_index:
        ts = ts.set_index('timestamp')
    if n_rows is not None:
        ts = ts.iloc[0:n_rows]

    return ts


def plot_ts_and_anomalies(ts, anomalies, anomaly_scores, ts_only=False, dir=None, show=True, plotly=False):
    assert isinstance(ts, pd.DataFrame)
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

    if (len(anomalies) == 0) or ts_only:
        # plot ts only
        if len(anomalies) == 0:
            log.debug('Found no anomalies.')
        plt.plot(ts['value'], color='blue')
        plt.title('ts', size=12)

        if show:
            plt.show()

    else:
        # plot ts and anomalies
        log.debug('Found {} anomalies.'.format(len(anomalies)))

        anom_score = []
        scores = []

        for (timestamp, value) in anomaly_scores.iteritems():
            # t_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
            # anom_score.append([t_str, value])
            scores.append(value)

        fig, ax = plt.subplots(2, 1)

        # plot ts
        # ax[0].plot(ts['epoch'], ts['value'], color='blue')
        ax[0].plot_date(ts['timestamp'], ts['value'], color='blue', fmt='-')
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
                    y=ts['value'],
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
                log.debug('Need to supply a dir in order for to generate plotly chart.')


if __name__ == '__main__':
    path = '/Users/yuval/Dropbox/MyData/Misc/Seebo/data/g.csv'
    ts = load_ts_data(path, epoch_col='timestamp', n_rows=10000)

    # path = '/Users/yuval/Dropbox/MyData/Misc/Seebo/data/art_noisy.csv'
    # ts = load_ts_data(path, timestamp_col='timestamp', n_rows=1000)

    # path = '/Users/yuval/Dropbox/MyData/Misc/Seebo/data/cpu4.csv'
    # ts = load_ts_data(path, epoch_col='timestamp', n_rows=30000)

    # path = '/Users/yuval/Dropbox/MyData/Misc/Seebo/data/network.csv'
    # ts = load_ts_data(path, timestamp_col='time', n_rows=10000)
    # ts.rename(columns={'In Octets': 'value'}, inplace=True)

    keys = ts['epoch']
    values = ts['value']
    ts_dict = dict(zip(keys, values))

    algorithm_name = 'exp_avg_detector'

    anomaly_detector = AnomalyDetector(ts_dict, algorithm_name=algorithm_name)
    anomalies = anomaly_detector.get_anomalies()
    anomaly_scores = anomaly_detector.get_all_scores()

    plot_ts_and_anomalies(ts, anomalies, anomaly_scores, ts_only=False, dir='/Users/yuval/Desktop/',
                          show=True, plotly=True)
