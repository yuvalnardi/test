import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
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
    assert 'value' in ts.columns, 'missing \'value\' column'
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
        ts['epoch'] = range(ts.shape[0])  # TODO

    if set_index:
        ts = ts.set_index('timestamp')
    if n_rows is not None:
        ts = ts.iloc[0:n_rows]

    return ts


def plot_ts_and_anomalies(ts, anomalies, anomaly_scores, ts_only=False, dir=None, show=True):
    assert isinstance(ts, pd.DataFrame)
    assert isinstance(anomalies, list)
    assert isinstance(anomaly_scores, TimeSeries)
    assert isinstance(ts_only, bool)
    assert isinstance(dir, (str, type(None)))
    assert isinstance(show, bool)

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
        # ax[0] = ts['value'].plot()  # use pd.Series plot method
        # #ts['value']['2016-10-16':'2016-10-18'].plot()  # plot a slice of the time series
        # ax[1] = anomaly_score.plot()

        # plot ts
        ax[0].plot(ts['epoch'], ts['value'], color='blue')
        ax[0].set_title('ts', size=12)

        # plot anomalies on top of ts
        # TODO: add anomaly ranges and plot them
        min = ts['value'].min()
        max = ts['value'].max()
        for anomaly in anomalies:
            # anomay_time_window = anomaly.get_time_window()
            # if anomay_time_window[0] == anomay_time_window[1]:
            #     anomay_exact_epoch = anomaly.exact_timestamp
            #     ax[0].plot([anomay_exact_epoch, anomay_exact_epoch], [min, max], color='gray')
            # else:
            #     # TODO: make sure the RHS is taken into consideration
            #     anomaly_time_range = range(anomay_time_window[0], anomay_time_window[1])
            #     for tm in anomaly_time_range:
            #         ax[0].plot([tm, tm], [min, max], color='gray')

            anomay_time_window = anomaly.get_time_window()
            if anomay_time_window[0] == anomay_time_window[1]:
                # anomay_exact_epoch = anomaly.exact_timestamp
                # ax[0].plot([anomay_exact_epoch, anomay_exact_epoch], [min, max], color='gray')
                print('gg')
            else:
                anomaly_time_range = range(anomay_time_window[0], anomay_time_window[1])
                ax[0].fill_between(anomaly_time_range, [min] * len(anomaly_time_range), [max] * len(anomaly_time_range),
                                   color='lightgreen', alpha=0.2)

        # plot anomaly scores
        ax[1].plot(ts['epoch'], scores, color='red')
        ax[1].set_title('scores', size=12)

        if show:
            fig.show()

        if dir is not None:
            file_name = 'yuval' + '.pdf'
            full_path = dir + file_name

            fig.set_size_inches(10, 10)
            fig.savefig(full_path, dpi=100)


if __name__ == '__main__':
    path = '/Users/yuval/Dropbox/MyData/Misc/Seebo/data/g.csv'
    ts = load_ts_data(path, epoch_col='timestamp', n_rows=10000)

    # path = '/Users/yuval/Dropbox/MyData/Misc/Seebo/data/art_noisy.csv'
    # ts = load_ts_data(path, timestamp_col='timestamp', n_rows=1000)

    # path = '/Users/yuval/Dropbox/MyData/Misc/Seebo/data/cpu4.csv'
    # ts = load_ts_data(path, epoch_col='timestamp', n_rows=30000)

    keys = ts['epoch']
    values = ts['value']
    ts_dict = dict(zip(keys, values))

    algorithm_name = 'exp_avg_detector'

    anomaly_detector = AnomalyDetector(ts_dict, algorithm_name=algorithm_name)
    anomalies = anomaly_detector.get_anomalies()
    anomaly_scores = anomaly_detector.get_all_scores()

    # plotly offline example
    from plotly.offline import plot
    import plotly.graph_objs as go

    # one = go.Scatter(
    #     x=ts['timestamp'],
    #     y=ts['value'],
    #     name='one',
    #     line=dict(color='#17BECF'),
    #     opacity=1.0)
    #
    # two = go.Scatter(
    #     x=ts['timestamp'],
    #     y=ts['value']+0.2,
    #     name='twi',
    #     line=dict(color='#7F7F7F'),
    #     opacity=1.0)
    # data = [one, two]
    # plot(data)

    trace0 = go.Scatter(
        x=[1, 2, 3, 4],
        y=[3, 4, 8, 3],
        fill=None,
        mode='lines',
        line=dict(color='rgb(143, 19, 131)',
        )
    )
    trace1 = go.Scatter(
        x=[1, 2, 3, 4],
        y=[5, 6, 10, 5],
        fill='tonexty',
        mode='lines',
        opacity=0.05,
        line=dict(color='rgb(143, 19, 131)',
        )
    )

    d = go.Scatter(
        x=[1, 2, 3, 4],
        y=[4, 5, 9, 4])

    data = [trace0, trace1, d]
    plot(data)

    plot_ts_and_anomalies(ts, anomalies, anomaly_scores, ts_only=False, dir='/Users/yuval/Desktop/', show=True)
