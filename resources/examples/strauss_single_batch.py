import pandas as pd
import csv

from src.utils.logger import log

pd.set_option('display.expand_frame_repr', False)
# pd.set_option('display.max_rows', None, 'display.max_columns', None)

change_timestamp_format = False

if change_timestamp_format:
    log.debug('Loading single_batch data ..')
    date_parser = lambda x: pd.datetime.strptime(x, '%d/%m/%y %H:%M')
    data = pd.read_csv('/Users/yuval/Downloads/Sensor_readings.csv',
                       parse_dates=['end_time_stamp'],
                       infer_datetime_format=True,
                       date_parser=date_parser)
    log.debug('Done loading single_batch data.')

    date_format = u'%Y-%m-%d'
    timestamp_format = u'%Y-%m-%d %H:%M:%S'

    log.debug('Persisting to csv new timestamp format ..')
    data.to_csv('/Users/yuval/Downloads/Sensor_readings_YUVAL.csv',
                index=False,
                quoting=csv.QUOTE_ALL,
                doublequote=True,
                date_format=timestamp_format)
    log.debug('Done persisting to csv new timestamp format.')
else:
    log.debug('Loading single_batch data ..')
    data = pd.read_csv('/Users/yuval/Desktop/Sensor_readings_YUVAL.csv',
                       parse_dates=['end_time_stamp'],
                       infer_datetime_format=True)
    log.debug('Done loading single_batch data.')

min_timestamp_per_sensor = data.groupby('metric_id')['end_time_stamp'].aggregate(lambda ts: ts.min())

min_timestamp_distribution = min_timestamp_per_sensor.value_counts()
min_timestamp_distribution = min_timestamp_distribution.sort_index()

print(min_timestamp_distribution)

time_range_per_sensor = data.groupby('metric_id')['end_time_stamp'].aggregate(
    lambda ts: (ts.max() - ts.min()).total_seconds() / 3600)

time_range_per_sensor.value_counts().sort_index()