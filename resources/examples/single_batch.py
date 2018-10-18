import numpy as np
import pandas as pd
import csv

from src.utils.logger import log

pd.set_option('display.expand_frame_repr', False)
# pd.set_option('display.max_rows', None, 'display.max_columns', None)

log.debug('Loading single_batch data ..')
#data = pd.read_csv('/Users/yuval/Downloads/Sensor_readings.csv')
data = pd.read_csv('/Users/yuval/Downloads/Sensor_readings.csv',
                   infer_datetime_format=True,
                   parse_dates=['end_time_stamp'])
log.debug('Done loading single_batch data.')

print(data.shape)
print(data.head())
print(data.dtypes)

# DATE_FORMAT = u'%Y-%m-%d'
# TIMESTAMP_FORMAT = u'%Y-%m-%d %H:%M:%S'
#
# data.to_csv('/Users/yuval/Downloads/Sensor_readings_YUVAL.csv',
#            index=False,
#             quoting=csv.QUOTE_ALL,
#             doublequote=True,
#             date_format=TIMESTAMP_FORMAT)
#
# data = pd.read_csv('/Users/yuval/Downloads/Sensor_readings_YUVAL.csv',
#                    parse_dates=['end_time_stamp'],
#                    infer_datetime_format=True)

min_timestamp_per_sensor = data.groupby('metric_id')['end_time_stamp'].aggregate(lambda ts: ts.min())

min_timestamp_distribution =  min_timestamp_per_sensor.value_counts()
min_timestamp_distribution = min_timestamp_distribution.sort_index()

print(min_timestamp_distribution)