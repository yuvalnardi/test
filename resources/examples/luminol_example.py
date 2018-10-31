import numpy as np
import pandas as pd
import random
import time
from luminol.modules.time_series import TimeSeries
import matplotlib.pyplot as plt

from src.utils.logger import log

# pd.set_option('display.max_rows', None, 'display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

from luminol.anomaly_detector import AnomalyDetector

# ts = {
#     '1490323038': 3,
#     '1490323048': 4,
#     '1490323058': 6,
#     '1490323068': 1078,
#     '1490323078': 1607,
#     '1490323088': 5,
# }
#
# my_detector = AnomalyDetector(ts)
# score = my_detector.get_all_scores()
# anomalies = my_detector.get_anomalies()
# anom_score = []
#
# for (timestamp, value) in score.iteritems():
#     t_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
#     anom_score.append([t_str, value])
#
# for score in anom_score:
#     print(score)


from luminol.utils import to_epoch

path = '/Users/yuval/Dropbox/MyData/Misc/Seebo/data/g.csv'
ts = pd.read_csv(path) # , parse_dates=['timestamp'], infer_datetime_format=True)
ts = ts.iloc[0:10000]
# convert to epoch (unix time)
#ts['epoch'] = ts['timestamp'].transform(lambda timestamp: to_epoch(timestamp))

keys = ts['timestamp']
values = ts['value']
ts_dict = dict(zip(keys, values))

plt.plot(ts['value'])
plt.show()


my_detector = AnomalyDetector(ts_dict)
score = my_detector.get_all_scores()
anomalies = my_detector.get_anomalies()
anom_score = []

for (timestamp, value) in score.iteritems():
    t_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
    anom_score.append([t_str, value])

for score in anom_score:
    print(score)