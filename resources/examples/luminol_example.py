import pandas as pd
import time
import matplotlib.pyplot as plt
from luminol.anomaly_detector import AnomalyDetector

from src.utils.logger import log

# pd.set_option('display.max_rows', None, 'display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

path = '/Users/yuval/Dropbox/MyData/Misc/Seebo/data/g.csv'
ts = pd.read_csv(path)  # , parse_dates=['timestamp'], infer_datetime_format=True)
ts = ts.iloc[0:10000]
ts.rename(columns={'timestamp': 'epoch'}, inplace=True)
# convert to timestamp
ts['timestamp'] = pd.to_datetime(ts['epoch'], unit='s')

keys = ts['epoch']
values = ts['value']
ts_dict = dict(zip(keys, values))

ts = ts.set_index('timestamp')

#plt.plot(ts['value'])
ts['value'].plot()  # use pd.Series plot method
#ts['value']['2016-10-16':'2016-10-18'].plot()  # plot a slice of the time series
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
