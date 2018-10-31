import numpy as np
import pandas as pd
import random
import time

from src.utils.logger import log

# pd.set_option('display.max_rows', None, 'display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

from luminol.anomaly_detector import AnomalyDetector

# ts = {0: 0, 1: 0.5, 2: 1, 3: 1, 4: 1, 5: 0, 6: 0, 7: 0, 8: 0}
ts = {
    '1490323038': 3,
    '1490323048': 4,
    '1490323058': 6,
    '1490323068': 78,
    '1490323078': 67,
    '1490323088': 5,
}

my_detector = AnomalyDetector(ts)
score = my_detector.get_all_scores()
anom_score = []

for (timestamp, value) in score.iteritems():
    t_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
    anom_score.append([t_str, value])

for score in anom_score:
    print(score)