import numpy as np
import pandas as pd
import random
import string
import datetime

from src.utils.logger import log

# pd.set_option('display.max_rows', None, 'display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

def generate_fake_data(n_batches=150, n_sensors=300, batch_default_probability=0.3):
    log.debug('Generating fake data: {} batches, {} sensors.'.format(n_batches, n_sensors))

    batch_ids = ['B-' + ''.join(random.choice('0123456789ABCDEF') for i in range(8)) for _ in range(n_batches)]
    sensor_ids = ['S-' + ''.join(random.choice(string.ascii_lowercase) for i in range(8)) for _ in range(n_sensors)]

    data = []

    for batch_id in batch_ids:

        # create target labels
        batch_label = np.random.binomial(1, batch_default_probability, 1)[0]

        # create data
        for sensor_id in sensor_ids:
            hour = random.randint(10, 12)
            min = random.choice(np.arange(0, 60, 5))
            min_timestamp = pd.Timestamp(2018, 11, 1, hour, min)
            sensor_duration_in_minutes = random.choice(np.arange(300, 360, 5))  # 5 to 6 hours
            max_timestamp = min_timestamp + datetime.timedelta(minutes=float(sensor_duration_in_minutes))
            timestamps = pd.date_range(min_timestamp, max_timestamp, freq='5min')

            if batch_label == 0:
                values = np.random.normal(0, 1, len(timestamps))
            else:
                values = np.random.normal(1.5, 1, len(timestamps))

            for timestamp, value in zip(timestamps, values):
                data.append([batch_id, sensor_id, timestamp, value, batch_label])

    data = pd.DataFrame(data, columns=['batch_id', 'sensor_id', 'timestamp', 'value', 'batch_label'])

    log.debug('Done generating fake data: {} batches, {} sensors.'.format(n_batches, n_sensors))

    return data


if __name__ == '__main__':
    n_batches = 100
    n_sensors = 10
    batch_default_probability = 0.3

    data = generate_fake_data(n_batches, n_sensors, batch_default_probability)

    print(data.shape)
    print(data.head())

    data.to_csv('/Users/yuval/Desktop/mock_data.csv', index=False)
