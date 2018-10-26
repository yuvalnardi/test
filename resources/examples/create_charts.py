import pandas as pd
import random
import matplotlib.pyplot as plt

from src.utils.logger import log

# pd.set_option('display.max_rows', None, 'display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


def create_prospect_chart(data, batch_id, sensor_id):
    log.debug('Creating prospect charts (Forward View) for batch {} and sensor {}.'.format(
        batch_id, sensor_id))

    assert isinstance(data, pd.DataFrame)
    expected_columns = ['batch_id', 'sensor_id', 'timestamp', 'value', 'batch_label']
    assert set(data.columns) == set(expected_columns)
    s = 'batch {} has no records for sensor {}'.format(batch_id, sensor_id)
    assert sensor_id in data.loc[data['batch_id'] == batch_id, 'sensor_id'].unique(), s

    # TODO:
    x = data.loc[(data['batch_id'] == batch_id) & (data['sensor_id'] == sensor_id), 'timestamp']
    minutes_since_start = [(x - x.iloc[0]).iloc[i].total_seconds()/60 for i in range(len(x))]
    y = data.loc[(data['batch_id'] == batch_id) & (data['sensor_id'] == sensor_id), 'value']

    y_average_normal = y + 2
    y_average_normal_lower = y_average_normal - 0.4
    y_average_normal_upper = y_average_normal + 0.4

    plt.plot(minutes_since_start, y, marker='', color='red', label='Batch id: {}'.format(batch_id))
    #plt.plot(minutes_since_start, y_average_normal, marker='', color='lightgreen', linewidth=10, label='Normal Batches')
    plt.fill_between(minutes_since_start, y_average_normal_lower, y_average_normal_upper, color='lightgreen', alpha='0.5')
    plt.title('Prospect (Forward) View: Sensor id {}'.format(sensor_id),
              fontsize=15)
    plt.xlabel('Minutes (since start)')
    plt.legend()
    plt.show()

    log.debug('Done creating prospect charts (Forward View) for batch {} and sensor {}.'.format(
        batch_id, sensor_id))


def create_retrospect_chart(data, batch_id, sensor_id):
    log.debug('Creating retrospect charts (Backward View) for batch {} and sensor {}.'.format(
        batch_id, sensor_id))

    assert isinstance(data, pd.DataFrame)
    expected_columns = ['batch_id', 'sensor_id', 'timestamp', 'value', 'batch_label']
    assert set(data.columns) == set(expected_columns)
    s = 'batch {} has no records for sensor {}'.format(batch_id, sensor_id)
    assert sensor_id in data.loc[data['batch_id'] == batch_id, 'sensor_id'].unique(), s

    # TODO:

    log.debug('Done creating retrospect charts (Forward View) for batch {} and sensor {}.'.format(
        batch_id, sensor_id))


if __name__ == '__main__':
    fullpath = '/Users/yuval/Desktop/mock_data.csv'
    data = pd.read_csv(fullpath, parse_dates=['timestamp'], infer_datetime_format=True)

    batch_id = random.choice(data['batch_id'].unique())
    sensor_id = random.choice(data['sensor_id'].unique())

    create_prospect_chart(data, batch_id, sensor_id)
    #create_retrospect_chart(data, batch_id, sensor_id)
