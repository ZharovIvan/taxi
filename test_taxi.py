import pickle
import pandas as pd

with open('model_NY.pkl', 'rb') as fid:
    date, model = pickle.load(fid)
path = r"C:\Users\miair\PycharmProjects\untitled10\test.csv"
path_predictions = r"C:\Users\miair\PycharmProjects\untitled10\sample_submission.csv"
data_test = pd.read_csv(path, skip_blank_lines =True)
data_predictions = pd.read_csv(path_predictions)
data_test = data_test.dropna(axis=0)
data_test = data_test.drop(data_test[(data_test['pickup_longitude'] <= -74.4461) | (
        data_test['pickup_longitude'] >= -73.3681) | (data_test['pickup_latitude'] <= 40.5493) | (
        data_test['pickup_latitude'] >= 41.0245)].index)
data_test = data_test.drop(data_test[(data_test['dropoff_longitude'] <= -74.4461) | (
        data_test['dropoff_longitude'] >= -73.3681) | (data_test['dropoff_latitude'] <= 40.5493) | (
        data_test['dropoff_latitude'] >= 41.0245)].index)

data_test['date'], data_test['time'], data_test['UTC'] = data_test['pickup_datetime'].str.split(' ', 2).str
data_test = data_test.drop(columns=['UTC'])
data_test['hour'], data_test['minute'], data_test['second'] = data_test['time'].str.split(':', 2).str
data_test['year'], data_test['month'], data_test['day'] = data_test['date'].str.split('-', 2).str
data_test = data_test.drop(columns=['second'])
data_test['date_number'] = date.transform(data_test['date'])
R = 6373.0
lat1, lon1, lat2, lon2 = pd.np.radians([data_test['pickup_latitude'], data_test['pickup_longitude'], data_test['dropoff_latitude'], data_test['dropoff_longitude']])
dlon = lon2 - lon1
dlat = lat2 - lat1
a = pd.np.sin(dlat / 2) ** 2 + pd.np.cos(lat1) * pd.np.cos(lat2) * pd.np.sin(dlon / 2) ** 2
c = 2 * pd.np.arctan(pd.np.sqrt(a), pd.np.sqrt(1 - a))
data_test['distance'] = R * c
data_NY = pd.concat((data_test['distance'], data_test['date_number'],data_test['hour'], data_test['minute'], data_test['year'], data_test['month'], data_test['day'],data_test['pickup_longitude'],data_test['pickup_latitude'],data_test['dropoff_longitude'], data_test['dropoff_latitude']), axis=1)
data_NY = pd.DataFrame(data_NY)
X = data_NY
predict_test = model.predict(X)
y_predict_test = pd.DataFrame({'fare_amount': predict_test})
data_predictions['fare_amount'] = y_predict_test['fare_amount']
data_predictions.to_csv('prediction_taxi.csv', sep=',', index=False)  # запись в csv предсказаний оценок