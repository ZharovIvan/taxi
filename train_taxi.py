import pickle
import pandas as pd
from sklearn import preprocessing
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate

def cross(mod, X, y, cv_number):
    cv_results = cross_validate(mod, X, y, cv=cv_number, scoring='neg_mean_squared_error')
    return cv_results

def grid(X, y):
    parameter_space = {'bootstrap': [True, False],
                       'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                       'max_features': ['auto', 'sqrt'],
                       'min_samples_leaf': [1, 2, 4],
                       'min_samples_split': [2, 5, 10],
                       'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
    model = RandomForestRegressor()
    GSCV = GridSearchCV(model, parameter_space, n_jobs=-1, cv=3, scoring= 'neg_mean_squared_error')
    GSCV.fit(X, y)
    return GSCV


path = r"C:\Users\miair\PycharmProjects\untitled10\train.csv" #путь к файлу reviews_train.csv
rows = 5000000
data_train = pd.read_csv(path, skip_blank_lines =True, nrows=rows) #чтение reviews_train.csv pickup_latitude
data_train = data_train.dropna(axis=0)
#Убираем лишние хначения
data_train = data_train.drop(data_train[(data_train['pickup_longitude'] <= -74.4461) | (
        data_train['pickup_longitude'] >= -73.3681) | (data_train['pickup_latitude'] <= 40.5493) | (
                                                data_train['pickup_latitude'] >= 41.0245)].index)
data_train = data_train.drop(data_train[(data_train['dropoff_longitude'] <= -74.4461) | (
        data_train['dropoff_longitude'] >= -73.3681) | (data_train['dropoff_latitude'] <= 40.5493) | (
                                                data_train['dropoff_latitude'] >= 41.0245)].index)

#Разделяем поля для большего числа параметров
data_train['date'], data_train['time'], data_train['UTC'] = data_train['pickup_datetime'].str.split(' ', 2).str
data_train = data_train.drop(columns=['UTC'])
data_train['hour'], data_train['minute'], data_train['second'] = data_train['time'].str.split(':', 2).str
data_train['year'], data_train['month'], data_train['day'] = data_train['date'].str.split('-', 2).str
data_train = data_train.drop(columns=['second'])
date = preprocessing.LabelEncoder()
date.fit(data_train['date'])
data_train['date_number'] = date.transform(data_train['date'])
#Считаем дистанцию между координатами
R = 6373.0
lat1, lon1, lat2, lon2 = pd.np.radians([data_train['pickup_latitude'], data_train['pickup_longitude'], data_train['dropoff_latitude'], data_train['dropoff_longitude']])
dlon = lon2 - lon1
dlat = lat2 - lat1
a = pd.np.sin(dlat / 2) ** 2 + pd.np.cos(lat1) * pd.np.cos(lat2) * pd.np.sin(dlon / 2) ** 2
c = 2 * pd.np.arctan(pd.np.sqrt(a), pd.np.sqrt(1 - a))
data_train['distance'] = R * c
data_NY = pd.concat((data_train['distance'], data_train['date_number'],data_train['hour'], data_train['minute'], data_train['year'], data_train['month'], data_train['day'],data_train['pickup_longitude'],data_train['pickup_latitude'],data_train['dropoff_longitude'], data_train['dropoff_latitude']), axis=1)
data_NY = pd.DataFrame(data_NY)
X = data_NY
y = data_train['fare_amount']

del data_train, dlon, dlat, lat1, lon1, lat2, lon2, data_NY, a, c
print('Кросс-валидация(1), подобор параметров(2), сохранение модели(3) или тестирование(4): ')
Choose_number = input()
model = RandomForestRegressor(bootstrap= False, max_depth= 50, max_features= 'sqrt', min_samples_leaf= 2, min_samples_split= 5, n_estimators=600)


if (Choose_number == '4'):
    # Тестирование
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)
    model.fit(X_train, y_train)
    predict_train = model.predict(X_train)
    predict_test = model.predict(X_test)
    print(sqrt(mean_squared_error(y_train, predict_train)), sqrt(mean_squared_error(y_test, predict_test)))
elif (Choose_number == '3'):
    # Обучение и сохранение модели
    model.fit(X, y)
    predict_train = model.predict(X)
    del X, y
    with open('model_NY.pkl', 'wb') as fid:
       pickle.dump((date, model), fid)

    print(sqrt(mean_squared_error(y, predict_train)))
elif (Choose_number == '2'):
    # Подбор параметров
    clf = grid(X, y)
    # Лучший результат
    print('Best parameters found:\n', clf.best_params_)
    # Все результаты
    print(" ")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
elif (Choose_number == '1'):
    # Кросс-валидация
    cv_results = cross(model, X, y, 10)
    print('Значение кросс-валидации тренировочного набора:', cv_results['train_score'])
    print('Значение кросс-валидации тестового набора:', cv_results['test_score'])




