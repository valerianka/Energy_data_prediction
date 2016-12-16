import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pylab as plt
import re
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

# Delete meaningless "07:00" or "08:00" at the end of each entry in date_time column
def parse_data(string):
    string = re.sub("-[\d]+:[\d]+$", "", string)
    return datetime.strptime(string, '%Y-%m-%d %H:%M:%S')

# Read time series data with index of date_time column.
data = pd.read_csv('data.csv', names=['date_time', 
    'kwh', 'temperature', 'date', 'time', 'dow', 'month'],
    index_col=['date_time'], header=0, parse_dates=['date_time'], 
    date_parser=parse_data)
# Omit the data after 2013/11
data = data[:'2013-11']
# Drop NaNs in kwh column since it's response variable we need it to train algorithm.
data = data.dropna(subset=['kwh'], how='all')
# Fill NaNs in temperature with closest non-NaN before and all the rest fill in forward.
data.fillna(method='bfill', inplace=True)
data.fillna(method='ffill', inplace=True)
# Add time_catogory column with time categorical variable
data['time_category'] = data['time'].astype('category').cat.codes
# Plot kwh and temperature changing over time to show overall pattern.
plt.plot(data['kwh'], color='blue', label='kwh with time')
plt.plot(data['temperature'], color='red', label='temperature with time')
plt.legend(loc='upper left')
plt.show()
# Plot temperature vs kwh shows funnel-shaped trend
plt.scatter(data['temperature'], data['kwh'],  color='black', label='kwh vs temperature')
plt.legend(loc='upper right')
plt.show()
# Closeup view of kwh and temperature changing in January 2013 to see weekly pattern in kwh plot
plt.plot(data['2013-01']['kwh'], color='blue', label='kwh in January 2013')
plt.plot(data['2013-01']['temperature'], color='red', label='temperature in January 2013')
plt.legend(loc='upper left')
plt.show()
# Weekday and weekend kwh data
plt.plot(data['2013-02-01']['kwh'], color='black', label='kwh on weekday')
plt.plot(data['2013-02-02']['kwh'], color='blue', label='kwh on weekend')
plt.legend(loc='upper right')
plt.show()

time_uniques, time = np.unique(data['time'], return_inverse=True)
# kwh distribution over time of the day and some outliers
plt.scatter(time, data['kwh'],  color='black')
plt.xticks(range(len(time_uniques)), time_uniques, rotation='vertical')
plt.legend(loc='upper right')
plt.show()
# kwh distribution over day of week with outliers
weekday_uniques, week = np.unique(data['dow'], return_inverse=True)
plt.scatter(week, data['kwh'], color='black')
plt.xticks(range(len(weekday_uniques)), weekday_uniques)
plt.legend(loc='upper right')
plt.show()
# Remove extreme outliers that lies more than 1.5 IQR below median
q75, q25 = np.percentile(data['kwh'], [75 ,25])
iqr = q75 - q25
kwh_median = np.median(data['kwh'])
data = data[data.kwh >= kwh_median - 1.5 * iqr]

data_train = data[:'2013-10']
data_test = data['2013-11']
X_train = data_train[['temperature', 'dow', 'time_category']]
X_test = data_test[['temperature', 'dow', 'time_category']]
y_train = data_train['kwh']
y_test = data_test['kwh']


class Forecaster(object):
    def __init__(self, parameters={}):
        self.estimator = GradientBoostingRegressor(**parameters)
        self.trained_model = None

    def fit(self, X_train, y_train):
        self.trained_model = self.estimator.fit(X_train, y_train)
        
    def predict(self, X_test):
        if self.trained_model is not None:
            return self.trained_model.predict(X_test)
        else:
            return "Error: trying to predict using unfitted model"

    def fit_parameters(self, X_train, y_train, parameters):
        self.trained_model = GridSearchCV(self.estimator, parameters).fit(X_train, y_train)
        self.parameters = self.trained_model.best_params_

    def plot_predictions(self, y_test, predictions):
        predicted_data = pd.DataFrame(data=predictions, index=y_test.index, columns=['predictions'])
        plt.plot(predicted_data['predictions'], color='red', label='predicted kwh')
        plt.plot(y_test, color='blue', label='observed kwh')
        plt.legend(loc='upper right')
        plt.show()

class Evaluator(object):
    # Forecaster instance should implement functions fit(X_train, y_train) and predict(X_test, y_test).
    def __init__(self, forecaster):
        self.forecaster = forecaster

    def validate_in_sample(self, X_sample, y_sample):
        training_size = int(len(X_sample) * 0.9)
        X_train = X_sample[:training_size]
        y_train = y_sample[:training_size]
        X_test = X_sample[training_size:]
        y_test = y_sample[training_size:]
        self.forecaster.fit(X_train, y_train)
        predictions = self.forecaster.predict(X_test)
        r_squared = r2_score(y_test, predictions)
        print("In sample R^2 score: %.2f" % r_squared)

    def validate_out_of_sample(self, X_test, y_test):
        predictions = self.forecaster.predict(X_test)
        r_squared = r2_score(y_test, predictions)
        print("Out of sample R^2 score: %.2f" % r_squared)

parameters = {'n_estimators': 70, 'learning_rate': 0.05, 'max_depth': 4}

param_grid = {'n_estimators': range(30, 100, 10), 
               'learning_rate': [0.05, 0.1, 0.15, 0.2], 'max_depth': range(1, 6)}
f = Forecaster(parameters)
f.predict(X_test)
# f.fit_parameters(X_train, y_train, param_grid)
f.fit(X_train, y_train)
predictions = f.predict(X_test)
f.plot_predictions(y_test, predictions)
evl = Evaluator(f)
evl.validate_in_sample(X_train, y_train)
evl.validate_out_of_sample(X_test, y_test)
