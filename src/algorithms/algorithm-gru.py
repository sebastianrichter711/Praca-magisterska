import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Activation, GRU
from keras.optimizers import SGD
from keras.metrics import Accuracy, RootMeanSquaredError
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, confusion_matrix
import seaborn as sn
from math import sqrt
import datetime
import matplotlib.dates as mdates

def create_list_of_dates(dates):
  list_of_dates = []
  for d in dates:
    splitted_date = d.split("-")
    list_of_dates.append(splitted_date)
  return list_of_dates

plt.style.use('fivethirtyeight')

location = "wieliczka"
data = pd.read_csv(
    "D:/Studia/Praca-magisterska/dane-z-PV/dane-do-badania/" + location + "-all.csv")

train_size = int(len(data) * 0.8)
test_size = len(data) - train_size
train, test = data.iloc[0:train_size], data.iloc[train_size:len(data)]
print(len(train), len(test))

list_of_dates = create_list_of_dates(data["datetime"])
list_of_test_dates = create_list_of_dates(test["datetime"])

data.pop("name")
data.pop("datetime")
data.pop("preciptype")
data.pop("severerisk")
data.pop("snow")
data.pop("snowdepth")

print(data.info())

print ("\nMissing values :  ", data.isnull().any())

x = [datetime.datetime(int(l[0]),int(l[1]),int(l[2])) for l in list_of_dates]

fig, ax = plt.subplots()
ax.plot(x, data["energy_produced"])
ax.xaxis.set_major_locator(mdates.DayLocator(interval=365))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
plt.ylabel("Energia wyprodukowana [kWh]")
plt.title("Produkcja energii elektrycznej z instalacji PV w Wieliczce")
plt.show()

sn.heatmap(data.corr(), annot=True)
plt.title("Macierz korelacji")
plt.show()

# f_columns = ['tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike', 'dew',
# 'humidity', 'precip', 'precipprob', 'precipcover', 'sealevelpressure', 'cloudcover', 'solarradiation',
# 'uvindex']

f_columns = ['tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike', 'dew',
'humidity', 'precip', 'precipprob', 'precipcover', 'windgust', 'windspeed', 'winddir', 'sealevelpressure', 'cloudcover', 'visibility', 'solarradiation',
'solarenergy', 'uvindex']

features_transformer = RobustScaler()

features_transformer = features_transformer.fit(train[f_columns].to_numpy())

train.loc[:, f_columns] = features_transformer.transform(
  train[f_columns].to_numpy()
)

test.loc[:, f_columns] = features_transformer.transform(
  test[f_columns].to_numpy()
)

energy_values_transformer = RobustScaler()

energy_values_transformer = energy_values_transformer.fit(train[['energy_produced']])

train['energy_produced'] = energy_values_transformer.transform(train[['energy_produced']])

test['energy_produced'] = energy_values_transformer.transform(test[['energy_produced']])

X_train, y_train = train.loc[:, f_columns].to_numpy(), train.loc[:, "energy_produced"].to_numpy()
X_test, y_test = test.loc[:, f_columns].to_numpy(), test.loc[:, "energy_produced"].to_numpy()

print("Train")
print(train)
print("Test")
print(test)
print("Enrgia wyporduk")
print(y_train)
print("Enrgia wyporduk(test)")
print(y_test)

X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

print(X_train.shape, y_train.shape)

# GRU
model = Sequential()
model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), activation='tanh'))
model.add(Dropout(0.2))
model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],X_train.shape[2]), activation='tanh'))
model.add(Dropout(0.2))
model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],X_train.shape[2]), activation='tanh'))
model.add(Dropout(0.2))
model.add(GRU(units=50, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer=SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False),loss='mean_squared_error',
metrics=['mse', 'mae', RootMeanSquaredError()])

history = model.fit(X_train,y_train,epochs=50,batch_size=32)

plt.plot(history.history['loss'], label='train')
plt.title("Wykres funkcji straty")
plt.legend()
plt.show()

acc = model.evaluate(X_test, y_test)
print(model.metrics_names)
print("test loss:", acc)

y_pred = model.predict(X_test)

y_train_inv = energy_values_transformer.inverse_transform(y_train.reshape(-1,1))
y_test_inv = energy_values_transformer.inverse_transform(y_test.reshape(-1,1))
y_pred_inv = energy_values_transformer.inverse_transform(y_pred)

fig, ax = plt.subplots()
x = [datetime.datetime(int(l[0]),int(l[1]),int(l[2])) for l in list_of_test_dates]

print(len(y_test_inv))
print(len(y_pred_inv))

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=100))
plt.plot(x, y_test_inv.flatten(),label='Dane rzeczywiste')
plt.plot(x, y_pred_inv.flatten(),label='Prognoza')
plt.gcf().autofmt_xdate()
plt.ylabel("Energia wyprodukowana [kWh]")
plt.title("Wykres danych testowych")
plt.legend()
plt.show()

print("Mean square error: " + str(mean_squared_error(y_test_inv, y_pred_inv)))
print("Mean absolute error: " + str(mean_absolute_error(y_test_inv, y_pred_inv)))
print("Root mean square error: " + str(sqrt(mean_squared_error(y_test_inv, y_pred_inv))))
print("Mean Absolute Percentage Error: " + str(mean_absolute_percentage_error(y_test_inv, y_pred_inv)))

forecast_data = pd.read_csv(
    "D:/Studia/Praca-magisterska/dane-z-PV/dane-do-badania/" + location + "-forecast-10.csv")

list_of_forecast_dates = create_list_of_dates(forecast_data["datetime"])

forecast_data.pop("name")
forecast_data.pop("datetime")
forecast_data.pop("preciptype")
forecast_data.pop("severerisk")
forecast_data.pop("snow")
forecast_data.pop("snowdepth")

print(forecast_data.info())

print(len(forecast_data))

# f_columns = ['tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike', 'dew',
# 'humidity', 'precip', 'precipprob', 'precipcover', 'sealevelpressure', 'cloudcover', 'solarradiation',
# 'uvindex']

f_columns = ['tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike', 'dew',
'humidity', 'precip', 'precipprob', 'precipcover', 'windgust', 'windspeed', 'winddir', 'sealevelpressure', 'cloudcover', 'visibility', 'solarradiation',
'solarenergy', 'uvindex']

features_transformer_2 = RobustScaler()

features_transformer_2 = features_transformer_2.fit(forecast_data[f_columns].to_numpy())

forecast_data.loc[:, f_columns] = features_transformer_2.transform(
  forecast_data[f_columns].to_numpy()
)

energy_values_transformer_2 = RobustScaler()

real_data_for_chart = forecast_data['energy_produced']

energy_values_transformer_2 = energy_values_transformer_2.fit(forecast_data[['energy_produced']])

forecast_data['energy_produced'] = energy_values_transformer_2.transform(forecast_data[['energy_produced']])

X_forecast, y_forecast = forecast_data.loc[:, f_columns].to_numpy(), forecast_data.loc[:, "energy_produced"].to_numpy()
X_forecast = X_forecast.reshape((X_forecast.shape[0], 1, X_forecast.shape[1]))

print(X_forecast.shape)

y_pred_forecast = model.predict(X_forecast)
y_pred_forecast_inv = energy_values_transformer_2.inverse_transform(y_pred_forecast)

print("Prognoza")
print(y_pred_forecast_inv)
print("First forecast")
print(y_pred_forecast_inv[0])

fig, ax = plt.subplots()
x = [datetime.datetime(int(l[0]),int(l[1]),int(l[2])) for l in list_of_forecast_dates]
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.plot(x, real_data_for_chart, label='Dane rzeczywiste')
plt.plot(x, y_pred_forecast_inv.flatten(), label='Prognoza')
plt.ylabel("Energia wyprodukowana [kWh]")
plt.title("Prognoza produkcji energii elektrycznej z instalacji PV w Wieliczce na kwiecień 2023")
plt.legend()
plt.show()

print("Mean square error: " + str(mean_squared_error(real_data_for_chart, y_pred_forecast_inv)))
print("Mean absolute error: " + str(mean_absolute_error(real_data_for_chart, y_pred_forecast_inv)))
print("Root mean square error: " + str(sqrt(mean_squared_error(real_data_for_chart, y_pred_forecast_inv))))









































