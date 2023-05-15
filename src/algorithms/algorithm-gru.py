import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Activation, GRU
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
# data.pop("precip")
# data.pop("precipprob")
# data.pop("precipcover")
data.pop("preciptype")
# data.pop("winddir")
# data.pop("windgust")
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

f_columns = ['tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike', 'dew',
'humidity', 'precip', 'precipprob', 'precipcover', 'windgust', 'windspeed', 'winddir', 'sealevelpressure', 'cloudcover', 'visibility', 'solarradiation',
'solarenergy', 'uvindex']

f_transformer = RobustScaler()

f_transformer = f_transformer.fit(train[f_columns].to_numpy())

train.loc[:, f_columns] = f_transformer.transform(
  train[f_columns].to_numpy()
)

test.loc[:, f_columns] = f_transformer.transform(
  test[f_columns].to_numpy()
)

en_transformer = RobustScaler()

en_transformer = en_transformer.fit(train[['energy_produced']])

train['energy_produced'] = en_transformer.transform(train[['energy_produced']])

test['energy_produced'] = en_transformer.transform(test[['energy_produced']])

# def create_dataset(X, y, time_steps=1):
#     Xs, ys = [], []
#     for i in range(len(X) - time_steps):
#         v = X.iloc[i:(i + time_steps)].values
#         Xs.append(v)
#         ys.append(y.iloc[i + time_steps])
#     return np.array(Xs), np.array(ys)

# time_steps = 7

# # reshape to [samples, time_steps, n_features]
# print(train.head(10))
# print(train.loc[:,["tempmax", "tempmin"]])

# X_train, y_train = create_dataset(train.loc[:, f_columns], train.energy_produced, time_steps)
# X_test, y_test = create_dataset(test.loc[:, f_columns], test.energy_produced, time_steps)

# print(X_train.shape, y_train.shape)

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
# First GRU layer with Dropout regularisation
model.add(GRU(units=500, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), activation='tanh'))
model.add(Dropout(0.2))
# Second GRU layer
model.add(GRU(units=500, return_sequences=True, input_shape=(X_train.shape[1],X_train.shape[2]), activation='tanh'))
model.add(Dropout(0.2))
# Third GRU layer
model.add(GRU(units=500, return_sequences=True, input_shape=(X_train.shape[1],X_train.shape[2]), activation='tanh'))
model.add(Dropout(0.2))
# Fourth GRU layer
model.add(GRU(units=500, activation='tanh'))
model.add(Dropout(0.2))
# The output layer
model.add(Dense(units=1))
# Compiling the RNN
model.compile(optimizer='adam',loss='mean_squared_error')
# Fitting to the training set
model.fit(X_train, y_train, epochs=75, batch_size=32, validation_split=0.1, shuffle=False)

acc = model.evaluate(X_test, y_test)
print("test loss, test acc:", acc)

y_pred = model.predict(X_test)

y_train_inv = en_transformer.inverse_transform(y_train.reshape(-1,1))
y_test_inv = en_transformer.inverse_transform(y_test.reshape(-1,1))
y_pred_inv = en_transformer.inverse_transform(y_pred)

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

def mape_function(y_test, pred):
    y_test, pred = np.array(y_test), np.array(pred)
    mape = np.mean(np.abs((y_test - pred) / y_test))
    return mape

print("Mean Absolute Percentage Error: " + str(mape_function(y_test_inv, y_pred_inv)))

forecast_data = pd.read_csv(
    "D:/Studia/Praca-magisterska/dane-z-PV/dane-do-badania/" + location + "-forecast.csv")

list_of_forecast_dates = create_list_of_dates(forecast_data["datetime"])

forecast_data.pop("name")
forecast_data.pop("datetime")
# forecast_data.pop("precip")
# forecast_data.pop("precipprob")
# forecast_data.pop("precipcover")
forecast_data.pop("preciptype")
# forecast_data.pop("winddir")
# forecast_data.pop("windgust")
forecast_data.pop("severerisk")
forecast_data.pop("snow")
forecast_data.pop("snowdepth")

print(forecast_data.info())

print(len(forecast_data))

f_columns = ['tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike', 'dew',
'humidity', 'precip', 'precipprob', 'precipcover', 'windgust', 'windspeed', 'winddir', 'sealevelpressure', 'cloudcover', 'visibility', 'solarradiation',
'solarenergy', 'uvindex']

f_transformer_2 = RobustScaler()

f_transformer_2 = f_transformer_2.fit(forecast_data[f_columns].to_numpy())

forecast_data.loc[:, f_columns] = f_transformer_2.transform(
  forecast_data[f_columns].to_numpy()
)

en_transformer_2 = RobustScaler()

real_data_for_chart = forecast_data['energy_produced']

en_transformer_2 = en_transformer_2.fit(forecast_data[['energy_produced']])

forecast_data['energy_produced'] = en_transformer_2.transform(forecast_data[['energy_produced']])

X_forecast, y_forecast = forecast_data.loc[:, f_columns].to_numpy(), forecast_data.loc[:, "energy_produced"].to_numpy()
X_forecast = X_forecast.reshape((X_forecast.shape[0], 1, X_forecast.shape[1]))

print(X_forecast.shape)

y_pred_forecast = model.predict(X_forecast)

y_pred_forecast_inv = en_transformer_2.inverse_transform(y_pred_forecast)
#y_data_real_inv = en_transformer_2.inverse_transform(forecast_data.energy_produced)

print("Prognoza")
print(y_pred_forecast_inv)
print("First forecast")
print(y_pred_forecast_inv[0])


fig, ax = plt.subplots()
x = [datetime.datetime(int(l[0]),int(l[1]),int(l[2])) for l in list_of_forecast_dates]
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
plt.plot(x, real_data_for_chart, label='Dane rzeczywiste')
plt.plot(x, y_pred_forecast_inv.flatten(), label='Prognoza')
plt.ylabel("Energia wyprodukowana [kWh]")
plt.title("Prognoza produkcji energii elektrycznej z instalacji PV w Wieliczce na kwiecień 2023")
plt.legend()
plt.show()









































