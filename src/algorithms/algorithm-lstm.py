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
from numpy import savetxt
import seaborn as sn

plt.style.use('fivethirtyeight')

location = "wieliczka"
data = pd.read_csv(
    "D:/Studia/Praca-magisterska/dane-z-PV/dane-do-badania/" + location + "-all.csv")

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

#data plot
plt.plot(data["energy_produced"])
plt.show()

copied_data = data
corr_matrix = data.corr()
sn.heatmap(corr_matrix, annot=True)
plt.show()

train_size = int(len(data) * 0.8)
test_size = len(data) - train_size
train, test = data.iloc[0:train_size], data.iloc[train_size:len(data)]
print(len(train), len(test))

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

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 7

# reshape to [samples, time_steps, n_features]
print(train.head(10))
print(train.loc[:,["tempmax", "tempmin"]])

X_train, y_train = create_dataset(train.loc[:, f_columns], train.energy_produced, time_steps)
X_test, y_test = create_dataset(test.loc[:, f_columns], test.energy_produced, time_steps)

print(X_train.shape, y_train.shape)

model = Sequential()

# LSTM

# model.add(
#   Bidirectional(
#     LSTM(
#       units=128,
#       input_shape=(X_train.shape[1], X_train.shape[2])
#     )
#   )
# )

# model.add(Dropout(rate=0.2))
# model.add(Dense(units=1))

model.add(LSTM(units=400, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=400, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=400))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(
    X_train, y_train,
    epochs=75,
    batch_size=32,
    validation_split=0.1,
    shuffle=False
)

y_pred = model.predict(X_test)

y_train_inv = en_transformer.inverse_transform(y_train.reshape(-1,1))
y_test_inv = en_transformer.inverse_transform(y_test.reshape(-1,1))
y_pred_inv = en_transformer.inverse_transform(y_pred)

plt.plot(y_test_inv.flatten(), label='true')
plt.plot(y_pred_inv.flatten(), label='predicted')
plt.legend()
plt.show()

mse = mean_squared_error(y_test_inv, y_pred_inv)
print("Mean square error: " + str(mse))
mae = mean_absolute_error(y_test_inv, y_pred_inv)
print("Mean absolute error: " + str(mae))
rmse = sqrt(mean_squared_error(y_test_inv, y_pred_inv))
print("Root mean square error: " + str(rmse))

def mape_function(y_test, pred):
    y_test, pred = np.array(y_test), np.array(pred)
    mape = np.mean(np.abs((y_test - pred) / y_test))
    return mape

mape = mape_function(y_test_inv, y_pred_inv)
print("Mean Absolute Percentage Error: " + str(mape))

forecast_data = pd.read_csv(
    "D:/Studia/Praca-magisterska/dane-z-PV/dane-do-badania/" + location + "-forecast.csv")

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

data = forecast_data['energy_produced']

en_transformer_2 = en_transformer_2.fit(forecast_data[['energy_produced']])

forecast_data['energy_produced'] = en_transformer_2.transform(forecast_data[['energy_produced']])

X_forecast, y_forecast = create_dataset(forecast_data.loc[:, f_columns], forecast_data.energy_produced, 1)

print(X_forecast.shape)

y_pred_forecast = model.predict(X_forecast)

y_pred_forecast_inv = en_transformer_2.inverse_transform(y_pred_forecast)
#y_data_real_inv = en_transformer_2.inverse_transform(forecast_data.energy_produced)

print("Prognoza")
print(y_pred_forecast_inv)
#print("Dane rzeczywiste")
#print(y_data_real_inv)

plt.plot(data, label='Dane rzeczywiste')
plt.plot(y_pred_forecast_inv.flatten(), label='Prognoza')
plt.legend()
plt.show()




















