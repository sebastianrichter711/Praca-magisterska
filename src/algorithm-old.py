import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Activation
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from numpy import savetxt

plt.style.use('fivethirtyeight')

location = "apa"
data = pd.read_csv(
    "D:/Studia/Praca-magisterska/dane-z-PV/dane-do-badania/" + location + "-all.csv")

data.pop("time")
data.pop("wdir")
data.pop("prcp")

print(data.info())

#data plot
plt.plot(data["energy_produced"])
plt.show()

train_size = int(len(data) * 0.8)
test_size = len(data) - train_size
train, test = data.iloc[0:train_size], data.iloc[train_size:len(data)]
print(len(train), len(test))

f_columns = ['temp', 'dwpt', 'rhum', 'wspd', 'wpgt', 'pres']

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

time_steps = 24

# reshape to [samples, time_steps, n_features]

X_train, y_train = create_dataset(train, train.energy_produced, time_steps)
X_test, y_test = create_dataset(test, test.energy_produced, time_steps)

print(X_train.shape, y_train.shape)

model = Sequential()

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

model.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=128))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(
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

forecast_data = pd.read_csv(
    "D:/Studia/Praca-magisterska/dane-z-PV/dane-do-badania/" + location + "-all-forecast-hourly.csv")

print(forecast_data.info())

forecast_data.pop("time")
forecast_data.pop("feels_like")
forecast_data.pop("uvi")
forecast_data.pop("clouds")
forecast_data.pop("visibility")
forecast_data.pop("wind_deg")
forecast_data.pop("pop")

print(forecast_data.info())

print(len(forecast_data))

f_columns = ['temp', 'pressure', 'humidity', 'dew_point', 'wind_speed', 'wind_gust']

f_transformer_2 = RobustScaler()

f_transformer_2 = f_transformer_2.fit(forecast_data[f_columns].to_numpy())

forecast_data.loc[:, f_columns] = f_transformer_2.transform(
  forecast_data[f_columns].to_numpy()
)

en_transformer_2 = RobustScaler()

en_transformer_2 = en_transformer_2.fit(forecast_data[['energy_produced']])

forecast_data['energy_produced'] = en_transformer_2.transform(forecast_data[['energy_produced']])

X_forecast, y_forecast = create_dataset(forecast_data, forecast_data.energy_produced, 1)

print(X_forecast.shape)

y_pred_forecast = model.predict(X_forecast)

y_pred_forecast_inv = en_transformer_2.inverse_transform(y_pred_forecast)
print(y_pred_forecast_inv)

plt.plot(y_pred_forecast_inv.flatten(), label='future prediction')
plt.legend()
plt.show()




















