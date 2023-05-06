import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Activation
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

plt.style.use('fivethirtyeight')

location = "apa"
data = pd.read_csv(
    "D:/Studia/Praca-magisterska/dane-z-PV/dane-do-badania/" + location + "-all-forecast-hourly.csv")

print(data.info())

data.pop("time")
data.pop("feels_like")
data.pop("uvi")
data.pop("clouds")
data.pop("visibility")
data.pop("wind_deg")
data.pop("pop")

print(data.info())

print(len(data))


