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
import csv

plt.style.use('fivethirtyeight')

data = []

for i in range(0,107):
    with open("D:/Studia/Praca-magisterska/dane-z-PV/wieliczka/statystyki-dobowe(" + str(i) + ").csv", 'r') as file:
        csvreader = csv.reader(file, delimiter=';')
        for row in csvreader:
            if "Produkcja PV" not in row:
                row[1]=row[1].replace(',', '.')
                print(row[0] + " " + row[1])
                record = [row[0],row[1]]
                data.append(record)

print(len(data))

with open("D:/Studia/Praca-magisterska/dane-z-PV/wieliczka/wieliczka-PV.txt", 'w') as f:
    f.write("Data,Produkcja PV"+ "\n")
    for line in data:
        f.write(line[0]+","+line[1])
        f.write('\n')