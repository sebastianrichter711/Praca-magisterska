import csv
import os
import gzip
from os import path
import datetime

location="wieliczka"
file = open('D:/Studia/Praca-magisterska/dane-pogodowe/Wieliczka-history-weather.csv')
csvreader = csv.reader(file)
rows = []
for row in csvreader:
    if "datetime" not in row:
        rows.append(row)

print(len(rows))
line = rows[1][0].split(",")
print(line)

PV_rows = []
with open('D:/Studia/Praca-magisterska/dane-z-PV/wieliczka/wieliczka-PV.txt') as f:
    PV_rows = f.readlines()

PV_rows.pop(0)

splitted = PV_rows[0].split(",")
print(splitted)

f = open('C:/Studia/Praca-magisterska/src/data-all/' + location + '-all-2.csv', 'w')
writer = csv.writer(f)
writer.writerow(["name","datetime","tempmax","tempmin","temp","feelslikemax","feelslikemin","feelslike","dew","humidity","precip",
                "precipprob","precipcover","preciptype","snow","snowdepth","windgust","windspeed","winddir","sealevelpressure",
                "cloudcover","visibility","solarradiation","solarenergy","uvindex","severerisk","energy_produced"])
for i in range(0,len(rows)):
    record = rows[i][0].split(",")
    splitted_PV_data = PV_rows[i].split(",")
    PV_value = splitted_PV_data[1].split("\r")
    PV_value_2 = PV_value[0].split("\n")
    print(PV_value_2)
    line = [record[0],record[1],record[2],record[3],record[4],record[5],record[6],record[7],record[8],record[9],record[10],
                record[11],record[12],record[13],record[14],record[15],record[16],record[17],record[18],record[19],
                record[20],record[21],record[22],record[23],record[24],record[25],PV_value_2[0]]
    writer.writerow(line)
f.close()




