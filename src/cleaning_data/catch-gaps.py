import csv
import os
import gzip
from os import path
import datetime

# location = "apa"
# file = open('C:/Studia/Praca-magisterska/src/merged/' + location + '-merged.csv')
# csvreader = csv.reader(file)
# rows = []
# for row in csvreader:
#     if row != []:
#         rows.append(row)

rows = []
with open('D:/Studia/Praca-magisterska/dane-z-PV/wieliczka/wieliczka-PV.txt') as f:
    rows = f.readlines()
    print(rows)

rows.pop(0)

catched_gaps = []

time = rows[0].split(",")
first_timestamp = time[0].split(" ")
day = first_timestamp[0].split("-")
hour = first_timestamp[1].split(":")

first_datetime = datetime.datetime(
    int(day[0]), int(day[1]), int(day[2]), int(hour[0]), int(hour[1]), int(hour[2]))
print(first_datetime)
print(type(first_datetime))

for row in rows:
    if rows.index(row) > 0:
        time=row.split(",")
        divided_timestamp = time[0].split(" ")
        divided_day = divided_timestamp[0].split("-")
        divided_hour = divided_timestamp[1].split(":")
        datetime_obj = datetime.datetime(
            int(divided_day[0]), int(divided_day[1]), int(divided_day[2]), int(divided_hour[0]), int(divided_hour[1]), int(divided_hour[2]))
        delta = datetime_obj - first_datetime
        if delta.days != 1:
            print(delta.days)
            catched_gaps.append(first_datetime)

        first_datetime = datetime_obj

print(len(catched_gaps))

for r in catched_gaps:
    print(r)

print(len(rows))