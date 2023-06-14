import os
import shutil
from os import path
import zipfile
from zipfile import ZipFile
import filetype
import gzip
import csv

rootdir = 'D:/Studia/Praca-magisterska/dane-z-PV/dane-posortowane'

months = ["03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
days = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15",
        "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"]
hours = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
         "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"]
inverters = [
    "solar_test",
]

divided_data = []
new_divided_data = []

solar_test_data = []


#Wyrzucenie folderów z nazwą inną niż solar_test
for month in months:
    for day in days:
        for hour in hours:
            dir = rootdir + "/" + month + "/" + day + "/" + hour
            if path.exists(dir) == True:
                for file in os.listdir(dir):
                    d = os.path.join(dir, file)
                    if os.path.isdir(d):
                        if "solar_test" not in d:
                            print(d)
                            shutil.rmtree(d)

for month in months:
    for day in days:
        for hour in hours:
            dir = rootdir + "/" + month + "/" + day + \
                "/" + hour + "/solar_test/rawdata.zip"
            if path.exists(dir) == True:
                with gzip.open(dir, "r") as f:
                    bindata = f.read()
                    data = bindata.decode('utf-8')
                    divided_data = data.split("\n")

                for line in divided_data:
                    if divided_data.index(line) > 0:
                        info = line.split(",")
                        new_divided_data.append(info)

                new_divided_data = new_divided_data[: -1]

                if "59:5" in new_divided_data[0][0] or "28:5" in new_divided_data[0][0]:
                    new_divided_data.reverse()

                print(new_divided_data[0][0])

                for data in new_divided_data:
                    match data[1]:
                        case "solar_test":
                            solar_test_data.append(data)
                        case other:
                            print("No match found")

                csv_filename = rootdir + "/" + month + "/" + day + "/" + hour
               
                if len(solar_test_data) > 0:
                    f = open(csv_filename+'/solar_test/solar_test.csv', 'w')
                    writer = csv.writer(f)
                    f.write(divided_data[0]+"\n")
                    for line in solar_test_data:
                        writer.writerow(line)
                    f.close()

                divided_data = []
                new_divided_data = []

                solar_test_data = []

            