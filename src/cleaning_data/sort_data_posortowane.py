import os
import shutil
from os import path
import zipfile
from zipfile import ZipFile
import filetype
import gzip
import csv

rootdir = 'D:/Studia/Praca-magisterska/dane-z-PV/dane-posortowane-cz2'

years = ["2022", "2023"]
months = ["01", "02", "03", "04", "05",
          "06", "07", "08", "09", "10", "11", "12"]
days = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15",
        "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"]
hours = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
         "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"]
inverters = [
    "solar_test",
]

divided_data = []
new_divided_data = []

for year in years:
    for month in months:
        for day in days:
            for hour in hours:
                for inverter in inverters:
                    dir = rootdir + "/" + year + "/" + month + "/" + \
                        day + "/" + hour + "/" + inverter + "/rawdata.zip"
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

                        if "59:5" in new_divided_data[0][0] or "28:5" in new_divided_data[0][0] or "59:4" in new_divided_data[0][0] or "59:3" in new_divided_data[0][0]:
                            new_divided_data.reverse()

                        print(new_divided_data[0][0])

                        csv_filename = rootdir + "/" + year + "/" + month + "/" + \
                            day + "/" + hour + "/" + inverter + "/" + inverter + ".csv"

                        if len(new_divided_data) > 0:
                            f = open(csv_filename, 'w')
                            writer = csv.writer(f)
                            f.write(divided_data[0]+"\n")
                            for line in new_divided_data:
                                writer.writerow(line)
                            f.close()

                        divided_data = []
                        new_divided_data = []


