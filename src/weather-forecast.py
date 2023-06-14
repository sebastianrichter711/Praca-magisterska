import requests
import os
from datetime import datetime
import tzlocal
import csv

API_KEY = ""

location = "wieliczka"

lat = 49.98738
lon = 20.06473
exclude = "current,minutely,hourly,alerts"
#exclude = "current,minutely,daily,alerts"

url=f"https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&exclude={exclude}&units=metric&appid={API_KEY}"
req = requests.get(url)
data = req.json()

for record in data["daily"]:
    ts=int(record["dt"])
    local_timezone = tzlocal.get_localzone() 
    local_time = datetime.fromtimestamp(ts, local_timezone)
    local_time_formatted = local_time.strftime('%Y-%m-%d %H:%M:%S')
    record["dt"] = local_time_formatted
    print(record["dt"])
    print(local_time_formatted)


# f = open('C:/Studia/Praca-magisterska/src/data-weather/' + location + '-forecast-hourly.csv', 'w')
# writer = csv.writer(f)
# writer.writerow(["time", "temp", "feels_like", "pressure", "humidity",
#                 "dew_point", "uvi", "clouds", "visibility", "wind_speed", "wind_deg", "wind_gust", "pop"])
# for record in data["hourly"]:
#     line = [record["dt"], record["temp"],record["feels_like"],record["pressure"],record["humidity"],
#                     record["dew_point"],record["uvi"],record["clouds"],record["visibility"],
#                     record["wind_speed"],record["wind_deg"],record["wind_gust"],record["pop"]]
#     writer.writerow(line)
# f.close()

f = open('C:/Studia/Praca-magisterska/src/data-weather/' + location + '-forecast-daily.csv', 'w')
writer = csv.writer(f)
writer.writerow(["time", "temp", "feels_like", "pressure", "humidity",
                "dew_point", "wind_speed", "wind_deg", "wind_gust", "clouds", "pop", "rain", "uvi"])
for record in data["daily"]:
    if "rain" not in record:
        line = [record["dt"], record["temp"]["day"],record["feels_like"]["day"],record["pressure"],record["humidity"],
                    record["dew_point"],record["wind_speed"],record["wind_deg"],record["wind_gust"],record["clouds"],
                    record["pop"],0.0,record["uvi"]]
    else:
        line = [record["dt"], record["temp"]["day"],record["feels_like"]["day"],record["pressure"],record["humidity"],
                    record["dew_point"],record["wind_speed"],record["wind_deg"],record["wind_gust"],record["clouds"],
                    record["pop"],record["rain"],record["uvi"]]
    writer.writerow(line)
f.close()
    



