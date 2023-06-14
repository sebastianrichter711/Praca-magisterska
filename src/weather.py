from datetime import datetime
from meteostat import Point, Hourly, Daily

start = datetime(2022, 2, 28, 23, 00, 00)
end = datetime(2023, 3, 13, 7, 59, 59)

# APA
latitude = 50.3287
longitude = 18.7069

location = Point(latitude, longitude, None)

data = Hourly(location, start, end)
data = data.fetch()

csv_filename = "data-weather/apa.csv"
data.to_csv(csv_filename)

# Wieliczka
latitude = 49.98738
longitude = 20.06473

location = Point(latitude, longitude, None)

start = datetime(2014, 6, 27, 00, 00, 00)
end = datetime(2023, 4, 30, 00,00,00)
data = Daily(location, start, end)
data = data.fetch()

csv_filename = "data-weather/wieliczka.csv"
data.to_csv(csv_filename)