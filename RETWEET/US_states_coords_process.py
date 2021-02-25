import csv

with open('./Utils/US_States_Coordinates.csv') as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=",")
    coordinates = {}
    for row in csv_reader:
        str = row["latitude"] + "," + row["longitude"] + "," + "1000km"
        coordinates[row["name"]] = str
    
    for key, value in coordinates.items():
        print(key, value)