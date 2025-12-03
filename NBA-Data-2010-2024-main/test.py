import csv

with open("regular_season_totals_2010_2024.csv", "r") as f:
    reader = csv.reader(f)
    first_column = [row[0] for row in reader]

for i in range(0, len(first_column), 10):
    print("\t".join(first_column[i : i + 10]))
