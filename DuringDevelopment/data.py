import json

# Sample data to be written to data.json
data = {
    "errorX": 0,
    "errorY": 0,
}

# Open the file in write mode and write the data
with open("data.json", "w") as file:
    json.dump(data, file, indent=4)  # 'indent=4' makes the file more readable
