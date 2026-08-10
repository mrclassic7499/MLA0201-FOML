import pandas as pd
import numpy as np

data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
]

columns = ['Sky', 'AirTemp', 'Humidity', 'Wind', 'Water', 'Forecast', 'EnjoySport']

df = pd.DataFrame(data, columns=columns)

concepts = np.array(df.iloc[:, :-1])
target = np.array(df.iloc[:, -1])

specific_h = concepts[0].copy()
general_h = [["?" for _ in range(len(specific_h))] for _ in range(len(specific_h))]

for i, h in enumerate(concepts):
    if target[i] == "Yes":
        for x in range(len(specific_h)):
            if h[x] != specific_h[x]:
                specific_h[x] = "?"
                general_h[x][x] = "?"
    else:
        for x in range(len(specific_h)):
            if h[x] != specific_h[x]:
                general_h[x][x] = specific_h[x]
            else:
                general_h[x][x] = "?"

general_h = [g for g in general_h if g != ["?"] * len(specific_h)]

print("Specific Hypothesis:")
print(specific_h)

print("\nGeneral Hypothesis:")
for g in general_h:
    print(g)
