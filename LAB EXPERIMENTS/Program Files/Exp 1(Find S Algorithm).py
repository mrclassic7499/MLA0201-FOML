import pandas as pd

data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
]

columns = ['Sky', 'AirTemp', 'Humidity', 'Wind', 'Water', 'Forecast', 'EnjoySport']

df = pd.DataFrame(data, columns=columns)

concepts = df.iloc[:, :-1].values
target = df.iloc[:, -1].values

hypothesis = None

for i in range(len(target)):
    if target[i] == "Yes":
        if hypothesis is None:
            hypothesis = concepts[i].copy()
        else:
            for j in range(len(hypothesis)):
                if hypothesis[j] != concepts[i][j]:
                    hypothesis[j] = "?"

print("Most Specific Hypothesis:")
print(hypothesis)
