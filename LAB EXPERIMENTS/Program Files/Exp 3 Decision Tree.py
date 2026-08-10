import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

le = LabelEncoder()

X = df.iloc[:, :-1].apply(le.fit_transform)
y = le.fit_transform(df['PlayTennis'])

model = DecisionTreeClassifier(criterion='entropy', random_state=0)
model.fit(X, y)

print("Decision Tree:\n")
print(export_text(model, feature_names=list(X.columns)))

sample = pd.DataFrame({
    'Outlook': ['Sunny'],
    'Temperature': ['Cool'],
    'Humidity': ['High'],
    'Wind': ['Strong']
})

sample_encoded = sample.copy()

for col in sample.columns:
    encoder = LabelEncoder()
    encoder.fit(df[col])
    sample_encoded[col] = encoder.transform(sample[col])

prediction = model.predict(sample_encoded)

print("\nPrediction:")
print("Play Tennis" if prediction[0] == 1 else "Don't Play Tennis")
