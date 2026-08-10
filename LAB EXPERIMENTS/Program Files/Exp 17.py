import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)

n = 300

data = {
    "battery_power": np.random.randint(500, 2000, n),
    "ram": np.random.randint(512, 8000, n),
    "internal_memory": np.random.randint(8, 256, n),
    "mobile_wt": np.random.randint(80, 220, n),
    "px_height": np.random.randint(300, 2000, n),
    "px_width": np.random.randint(400, 2500, n),
    "camera": np.random.randint(2, 64, n),
    "four_g": np.random.randint(0, 2, n),
    "wifi": np.random.randint(0, 2, n),
    "bluetooth": np.random.randint(0, 2, n)
}

df = pd.DataFrame(data)

score = (
    df["ram"] * 0.01 +
    df["battery_power"] * 0.01 +
    df["internal_memory"] * 0.2 +
    df["px_height"] * 0.01 +
    df["px_width"] * 0.01 +
    df["camera"] * 0.5
)

df["price_range"] = pd.qcut(
    score,
    q=4,
    labels=[0, 1, 2, 3]
)

X = df.drop("price_range", axis=1)
y = df["price_range"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

sample = pd.DataFrame({
    "battery_power": [1800],
    "ram": [6000],
    "internal_memory": [128],
    "mobile_wt": [170],
    "px_height": [1200],
    "px_width": [2000],
    "camera": [48],
    "four_g": [1],
    "wifi": [1],
    "bluetooth": [1]
})

prediction = model.predict(sample)

print("Predicted Mobile Price Range:", prediction[0])
