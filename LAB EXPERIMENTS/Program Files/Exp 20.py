import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = {
    "Month": np.arange(1, 25),
    "Advertising": [
        10, 12, 15, 14, 18, 20, 22, 25, 24, 28, 30, 32,
        35, 34, 38, 40, 42, 45, 44, 48, 50, 52, 55, 58
    ],
    "Sales": [
        120, 135, 150, 148, 175, 190, 205, 220, 215, 240, 255, 270,
        290, 285, 310, 325, 340, 360, 355, 380, 395, 410, 430, 450
    ]
}

df = pd.DataFrame(data)

X = df[["Month", "Advertising"]]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

future = pd.DataFrame({
    "Month": [25],
    "Advertising": [60]
})

prediction = model.predict(future)

print("Predicted Future Sales:", prediction[0])
