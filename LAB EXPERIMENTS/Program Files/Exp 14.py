import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X = np.array([
    [1000, 2, 1, 10],
    [1200, 2, 1, 8],
    [1500, 3, 2, 5],
    [1800, 3, 2, 4],
    [2000, 4, 2, 3],
    [2200, 4, 3, 2],
    [2500, 4, 3, 2],
    [2800, 5, 3, 1],
    [3000, 5, 4, 1],
    [3500, 5, 4, 1],
    [1100, 2, 1, 9],
    [1400, 3, 2, 6],
    [1700, 3, 2, 5],
    [2100, 4, 3, 3],
    [2400, 4, 3, 2],
    [2700, 5, 3, 2],
    [3200, 5, 4, 1],
    [3600, 6, 4, 1],
    [1300, 2, 1, 7],
    [1900, 3, 2, 4]
])

y = np.array([
    35, 42, 55, 65, 75, 85, 92, 105, 120, 135,
    38, 50, 62, 80, 90, 102, 125, 145, 45, 70
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

sample = np.array([[2000, 3, 2, 3]])

prediction = model.predict(sample)

print("Predicted House Price:", prediction[0], "Lakhs")
