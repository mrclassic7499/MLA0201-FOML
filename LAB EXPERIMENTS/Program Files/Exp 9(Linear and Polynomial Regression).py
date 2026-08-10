import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

np.random.seed(42)

X = np.arange(1, 21).reshape(-1, 1)
y = X.flatten()**2 + np.random.randint(-20, 20, size=20)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)

poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
poly_pred = poly_model.predict(X_test_poly)

print("Linear Regression R2 Score:", r2_score(y_test, linear_pred))
print("Polynomial Regression R2 Score:", r2_score(y_test, poly_pred))

X_plot = np.linspace(1, 20, 100).reshape(-1, 1)

plt.scatter(X, y, color="blue", label="Data")

plt.plot(X_plot, linear_model.predict(X_plot), color="red", label="Linear Regression")

plt.plot(
    X_plot,
    poly_model.predict(poly.transform(X_plot)),
    color="green",
    label="Polynomial Regression"
)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show()
