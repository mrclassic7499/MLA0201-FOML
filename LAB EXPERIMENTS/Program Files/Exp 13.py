import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = {
    "Year": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2017, 2018, 2019,
             2020, 2021, 2016, 2017, 2018, 2019, 2020, 2021, 2015, 2016],
    "Present_Price": [5.5, 6.0, 7.5, 8.0, 9.5, 10.0, 12.0, 7.0, 8.5, 9.0,
                      11.0, 13.0, 5.8, 6.8, 8.2, 9.8, 10.5, 12.5, 4.8, 5.5],
    "Kms_Driven": [30000, 25000, 20000, 18000, 15000, 12000, 8000, 22000,
                   17000, 14000, 10000, 7000, 28000, 23000, 19000, 16000,
                   11000, 6000, 35000, 29000],
    "Owner": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    "Selling_Price": [3.8, 4.5, 5.8, 6.5, 7.8, 8.5, 10.5, 5.2, 7.0, 7.5,
                      9.2, 11.0, 4.0, 5.0, 6.8, 8.0, 9.0, 11.5, 3.2, 4.0]
}

df = pd.DataFrame(data)

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

sample = pd.DataFrame({
    "Year": [2022],
    "Present_Price": [14.0],
    "Kms_Driven": [5000],
    "Owner": [0]
})

prediction = model.predict(sample)

print("Predicted Car Price:", prediction[0])
