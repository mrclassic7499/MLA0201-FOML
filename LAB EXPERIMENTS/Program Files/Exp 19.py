import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

np.random.seed(42)

n = 300

data = {
    "Age": np.random.randint(21, 65, n),
    "Income": np.random.randint(20000, 150000, n),
    "LoanAmount": np.random.randint(5000, 100000, n),
    "CreditScore": np.random.randint(300, 850, n),
    "LoanTerm": np.random.choice([12, 24, 36, 48, 60], n),
    "Employment": np.random.choice(["Salaried", "Self-Employed"], n),
    "Education": np.random.choice(["Graduate", "Not Graduate"], n)
}

df = pd.DataFrame(data)

df["Loan_Status"] = np.where(
    (df["CreditScore"] >= 650) &
    (df["Income"] >= 30000) &
    (df["LoanAmount"] <= df["Income"] * 0.8),
    "Approved",
    "Rejected"
)

encoder = LabelEncoder()

df["Employment"] = encoder.fit_transform(df["Employment"])
df["Education"] = encoder.fit_transform(df["Education"])
df["Loan_Status"] = encoder.fit_transform(df["Loan_Status"])

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = GaussianNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

sample = pd.DataFrame({
    "Age": [30],
    "Income": [60000],
    "LoanAmount": [20000],
    "CreditScore": [750],
    "LoanTerm": [36],
    "Employment": [1],
    "Education": [0]
})

prediction = model.predict(sample)

if prediction[0] == 0:
    print("Loan Prediction: Approved")
else:
    print("Loan Prediction: Rejected")
