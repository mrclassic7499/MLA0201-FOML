from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
import numpy as np

iris = load_iris()

X = iris.data
y = iris.target

model = GaussianMixture(
    n_components=3,
    covariance_type='full',
    random_state=42
)

model.fit(X)

predictions = model.predict(X)

mapping = {}

for i in range(3):
    labels = y[predictions == i]
    if len(labels) > 0:
        mapping[i] = np.bincount(labels).argmax()

predicted_labels = np.array([mapping[p] for p in predictions])

accuracy = accuracy_score(y, predicted_labels)

print("Cluster Mapping:")
print(mapping)

print("\nAccuracy:", accuracy)

print("\nFirst 10 Predictions:")
for i in range(10):
    print(f"Actual: {y[i]}  Predicted: {predicted_labels[i]}")
