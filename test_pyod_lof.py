import numpy as np
from sklearn.ensemble import IsolationForest
from pyod.models.lof import LOF
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
print("toto")
# Generate synthetic data
X, y = make_classification(n_samples=200, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=2,
                           n_clusters_per_class=1, weights=[0.9, 0.1],
                           flip_y=0, random_state=42)
print(X)
# Scale data
scaler = MinMaxScaler()
print(scaler)
X = scaler.fit_transform(X)

# Train LOF detector
clf = LOF(n_neighbors=20, contamination=0.1)
clf.fit(X)

# Get the anomaly scores
y_train_scores = clf.decision_scores_

# Predict raw anomaly scores of X_test
y_test_scores = clf.decision_function(X)

# Predict the labels (0: inlier, 1: outlier)
y_test_pred = clf.predict(X)

print("LOF model trained and predictions made successfully.")
