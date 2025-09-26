import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# Load data (no header)
X_train = pd.read_csv('D:/1-LGS/KU course/Machine Learning A/week4/X_train.csv', header=None)
y_train = pd.read_csv('D:/1-LGS/KU course/Machine Learning A/week4/y_train.csv', header=None)
X_test = pd.read_csv('D:/1-LGS/KU course/Machine Learning A/week4/X_test.csv', header=None)
y_test = pd.read_csv('D:/1-LGS/KU course/Machine Learning A/week4/y_test.csv', header=None)

# Convert y to 1D arrays
y_train = y_train.iloc[:, 0].values
y_test = y_test.iloc[:, 0].values


# Logistic regression with L2 regularization (default)
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

# Predictions
lr_train_pred = lr_model.predict(X_train)
lr_test_pred = lr_model.predict(X_test)

# Calculate errors
lr_train_error = 1 - accuracy_score(y_train, lr_train_pred)
lr_test_error = 1 - accuracy_score(y_test, lr_test_pred)

print(f"Training error: {lr_train_error:.4f}")
print(f"Test error: {lr_test_error:.4f}")


# Random forest with different number of trees
n_trees_list = [50, 100, 200]

for n_trees in n_trees_list:
    print(f"\nRandom Forest with {n_trees} trees:")
    
    rf_model = RandomForestClassifier(n_estimators=n_trees, random_state=42, oob_score=True)
    rf_model.fit(X_train, y_train)
    
    # Predictions
    rf_train_pred = rf_model.predict(X_train)
    rf_test_pred = rf_model.predict(X_test)
    
    # Calculate errors
    rf_train_error = 1 - accuracy_score(y_train, rf_train_pred)
    rf_test_error = 1 - accuracy_score(y_test, rf_test_pred)
    rf_oob_error = 1 - rf_model.oob_score_
    
    print(f"Training error: {rf_train_error:.4f}")
    print(f"Test error: {rf_test_error:.4f}")
    print(f"OOB error: {rf_oob_error:.4f}")


# Find best k using cross-validation
k_values = [1, 3, 5, 7, 9, 11, 15, 20]
cv_scores = []

print("Cross-validation for k selection:")
for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_model, X_train, y_train, cv=5, scoring='accuracy')
    mean_score = scores.mean()
    cv_scores.append(mean_score)
    print(f"k={k}: CV accuracy = {mean_score:.4f}")

# Select best k
best_k = k_values[np.argmax(cv_scores)]
print(f"\nBest k selected: {best_k}")

# Train final model with best k
final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train, y_train)

# Final predictions
knn_train_pred = final_knn.predict(X_train)
knn_test_pred = final_knn.predict(X_test)

# Calculate final errors
knn_train_error = 1 - accuracy_score(y_train, knn_train_pred)
knn_test_error = 1 - accuracy_score(y_test, knn_test_pred)

print(f"\nFinal results with k={best_k}:")
print(f"Training error: {knn_train_error:.4f}")
print(f"Test error: {knn_test_error:.4f}")
