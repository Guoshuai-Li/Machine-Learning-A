import pandas as pd
import numpy as np

# Load training data (no header)
X_train = pd.read_csv('D:/1-LGS/KU course/Machine Learning A/week4/X_train.csv', header=None)
y_train = pd.read_csv('D:/1-LGS/KU course/Machine Learning A/week4/y_train.csv', header=None)

# Calculate class frequencies
class_counts = y_train.iloc[:,0].value_counts().sort_index()
total_samples = len(y_train)

print("Class frequencies:")
for class_label in sorted(class_counts.index):
    count = class_counts[class_label]
    frequency = count / total_samples
    print(f"Class {class_label}: {frequency:.4f}")
