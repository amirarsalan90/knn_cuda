import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Parameters
n_samples = 50000
n_features = 1024
n_informative = 10  # Increased to handle 10 classes and clusters
n_classes = 10
test_size = 500

# Generate the dataset
X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative, 
                           n_classes=n_classes, random_state=42)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Save training and test data to CSV
pd.DataFrame(X_train).to_csv('train_features.csv', index=False, header=False)
pd.DataFrame(y_train).to_csv('train_labels.csv', index=False, header=False)
pd.DataFrame(X_test).to_csv('test_features.csv', index=False, header=False)
pd.DataFrame(y_test).to_csv('test_labels.csv', index=False, header=False)