
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
import numpy as np
import random

# Load the dataset
data = pd.read_csv("Creditcard_data.csv")

# Balance the dataset using SMOTE
X = data.drop('Class', axis=1)
y = data['Class']
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Detect sample size (example: 10% of the balanced dataset size)
sample_size = int(0.1 * len(X_balanced))

# Create different sampling techniques
# 1. Simple Random Sampling
random_indices = random.sample(range(len(X_balanced)), sample_size)
X_random = X_balanced.iloc[random_indices]
y_random = y_balanced.iloc[random_indices]

# 2. Systematic Sampling
step = len(X_balanced) // sample_size
systematic_indices = list(range(0, len(X_balanced), step))[:sample_size]
X_systematic = X_balanced.iloc[systematic_indices]
y_systematic = y_balanced.iloc[systematic_indices]

# 3. Stratified Sampling
stratified_split = ShuffleSplit(n_splits=1, train_size=sample_size, random_state=42)
stratified_indices = next(stratified_split.split(X_balanced, y_balanced))[0]
X_stratified = X_balanced.iloc[stratified_indices]
y_stratified = y_balanced.iloc[stratified_indices]

# 4. Cross-Validation Sampling
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_indices = next(cv.split(X_balanced, y_balanced))[0]
X_cv = X_balanced.iloc[cv_indices]
y_cv = y_balanced.iloc[cv_indices]

# 5. Bootstrap Sampling
bootstrap_indices = np.random.choice(range(len(X_balanced)), size=sample_size, replace=True)
X_bootstrap = X_balanced.iloc[bootstrap_indices]
y_bootstrap = y_balanced.iloc[bootstrap_indices]

# Combine samples
samples = [
    (X_random, y_random),
    (X_systematic, y_systematic),
    (X_stratified, y_stratified),
    (X_cv, y_cv),
    (X_bootstrap, y_bootstrap)
]

# Define models
models = {
    'M1': LogisticRegression(max_iter=5000),
    'M2': RandomForestClassifier(),
    'M3': SVC(),
    'M4': KNeighborsClassifier(),
    'M5': DecisionTreeClassifier()
}

# Evaluate models with sampling techniques
results = pd.DataFrame(columns=["Model", "Simple Random", "Systematic", "Stratified", "Cross-Validation", "Bootstrap"])
rows = []
for model_name, model in models.items():
    row = {"Model": model_name}
    for i, (X_sample, y_sample) in enumerate(samples):
        X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.3, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        row[results.columns[i + 1]] = accuracy
    rows.append(row)

# Convert rows into a DataFrame
results = pd.DataFrame(rows)


# Determine which sampling technique gives the highest accuracy for each model
results["Best Sampling Technique"] = results.iloc[:, 1:].idxmax(axis=1)
results["Best Sampling and Model"] = results.apply(lambda row: f"{row['Best Sampling Technique']} ({row['Model']})", axis=1)

# Print results in the required format
print(results)

# Save results to CSV
results.to_csv("Output.csv", index=False)
print("Results saved to output.csv")
