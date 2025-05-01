# ✅ Task 5: Decision Tree & Random Forest - FULLY FIXED for broken heart.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ✅ STEP 1: Manually read and parse broken heart.csv
with open("heart.csv", "r") as file:
    lines = file.read().strip().split("\n")

rows = [line.strip().split(",") for line in lines]
header = rows[0]
data = rows[1:]

# ✅ STEP 2: Convert to DataFrame
df = pd.DataFrame(data, columns=header)

# ✅ STEP 3: Convert all columns to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ✅ STEP 4: Drop rows with any NaN values (ensures clean data)
df.dropna(inplace=True)

# ✅ STEP 5: Show structure
print("✅ First 5 rows:\n", df.head())
print("\n✅ Dataset Info:")
print(df.info())
print("\n✅ Target Distribution:\n", df['target'].value_counts())

# ✅ STEP 6: Split features and labels
X = df.drop('target', axis=1)
y = df['target']

# ✅ STEP 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ STEP 8: Decision Tree Classifier
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)
y_pred_dt = dtree.predict(X_test)

print("\n🟩 Decision Tree Results")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))

# ✅ STEP 9: Visualize Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(dtree, filled=True, feature_names=X.columns, class_names=['No Disease', 'Disease'])
plt.title("Decision Tree")
plt.show()

# ✅ STEP 10: Random Forest Classifier
rforest = RandomForestClassifier(n_estimators=100, random_state=42)
rforest.fit(X_train, y_train)
y_pred_rf = rforest.predict(X_test)

print("\n🌲 Random Forest Results")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# ✅ STEP 11: Feature Importances
importances = rforest.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
sns.barplot(x=importances[sorted_idx], y=X.columns[sorted_idx])
plt.title("Feature Importances - Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# ✅ STEP 12: Cross-validation
cv_scores = cross_val_score(rforest, X, y, cv=5)
print("\n📊 Cross-Validation Scores:", cv_scores)
print("📈 Average Accuracy:", np.mean(cv_scores))
