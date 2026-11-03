import pandas as pd
import numpy as np
import joblib

# STEP 8: MACHINE LEARNING MODEL BUILDING

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("\n--- MACHINE LEARNING MODEL TRAINING ---")

# Load dataset
data = pd.read_csv(
    "dataset/bank-additional-full.csv",
    sep=";"
)

# Display first five records
print("FIRST FIVE RECORDS")
print(data.head())

# Dataset information
print("\nDATASET INFORMATION")
print(data.info())

# Dataset shape
print("\nDATASET SHAPE")
print(data.shape)

# Check missing values
print("\nMISSING VALUES")
print(data.isnull().sum())

# Target distribution
print("\nTARGET VARIABLE DISTRIBUTION")
print(data["y"].value_counts())

# Exploratory Data Analysis

print("\n--- EXPLORATORY DATA ANALYSIS ---")

# Dataset columns
print("\nColumn Names:")
print(data.columns.tolist())

# Statistical summary
print("\nStatistical Summary:")
print(data.describe())

# Duplicate records
print("\nDuplicate Records:")
print(data.duplicated().sum())

# Target variable percentage
print("\nTarget Variable Percentage:")
print(data["y"].value_counts(normalize=True) * 100)

import matplotlib.pyplot as plt
import seaborn as sns

# Target distribution graph
plt.figure(figsize=(7, 5))

sns.countplot(data=data, x="y")

plt.title("Bank Marketing Subscription Distribution")
plt.xlabel("Subscription")
plt.ylabel("Number of Customers")

plt.savefig("target_distribution.png")
plt.show()

# STEP 7: DATA PREPROCESSING

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

print("\n--- DATA PREPROCESSING ---")

# Remove duplicate records
data = data.drop_duplicates().copy()

print("Dataset shape after removing duplicates:")
print(data.shape)

# Separate input and target
# Remove duration to avoid data leakage
X = data.drop(columns=["y", "duration"])
y = data["y"].map({"no": 0, "yes": 1})

# Identify categorical and numerical columns
categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print("\nCategorical Features:")
print(categorical_features)

print("\nNumerical Features:")
print(numerical_features)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Fit and transform data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\nTraining Data Shape:", X_train_processed.shape)
print("Testing Data Shape:", X_test_processed.shape)

print("\nData Preprocessing Completed Successfully!")

# Initialize Models

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced"
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42, class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
}

# Store Accuracy Results
results = {}

# Train and Evaluate Models

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train_processed, y_train)

    predictions = model.predict(X_test_processed)

    accuracy = accuracy_score(y_test, predictions)

    results[name] = accuracy

    print(f"{name} Accuracy: {accuracy * 100:.2f}%")

print("\n--- MODEL TRAINING COMPLETED ---")

print("\nMODEL ACCURACY COMPARISON:")

for name, accuracy in results.items():
    print(f"{name}: {accuracy * 100:.2f}%")

# STEP 9: MODEL EVALUATION

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)

import matplotlib.pyplot as plt

# Select best model
best_model = models["Random Forest"]

# Predictions
y_pred = best_model.predict(X_test_processed)

# Classification Report
print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(
    y_test,
    y_pred,
    target_names=["No", "Yes"]
))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Save Confusion Matrix Graph
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No", "Yes"]
)

disp.plot(cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()

# ROC Curve
y_probability = best_model.predict_proba(X_test_processed)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_probability)

roc_auc = roc_auc_score(y_test, y_probability)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"Random Forest AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Random Forest ROC Curve")
plt.legend()

plt.savefig("roc_curve.png", dpi=300)
plt.close()

print("\nROC AUC Score:", round(roc_auc, 4))
print("\nModel Evaluation Completed Successfully!")

# STEP 10: PERFORMANCE VISUALIZATION

import matplotlib.pyplot as plt

model_names = list(results.keys())
accuracy_values = [
    value * 100 for value in results.values()
]

plt.figure(figsize=(9, 5))

plt.bar(
    model_names,
    accuracy_values
)

plt.title("Machine Learning Model Accuracy Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy (%)")

plt.ylim(0, 100)
plt.xticks(rotation=15)

for i, value in enumerate(accuracy_values):
    plt.text(i, value + 1, f"{value:.2f}%", ha="center")

plt.tight_layout()

plt.savefig("model_accuracy_comparison.png", dpi=300)

plt.close()

print("\nPerformance Visualization Completed!")
print("Model Accuracy Graph Saved Successfully!")

# Save trained model
joblib.dump(best_model, "bank_marketing_model.pkl")

# Save preprocessing object
joblib.dump(preprocessor, "preprocessor.pkl")

print("Model Saved Successfully!")