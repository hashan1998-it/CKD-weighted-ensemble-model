import numpy as np
import pandas as pd
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("ckd.csv")

# Drop unnecessary columns if they exist
columns_to_drop = ["id"]
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1)

# Handle missing values
df = df.replace("?", np.nan)
df = df.fillna(df.median(numeric_only=True))
df = df.fillna(df.mode().iloc[0])

# Encode categorical columns
categorical_cols = df.select_dtypes(include=["object"]).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split features and target
X = df.drop("classification", axis=1)
y = df["classification"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models
models = {
    "Bagging": BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5, max_features=1.0, n_estimators=20),
    "Random Forest": RandomForestClassifier(max_depth=5, random_state=42),
    "Neural Network": MLPClassifier(solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(130,), random_state=42),
    "SVM": SVC(kernel="linear", probability=True),
    "AdaBoost": AdaBoostClassifier(algorithm="SAMME"),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
}

# Evaluate each model
results = {}
for name, model in models.items():
    print(f"Training and evaluating {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=1)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=1)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=1)

    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
    }

    print(f"{name} Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print("\n")

# Voting Classifier
print("Training and evaluating Voting Classifier...")
evs = VotingClassifier(
    estimators=[(name, model) for name, model in models.items()],
    voting="hard",
)
evs.fit(X_train, y_train)
y_pred = evs.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro", zero_division=1)
recall = recall_score(y_test, y_pred, average="macro", zero_division=1)
f1 = f1_score(y_test, y_pred, average="macro", zero_division=1)

results["Voting Classifier"] = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
}

print("Voting Classifier Results:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1 Score: {f1:.4f}")
print("\n")

# Voting Classifier
print("Training and evaluating Voting Classifier...")
evs = VotingClassifier(
    estimators=[(name, model) for name, model in models.items()],
    voting="hard",
)
evs.fit(X_train, y_train)
y_pred = evs.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")

results["Voting Classifier"] = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
}

print("Voting Classifier Results:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1 Score: {f1:.4f}")
print("\n")

# Confusion Matrix for Voting Classifier
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix for Voting Classifier")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Summarize results
print("Summary of Results:")
for name, metrics in results.items():
    print(f"{name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    print("\n")
