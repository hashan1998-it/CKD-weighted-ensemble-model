import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Load the dataset
data = pd.read_csv("ckd.csv")

# Drop rows with all missing values and reset the index
data.dropna(how="all", inplace=True)
data.reset_index(drop=True, inplace=True)

# Encode categorical variables
label_encoders = {}
for col in data.select_dtypes(include="object").columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Fill missing values
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = data[col].fillna(data[col].mode()[0])  # Fill categorical with mode
    else:
        data[col] = data[col].fillna(data[col].mean())  # Fill numeric with mean

# Split data into features and labels
X = data.drop("classification", axis=1)  # Assuming 'classification' is the target column
y = data["classification"]

# Handle rare classes (remove classes with fewer than 2 samples)
class_counts = y.value_counts()
rare_classes = class_counts[class_counts < 2].index
X = X[~y.isin(rare_classes)]
y = y[~y.isin(rare_classes)]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Check class distribution
print("Class distribution in training set before SMOTE:")
print(y_train.value_counts())

# Apply SMOTE only if classes have more than 1 sample
min_class_count = min(y_train.value_counts())
k_neighbors = min(5, min_class_count - 1)  # Ensure k_neighbors is at least 1

if min_class_count > 1:
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_train, y_train = smote.fit_resample(X_train, y_train)
else:
    print("Skipping SMOTE as one or more classes have less than 2 samples.")

# Standardize numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Base classifiers
rf = RandomForestClassifier(random_state=42, max_depth=10, n_estimators=100)
gb = GradientBoostingClassifier(random_state=42, n_estimators=100)
xgb = XGBClassifier(random_state=42, eval_metric="logloss", max_depth=5, n_estimators=100)

# Fit models and get feature importances
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# Aggregate feature importances
feature_importances = np.mean([rf.feature_importances_, gb.feature_importances_, xgb.feature_importances_], axis=0)

# Adaptive feature weighting
weighted_X_train = X_train * feature_importances
weighted_X_test = X_test * feature_importances

# Train ensemble model on weighted features
ensemble = RandomForestClassifier(random_state=42, max_depth=10, n_estimators=100)
ensemble.fit(weighted_X_train, y_train)

# Predictions
y_pred = ensemble.predict(weighted_X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))

# Check if any class has fewer than 2 samples
if min(y_train.value_counts()) >= 2:
    # Cross-validation with adjusted splits
    cv_splits = min(5, min(y_train.value_counts()))
    cv_splits = max(2, cv_splits)  # Ensure at least 2 splits for cross-validation
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(ensemble, weighted_X_train, y_train, cv=cv)
    print("Cross-Validation Scores:", cv_scores)
    print("Mean CV Accuracy:", np.mean(cv_scores))
else:
    print("Skipping Cross-Validation due to insufficient class samples for splitting.")

# Confusion Matrix
ConfusionMatrixDisplay.from_estimator(ensemble, weighted_X_test, y_test)
