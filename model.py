import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import VotingClassifier

# Load the data
data = pd.read_csv("ckd.csv")

# Remove 'id' column from features
X = data.drop(['classification', 'id'], axis=1)
y = data['classification']

# Convert target to binary (0 for 'notckd', 1 for 'ckd')
y = (y == 'ckd').astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create base models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
lr_model = LogisticRegression(random_state=42)

# Create voting classifier
ensemble_model = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('gb', gb_model),
        ('lr', lr_model)
    ],
    voting='soft'
)

# Create final pipeline
final_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', ensemble_model)
])

# Fit the model
final_model.fit(X_train, y_train)

# Make predictions
y_pred = final_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not CKD', 'CKD']))


# Function to predict for new data
def predict_ckd(new_data):
    # Ensure new_data has all required columns (except 'id' and 'classification')
    required_columns = set(X.columns)
    new_data_columns = set(new_data.columns)

    if not required_columns.issubset(new_data_columns):
        missing_columns = required_columns - new_data_columns
        raise ValueError(f"Missing columns in new data: {missing_columns}")

    # Select only the required columns in the correct order
    new_data = new_data[X.columns]

    prediction = final_model.predict(new_data)
    probability = final_model.predict_proba(new_data)
    return "CKD" if prediction[0] == 1 else "Not CKD", probability[0]


# Example usage
new_patient = pd.DataFrame({
    'age': [50],
    'bp': [80],
    'sg': [1.020],
    'al': [1],
    'su': [0],
    'rbc': ['normal'],
    'pc': ['normal'],
    'pcc': ['notpresent'],
    'ba': ['notpresent'],
    'bgr': [100],
    'bu': [30],
    'sc': [1.2],
    'sod': [135],
    'pot': [4.0],
    'hemo': [12.0],
    'pcv': [40],
    'wc': [8000],
    'rc': [4.5],
    'htn': ['no'],
    'dm': ['no'],
    'cad': ['no'],
    'appet': ['good'],
    'pe': ['no'],
    'ane': ['no']
})

try:
    result, prob = predict_ckd(new_patient)
    print(f"\nPrediction for new patient: {result}")
    print(f"Probability: Not CKD: {prob[0]:.2f}, CKD: {prob[1]:.2f}")
except ValueError as e:
    print(f"Error: {e}")