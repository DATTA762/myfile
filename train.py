import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
import pickle

# Load dataset
df = pd.read_csv("yourfile.csv")
df.columns = df.columns.str.strip()
print("Columns in dataset:", df.columns)

# Drop ID column
df.drop('id', axis=1, inplace=True)

# Replace special values in 'wc'
df['wc'] = df['wc'].replace(["\t6200", "\t8400"], [6200, 8400])

# Replace '?' or '\t?' with NaN
df.replace(['?', '\t?'], np.nan, inplace=True)

# Forward fill for all NaNs
df = df.fillna(method='ffill')

# Correct 'classification' labels
df['classification'] = df['classification'].replace("ckd\t", "ckd")

# Ensure 'wc' numeric
df['wc'] = pd.to_numeric(df['wc'], errors='coerce')

# Remove wc outliers using IQR
Q1 = df['wc'].quantile(0.25)
Q3 = df['wc'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df = df[(df['wc'] >= lower) & (df['wc'] <= upper)]

# Identify numeric and categorical columns
# numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('classification')
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

if 'classification' in numeric_cols:
    numeric_cols = numeric_cols.drop('classification')

categorical_cols = ['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane']

# Separate features and target
X = df.drop('classification', axis=1)
y = df['classification']

# Preprocessing for numeric columns
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical columns
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# Full pipeline with GaussianNB
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', GaussianNB())
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit pipeline
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print("Accuracy:", accuracy_score(y_test, y_pred)*100)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the pipeline for Flask deployment
with open('kidney_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("✅ Pipeline saved successfully!")
