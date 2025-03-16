# Import necessary libraries
import pandas as pd
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.impute import KNNImputer, SimpleImputer
from imblearn.over_sampling import SMOTE
import os

# Load data
df = pd.read_csv('C:/ml_project/data/raw/heart_disease_uci.csv')
df = df.drop(['id', 'dataset'], axis=1)

print("Dataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# Separate features and target
X = df.drop(['num'], axis=1)
y = df['num']

# Define numerical and categorical columns
numeric_columns = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

# Create separate DataFrames
X_numeric = X[numeric_columns].copy()
X_categorical = X[categorical_columns].copy()

# Handle missing values for numerical features
numeric_imputer = KNNImputer(n_neighbors=5)
X_numeric_imputed = pd.DataFrame(
    numeric_imputer.fit_transform(X_numeric),
    columns=X_numeric.columns
)

# Handle missing values for categorical features
categorical_imputer = SimpleImputer(strategy='most_frequent')
X_categorical_imputed = pd.DataFrame(
    categorical_imputer.fit_transform(X_categorical),
    columns=X_categorical.columns
)

# Feature Engineering for numerical features
X_numeric_imputed['age_group'] = pd.cut(X_numeric_imputed['age'], 
                                       bins=[0, 40, 50, 60, 70, 100], 
                                       labels=[0, 1, 2, 3, 4], 
                                       include_lowest=True)
X_numeric_imputed['is_elderly'] = (X_numeric_imputed['age'] > 60).astype(int)
X_numeric_imputed['bp_category'] = pd.cut(X_numeric_imputed['trestbps'], 
                                         bins=[0, 120, 140, 160, 200], 
                                         labels=[0, 1, 2, 3], 
                                         include_lowest=True)
X_numeric_imputed['hr_reserve'] = 220 - X_numeric_imputed['age'] - X_numeric_imputed['thalch']
X_numeric_imputed['hr_category'] = pd.cut(X_numeric_imputed['thalch'], 
                                         bins=[0, 100, 120, 140, 160, 200], 
                                         labels=[0, 1, 2, 3, 4], 
                                         include_lowest=True)
X_numeric_imputed['chol_category'] = pd.cut(X_numeric_imputed['chol'], 
                                           bins=[0, 200, 240, 300, 600], 
                                           labels=[0, 1, 2, 3], 
                                           include_lowest=True)
X_numeric_imputed['age_bp'] = X_numeric_imputed['age'] * X_numeric_imputed['trestbps']
X_numeric_imputed['age_hr'] = X_numeric_imputed['age'] * X_numeric_imputed['thalch']
X_numeric_imputed['bp_hr'] = X_numeric_imputed['trestbps'] * X_numeric_imputed['thalch']

# Check for NaNs after feature engineering
print("\nNaNs after feature engineering:")
print(X_numeric_imputed.isnull().sum())

# Convert categorical bins to numeric and fill any remaining NaNs
for col in ['age_group', 'bp_category', 'hr_category', 'chol_category']:
    X_numeric_imputed[col] = X_numeric_imputed[col].cat.codes
    X_numeric_imputed[col] = X_numeric_imputed[col].replace(-1, 0)  # Replace invalid category codes

# Fill any remaining NaNs in numerical features
X_numeric_imputed = X_numeric_imputed.fillna(0)  # Conservative fallback

# Encode categorical variables
le = LabelEncoder()
for col in X_categorical_imputed.columns:
    X_categorical_imputed[col] = le.fit_transform(X_categorical_imputed[col].astype(str))

# Combine all features
X_processed = pd.concat([X_numeric_imputed, X_categorical_imputed], axis=1)

# Verify no NaNs remain
print("\nNaNs in final processed data:")
print(X_processed.isnull().sum())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Verify scaled data has no NaNs
print("\nNaNs in scaled training data:", np.isnan(X_train_scaled).sum())

# Apply SMOTE
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# Train models - ลดจำนวน estimators และปรับพารามิเตอร์เพื่อความเร็ว
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,  
        max_depth=8,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1  
    ),
    'HistGradient Boosting': HistGradientBoostingClassifier(
        max_iter=100,  
        learning_rate=0.1,
        max_depth=8,
        random_state=42
    )
}

# Train and evaluate models
results = {}
for name, model in models.items():
    print(f"\nกำลังฝึกโมเดล {name}...")
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Plot model comparison
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values())
plt.title('Model Performance Comparison')
plt.ylabel('Accuracy')
plt.xlabel('Models')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Fine-tune Random Forest
print("\nกำลังปรับแต่ง Random Forest...")
rf_model = RandomForestClassifier(
    random_state=42,
    class_weight='balanced',
    n_jobs=-1  # ใช้ทุก CPU cores
)

# ลดจำนวนค่าที่ต้องทดสอบ
rf_param_grid = {
    'max_depth': [5, 7],
    'n_estimators': [100, 200],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=rf_param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,  # ใช้ทุก CPU cores
    verbose=1  # แสดงความคืบหน้า
)

grid_search.fit(X_train_balanced, y_train_balanced)

print("\nBest Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)

# Use best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)
final_accuracy = accuracy_score(y_test, y_pred_best)
print("\nFinal Test Accuracy:", final_accuracy)
print("\nFinal Classification Report:")
print(classification_report(y_test, y_pred_best))

# บันทึกโมเดล
os.makedirs('models', exist_ok=True)
import joblib
joblib.dump(best_model, 'models/heart_disease_model.pkl')
print("\nSaved model'models/heart_disease_model.pkl' successfully")

