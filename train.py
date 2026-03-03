import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# --- 1. Load the Data ---
print("Loading data...")
df = pd.read_csv('data/students.csv')

# Separate Features (X) and Target (y)
X = df.drop('dropout', axis=1)
y = df['dropout']

# --- 2. Train-Test Split ---
# We use stratify=y to ensure the 80/20 ratio of safe/at-risk is maintained in both splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# --- 3. Data Scaling ---
# Machine learning models perform better when numerical features are on the same scale (0 to 1, or standard normal)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 4. Train Model 1: Logistic Regression ---
print("\n--- Training Logistic Regression ---")
# Using your insight: class_weight='balanced'
log_reg = LogisticRegression(class_weight='balanced', random_state=42)
log_reg.fit(X_train_scaled, y_train)

y_pred_lr = log_reg.predict(X_test_scaled)

print("Logistic Regression Results:")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_lr):.4f} <-- (Most important metric!)")
print(f"F1 Score:  {f1_score(y_test, y_pred_lr):.4f}")

# --- 5. Train Model 2: Random Forest ---
print("\n--- Training Random Forest ---")
# Random Forest handles non-linear relationships better
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train_scaled, y_train)

y_pred_rf = rf_model.predict(X_test_scaled)

print("Random Forest Results:")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_rf):.4f} <-- (Most important metric!)")
print(f"F1 Score:  {f1_score(y_test, y_pred_rf):.4f}")

print("\nRandom Forest Detailed Classification Report:")
print(classification_report(y_test, y_pred_rf))

# --- 6. Save the Model and Scaler ---
# Random Forest usually performs slightly better on complex data, so we will deploy that one.
# We must save BOTH the model and the scaler, because new data in the web app must be scaled exactly the same way!

print("\nSaving model and scaler to 'model/' directory...")
os.makedirs('model', exist_ok=True)

joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(rf_model, 'model/model.pkl')

print("✅ Training complete! 'scaler.pkl' and 'model.pkl' have been saved successfully.")