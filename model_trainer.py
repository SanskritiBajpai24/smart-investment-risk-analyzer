import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Features: [Z-Score, Current Ratio, Debt-Equity, Rev Growth]
X = np.array([
    [4.5, 3.0, 0.1, 0.2],   # Healthy
    [3.8, 2.5, 0.2, 0.15],  # Healthy
    [1.1, 0.8, 2.5, -0.3],  # High Risk
    [0.5, 0.4, 4.0, -0.5],  # High Risk
    [2.1, 1.5, 0.8, 0.05],  # Moderate
])
y = np.array(['Healthy', 'Healthy', 'High Risk', 'High Risk', 'Moderate'])

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)
joblib.dump(model, 'risk_model.pkl')
print("✅ Brain (risk_model.pkl) created successfully!")