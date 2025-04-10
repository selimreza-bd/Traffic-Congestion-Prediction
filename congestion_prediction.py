# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns

# Simulate larger, noisier data
np.random.seed(42)
n_samples = 500
data = {
    'Vehicles': np.random.randint(20, 200, n_samples),
    'Speed': np.random.uniform(20, 90, n_samples),
    'Hour': np.random.randint(0, 24, n_samples),
    'Weather': np.random.randint(0, 2, n_samples),  # 0 = clear, 1 = rainy
    'Congested': np.zeros(n_samples)
}
df = pd.DataFrame(data)
df['Congested'] = np.where((df['Vehicles'] > 100) | (df['Speed'] < 40) | (df['Weather'] == 1), 1, 0)
noise_idx = np.random.choice(n_samples, int(0.15 * n_samples), replace=False)
df.loc[noise_idx, 'Congested'] = 1 - df.loc[noise_idx, 'Congested']

# Features and target
X = df[['Vehicles', 'Speed', 'Hour', 'Weather']].values
y = df['Congested'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Random Forest ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Evaluate Random Forest
print("Random Forest Performance:")
print(f"Accuracy: {accuracy_score(y_test, rf_pred):.2f}")
print(classification_report(y_test, rf_pred))

# Confusion matrix
rf_cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Feature importance
importances = rf_model.feature_importances_
features = ['Vehicles', 'Speed', 'Hour', 'Weather']
plt.bar(features, importances)
plt.title('Feature Importance (Random Forest)')
plt.ylabel('Importance')
plt.show()

# --- Neural Network ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nn_model = Sequential([
    Dense(16, activation='relu', input_shape=(4,)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = nn_model.fit(X_train_scaled, y_train, epochs=50, batch_size=4, validation_split=0.2, verbose=0)

# Evaluate Neural Network
nn_loss, nn_accuracy = nn_model.evaluate(X_test_scaled, y_test)
print("\nNeural Network Performance:")
print(f"Accuracy: {nn_accuracy:.2f}")
print(f"Loss: {nn_loss:.2f}")

# Confusion matrix
nn_pred_binary = (nn_model.predict(X_test_scaled) > 0.5).astype(int)
nn_cm = confusion_matrix(y_test, nn_pred_binary)
sns.heatmap(nn_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Neural Network Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plot accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Neural Network Accuracy Over Time')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Predict scenarios
scenarios = pd.DataFrame([
    [50, 80, 8, 0],   # Morning, low traffic, clear
    [150, 30, 17, 1], # Evening rush, rainy
    [80, 60, 2, 0]    # Late night, clear
], columns=['Vehicles', 'Speed', 'Hour', 'Weather'])

print("\nScenario Predictions:")
for i, scenario in scenarios.iterrows():
    scaled = scaler.transform([scenario])
    pred = nn_model.predict(scaled)[0][0]
    print(f"Scenario: {scenario.values.tolist()} -> Prob: {pred:.2f} ({'Congested' if pred > 0.5 else 'Not Congested'})")
