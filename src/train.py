import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Load the dataset
data_path = os.path.join("data", "Iris.csv")
df = pd.read_csv(data_path)

# 2. Separate features (X) and target (y)
X = df.drop(columns=["Id", "Species"])
y = df["Species"]

# 3. Encode species names to numeric labels (0, 1, 2)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 4. Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 5. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train Models
# Model A: Logistic Regression
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train_scaled, y_train)
lr_preds = lr_model.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, lr_preds)

# Model B: K-Nearest Neighbors (K=3)
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_scaled, y_train)
knn_preds = knn_model.predict(X_test_scaled)
knn_acc = accuracy_score(y_test, knn_preds)

# 7. Print Results
print("--- MODEL EVALUATION ---")
print(f"Logistic Regression Accuracy: {lr_acc * 100:.2f}%")
print(f"KNN Accuracy: {knn_acc * 100:.2f}%")

print("\n--- DETAILED REPORT (Logistic Regression) ---")
print(classification_report(y_test, lr_preds, target_names=label_encoder.classes_))

# 8. Save the trained model, scaler, and label encoder for future predictions
os.makedirs("models", exist_ok=True)
joblib.dump(lr_model, os.path.join("models", "iris_lr_model.pkl"))
joblib.dump(scaler, os.path.join("models", "scaler.pkl"))
joblib.dump(label_encoder, os.path.join("models", "label_encoder.pkl"))
print("\nArtifacts saved in 'models/' directory.")