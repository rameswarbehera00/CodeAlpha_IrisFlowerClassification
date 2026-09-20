import os
import joblib
import numpy as np

# Load saved artifacts
model_path = os.path.join("models", "iris_lr_model.pkl")
scaler_path = os.path.join("models", "scaler.pkl")
encoder_path = os.path.join("models", "label_encoder.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
label_encoder = joblib.load(encoder_path)

def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    # Prepare input array
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Scale input using the fitted scaler
    features_scaled = scaler.transform(features)
    
    # Predict numeric class and probabilities
    pred_idx = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    # Convert numeric index back to species string
    species_name = label_encoder.inverse_transform([pred_idx])[0]
    confidence = probabilities[pred_idx] * 100
    
    return species_name, confidence

if __name__ == "__main__":
    print("--- IRIS FLOWER PREDICTOR ---")
    # Sample test inputs: [SepalLength, SepalWidth, PetalLength, PetalWidth]
    sample_samples = [
        (5.1, 3.5, 1.4, 0.2),  # Expected: Iris-setosa
        (6.0, 2.7, 5.1, 1.6),  # Expected: Iris-versicolor or virginica
        (6.9, 3.1, 5.4, 2.1),  # Expected: Iris-virginica
    ]
    
    for sl, sw, pl, pw in sample_samples:
        species, conf = predict_species(sl, sw, pl, pw)
        print(f"\nMeasurements: Sepal({sl}, {sw}), Petal({pl}, {pw})")
        print(f"Predicted Species: {species} ({conf:.2f}% confidence)")