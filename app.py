from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import urllib.request

app = Flask(__name__)

# =========================
# 📁 PATH SETUP
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "fertilizer_rf_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "fertilizer_label_encoder.pkl")

# 🔥 GitHub RAW URLs
MODEL_URL = "https://raw.githubusercontent.com/ayush-eye/Fertilizer-Recommendation/main/fertilizer_rf_model.pkl"
ENCODER_URL = "https://raw.githubusercontent.com/ayush-eye/Fertilizer-Recommendation/main/fertilizer_label_encoder.pkl"

# =========================
# ⬇️ DOWNLOAD FILES
# =========================

def download_file(url, path):
    print(f"Downloading {path}...")
    urllib.request.urlretrieve(url, path)

# Download model
if not os.path.exists(MODEL_PATH):
    download_file(MODEL_URL, MODEL_PATH)

# Download encoder
if not os.path.exists(ENCODER_PATH):
    download_file(ENCODER_URL, ENCODER_PATH)

# Verify
if not os.path.exists(MODEL_PATH):
    raise Exception("Model download failed ❌")

if not os.path.exists(ENCODER_PATH):
    raise Exception("Encoder download failed ❌")

print("Files ready ✅")

# =========================
# 📦 LOAD MODEL
# =========================

try:
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    print("Model and Encoder Loaded Successfully ✅")
except Exception as e:
    print("Loading Error:", str(e))
    raise e

# =========================
# 🌐 ROUTES
# =========================

@app.route('/')
def home():
    return "Fertilizer Recommendation API is running 🌱"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Required fields
        required_keys = [
            'Temperature', 'Humidity', 'Moisture',
            'Soil_Type', 'Crop_Type',
            'Nitrogen', 'Potassium', 'Phosphorous'
        ]

        for key in required_keys:
            if key not in data:
                return jsonify({"error": f"Missing key: {key}"})

        # ⚠️ MUST match training order
        features = [
            data['Temperature'],
            data['Humidity'],
            data['Moisture'],
            data['Soil_Type'],
            data['Crop_Type'],
            data['Nitrogen'],
            data['Potassium'],
            data['Phosphorous']
        ]

        final_input = np.array([features])

        prediction = model.predict(final_input)
        fertilizer_name = encoder.inverse_transform(prediction)

        return jsonify({
            "recommended_fertilizer": str(fertilizer_name[0])
        })

    except Exception as e:
        print("Prediction Error:", str(e))
        return jsonify({"error": str(e)})

# =========================
# ▶️ RUN APP
# =========================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)