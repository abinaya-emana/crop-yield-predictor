from flask import Flask, request, jsonify, render_template
import joblib, json, pandas as pd, numpy as np, os

app = Flask(__name__)

# Load model and encoders
model = joblib.load("model.joblib")
label_encoders = joblib.load("label_encoders.joblib")
with open("features.json") as f:
    feature_names = json.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        sample = pd.DataFrame({
            "Area":       [data["area"]],
            "Item":       [data["crop"]],
            "Year":       [int(data["year"])],
            "average_rain_fall_mm_per_year": [float(data["rainfall"])],
            "pesticides_tonnes":             [float(data["pesticides"])],
            "avg_temp":                      [float(data["temp"])]
        })

        # Encode categorical columns
        for col in ["Area", "Item"]:
            if col in label_encoders:
                try:
                    sample[col] = label_encoders[col].transform(sample[col].astype(str))
                except ValueError:
                    return jsonify({"error": f"Unknown value for {col}. Check valid options."}), 400

        # Predict
        prediction = model.predict(sample)[0]

        return jsonify({
            "yield_hgha":     round(float(prediction), 2),
            "yield_tonnes":   round(float(prediction) / 10000, 2),
            "yield_kg_100sqm": round(float(prediction) / 100, 1)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/valid-options")
def valid_options():
    return jsonify({
        "areas": sorted(list(label_encoders["Area"].classes_)),
        "items": sorted(list(label_encoders["Item"].classes_))
    })

if __name__ == "__main__":
    print("Starting CropPredict server...")
    print("Open http://localhost:5000")
    app.run(debug=True, port=5000)
