import utils.clients
import utils.http_session
import utils.transport
import utils.session
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load model
model = joblib.load("keras_model_wrapped.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return "🍷 Wine Quality Prediction API (Keras Model in .pkl)"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    # Extract values in correct order
    features = [
        data["fixed_acidity"],
        data["volatile_acidity"],
        data["citric_acid"],
        data["residual_sugar"],
        data["chlorides"],
        data["free_sulfur_dioxide"],
        data["total_sulfur_dioxide"],
        data["density"],
        data["pH"],
        data["sulphates"],
        data["alcohol"],
        data["type_white"]
    ]

    input_data = np.array([features])
    prediction = model.predict(input_data)[0]

    return jsonify({
        "prediction": int(prediction),
        "quality": "Good" if prediction == 1 else "Not Good"
    })

if __name__ == "__main__":
    app.run(debug=True)
