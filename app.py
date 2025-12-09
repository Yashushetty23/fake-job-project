from flask import Flask, request, jsonify
import joblib
import numpy as np
from utils.preprocess import clean_text
from flask_cors import CORS

app = Flask(__name__)
CORS(app)   # 👈 important: allows frontend to call backend

# Load model + vectorizer
MODEL_PATH = "model/fake_job_model.pkl"
VECTORIZER_PATH = "model/tfidf_vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        description = data.get("description", "")
        if not description:
            return jsonify({"error": "Description is required"}), 400

        # clean text
        cleaned_text = clean_text(description)

        # vectorize
        vector_input = vectorizer.transform([cleaned_text])

        # prediction
        prediction = model.predict(vector_input)[0]
        probability = model.predict_proba(vector_input)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "probability": float(probability),
            "label": "fake" if prediction == 1 else "real"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return {"message": "Fake Job Detector API working"}


if __name__ == "__main__":
    app.run(debug=True)
