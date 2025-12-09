from flask import Flask, request, render_template, jsonify
import joblib
import os
import logging
from preprocess import clean_text

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# ---------- Config ----------
MODEL_PATH = "fake_job_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

# ---------- Load artifacts ----------
if not os.path.exists(MODEL_PATH):
    app.logger.error("Model file not found: %s", MODEL_PATH)
    raise FileNotFoundError(MODEL_PATH)

if not os.path.exists(VECTORIZER_PATH):
    app.logger.error("Vectorizer file not found: %s", VECTORIZER_PATH)
    raise FileNotFoundError(VECTORIZER_PATH)

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
app.logger.info("Model and vectorizer loaded.")

# ---------- Helpers ----------
def prepare_full_text(title, company, location, salary, description, requirements):
    # join fields with clear separators so preprocessing can treat them
    parts = [title or "", company or "", location or "", salary or "", description or "", requirements or ""]
    joined = " ||| ".join([p.strip() for p in parts if p is not None])
    return joined

def predict_text(text):
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])

    # Predicted class
    pred = None
    try:
        pred = model.predict(X)[0]
    except Exception as e:
        app.logger.error("model.predict failed: %s", e)
        raise

    # Probability (if available)
    prob = None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            # choose probability of predicted class
            # find index of predicted label if labels are not 0/1
            if hasattr(model, "classes_"):
                import numpy as np
                idx = int(np.where(model.classes_ == pred)[0][0])
                prob = float(proba[idx])
            else:
                prob = float(proba.max())
    except Exception as e:
        app.logger.warning("predict_proba not available or failed: %s", e)

    # Map pred to readable label
    # Common encoding: 0 -> Real, 1 -> Fake. Accept string labels too.
    label = None
    try:
        if str(pred).lower() in ("1", "fake", "true", "1.0"):
            label = "Fake Job"
        else:
            label = "Real Job"
    except Exception:
        label = "Fake Job" if pred == 1 else "Real Job"

    return {"label": label, "pred": str(pred), "prob": prob}

# ---------- Routes ----------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/job_details")
def job_details():
    return render_template("job_details.html")


@app.route("/predict", methods=["POST"])
def predict():
    # read form fields
    title = request.form.get("title", "")
    company = request.form.get("company", "")
    location = request.form.get("location", "")
    salary = request.form.get("salary", "")
    description = request.form.get("description", "")
    requirements = request.form.get("requirements", "")

    full_text = prepare_full_text(title, company, location, salary, description, requirements)
    app.logger.info("Predict: length=%d", len(full_text))

    if not full_text.strip():
        return render_template("result.html", result="No text provided", prob=None)

    try:
        res = predict_text(full_text)
    except Exception as e:
        app.logger.error("Prediction error: %s", e)
        return render_template("result.html", result="Error during prediction", prob=None)

    return render_template("result.html", result=res["label"], prob=res["prob"])


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Send JSON body"}), 400

    title = data.get("title", "")
    company = data.get("company", "")
    location = data.get("location", "")
    salary = data.get("salary", "")
    description = data.get("description", "")
    requirements = data.get("requirements", "")

    full_text = prepare_full_text(title, company, location, salary, description, requirements)
    if not full_text.strip():
        return jsonify({"error": "No text provided"}), 400

    try:
        res = predict_text(full_text)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    return jsonify(res)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
