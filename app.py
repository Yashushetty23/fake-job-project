from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from preprocess import clean_text   # your preprocessing function

app = Flask(__name__)

# Load model + vectorizer
model = joblib.load("fake_job_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get values from form
    title = request.form.get("title", "")
    company = request.form.get("company", "")
    description = request.form.get("description", "")
    requirements = request.form.get("requirements", "")

    # Combine all text
    full_text = f"{title} {company} {description} {requirements}"

    # Clean text using your preprocess.py
    cleaned = clean_text(full_text)

    # Vectorize
    transformed = vectorizer.transform([cleaned])

    # Predict
    prediction = model.predict(transformed)[0]   # 0 = real, 1 = fake

    result = "Fake Job" if prediction == 1 else "Real Job"

    return render_template("result.html", result=result)
    

if __name__ == "__main__":
    app.run(debug=True)
