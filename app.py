"""
Spam Email Detection - Flask Web App
--------------------------------------
Run this file to start the website:
    python app.py

Then open your browser and go to:
    http://127.0.0.1:5000
"""

import os
import string
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# ---- Train model on startup ----
def clean_text(text):
    text = text.lower()
    text = "".join([ch for ch in text if ch not in string.punctuation])
    return text

def train_model():
    csv_path = os.path.join(os.path.dirname(__file__), "mail_data.csv")
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"Category": "label", "Message": "message"})
    df = df.dropna(subset=["label", "message"])
    df["clean_message"] = df["message"].apply(clean_text)
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(df["clean_message"])
    y = df["label_num"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    model = MultinomialNB()
    model.fit(X_train, y_train)
    print(f"✅ Model trained on {len(df)} messages.")
    return model, vectorizer

model, vectorizer = train_model()

# ---- Routes ----
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Please enter a message."}), 400

    cleaned = clean_text(message)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    confidence = round(max(probability) * 100, 2)
    label = "spam" if prediction == 1 else "ham"

    return jsonify({
        "label": label,
        "confidence": confidence,
        "message": message
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
    
