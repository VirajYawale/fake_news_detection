# from flask import Flask, request, render_template
# import joblib

# app = Flask(__name__)
# model = joblib.load("saved_model.pkl")
# vectorizer = joblib.load("vectorizer.pkl")  # assuming TF-IDF or CountVectorizer is saved

# @app.route("/", methods=["GET", "POST"])
# def index():
#     prediction = None
#     label = None
#     if request.method == "POST":
#         user_input = request.form.get("news_text")
#         if user_input:
#             transformed_input = vectorizer.transform([user_input])
#             result = model.predict(transformed_input)[0]
#             prediction = "🟢 Real News" if result == 1 else "🔴 Fake News"
#             label = "real" if result == 1 else "fake"
#     return render_template("index.html", prediction=prediction, label=label)

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------
# CONFIG
# -----------------
MODEL_PATH = "model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
MAXLEN = 100  # same maxlen used in training

# -----------------
# LOAD MODEL + TOKENIZER
# -----------------
model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# -----------------
# FLASK APP
# -----------------
app = Flask(__name__)
CORS(app)  # allow browser extension requests

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text.strip():
            return jsonify({"error": "No text provided"}), 400

        # Tokenize and pad
        seq = tokenizer.texts_to_sequences([text])
        seq_padded = pad_sequences(seq, maxlen=MAXLEN)

        # Predict
        prediction = model.predict(seq_padded)
        confidence = float(prediction[0][0])  # assuming single output neuron
        label = "FAKE" if confidence > 0.5 else "REAL"

        return jsonify({
            "label": label,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "Fake News Detection API is running."

if __name__ == "__main__":
    app.run(debug=True)
