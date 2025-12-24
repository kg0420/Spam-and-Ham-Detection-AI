import os
import numpy as np
import regex as re
from flask import Flask, request, render_template
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# -------------------------
# Reduce TensorFlow noise
# -------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# -------------------------
# NLTK downloads (run once)
# -------------------------
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# -------------------------
# Flask app
# -------------------------
app = Flask(__name__)

# -------------------------
# Load trained model
# -------------------------
model = load_model("spam_model.keras", compile=False)

# -------------------------
# Parameters
# -------------------------
VOCAB_SIZE = 10000
MAX_LEN = 20

# -------------------------
# NLP tools
# -------------------------
lemma = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = re.sub("[^A-Za-z]", " ", text)
    text = text.lower()
    words = word_tokenize(text)
    words = [lemma.lemmatize(w) for w in words if w not in stop_words]
    encoded = one_hot(" ".join(words), VOCAB_SIZE)
    padded = pad_sequences([encoded], maxlen=MAX_LEN)
    return padded

# -------------------------
# Routes
# -------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        message = request.form.get("message", "")

        if not message.strip():
            return render_template("index.html", error="Message cannot be empty")

        processed = preprocess_text(message)
        prediction = model.predict(processed)[0]

        label = "Spam 🚫" if np.argmax(prediction) == 0 else "Ham ✅"
        confidence = round(float(np.max(prediction)) * 100, 2)

        return render_template(
            "index.html",
            message=message,
            prediction=label,
            confidence=confidence
        )

    except Exception as e:
        return render_template("index.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True, port=5002)
