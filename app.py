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
import onnxruntime as ort

# -------------------------
# Reduce logs
# -------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# -------------------------
# NLTK setup (no downloads)
# -------------------------
nltk.data.path.append("/opt/render/nltk_data")

# -------------------------
# Flask app
# -------------------------
app = Flask(__name__)

# -------------------------
# Load ONNX model
# -------------------------
session = ort.InferenceSession(
    "spam_model.onnx",
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

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
    return padded.astype(np.int64)

# -------------------------
# Routes
# -------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form.get("message", "").strip()

    if not message:
        return render_template("index.html", error="Message cannot be empty")

    processed = preprocess_text(message)

    prediction = session.run(
        [output_name],
        {input_name: processed}
    )[0][0]

    class_id = int(np.argmax(prediction))
    confidence = round(float(prediction[class_id]) * 100, 2)

    label = "Spam 🚫" if class_id == 0 else "Ham ✅"

    return render_template(
        "index.html",
        message=message,
        prediction=label,
        confidence=confidence
    )

# -------------------------
# Run locally
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
