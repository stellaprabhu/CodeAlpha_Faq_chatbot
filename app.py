"""
FAQ Chatbot
-----------
Loads a set of FAQs, preprocesses the questions with NLTK (tokenize,
lowercase, remove stopwords, lemmatize), builds a TF-IDF matrix, and
answers user queries by finding the FAQ question with the highest
cosine similarity to what the user typed.

Run with:  python app.py
Then open: http://127.0.0.1:5000
"""

import json
import string

import nltk
from flask import Flask, render_template, request, jsonify
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Make sure the required NLTK data is available (only downloads once).
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

app = Flask(__name__)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# --- Similarity below this threshold means "I don't understand" ---
CONFIDENCE_THRESHOLD = 0.25


def preprocess(text: str) -> str:
    """Lowercase, tokenize, strip punctuation/stopwords, and lemmatize."""
    text = text.lower()
    tokens = word_tokenize(text)
    cleaned = [
        LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in string.punctuation and tok not in STOP_WORDS and tok.isalpha()
    ]
    return " ".join(cleaned)


def load_faqs(path="faqs.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Load FAQ data once at startup
FAQS = load_faqs()
FAQ_QUESTIONS = [item["question"] for item in FAQS]
# Combine each question with its optional "keywords" field (extra synonyms)
# so paraphrased user questions still match, not just exact word overlap.
PROCESSED_QUESTIONS = [
    preprocess(item["question"] + " " + item.get("keywords", ""))
    for item in FAQS
]

# Fit a single TF-IDF vectorizer (with unigrams + bigrams) on all FAQ text
VECTORIZER = TfidfVectorizer(ngram_range=(1, 2))
FAQ_MATRIX = VECTORIZER.fit_transform(PROCESSED_QUESTIONS)


def get_best_match(user_message: str):
    """Return (answer, matched_question, score) for the closest FAQ match."""
    processed = preprocess(user_message)
    if not processed:
        return None, None, 0.0

    user_vec = VECTORIZER.transform([processed])
    scores = cosine_similarity(user_vec, FAQ_MATRIX)[0]
    best_idx = scores.argmax()
    best_score = float(scores[best_idx])

    if best_score < CONFIDENCE_THRESHOLD:
        return None, None, best_score

    return FAQS[best_idx]["answer"], FAQS[best_idx]["question"], best_score


@app.route("/")
def index():
    return render_template("index.html", faq_count=len(FAQS))


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({"error": "Please type a question."}), 400

    answer, matched_question, score = get_best_match(message)

    if answer is None:
        return jsonify({
            "answer": "Sorry, I'm not sure about that. Could you rephrase your question, "
                      "or contact our support team directly for help?",
            "matched_question": None,
            "confidence": round(score, 2)
        })

    return jsonify({
        "answer": answer,
        "matched_question": matched_question,
        "confidence": round(score, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)
