# FAQ Chatbot

A simple chatbot that answers user questions by matching them against a
set of FAQs using NLP preprocessing + TF-IDF cosine similarity.

## How it meets the task requirements
- **Collect FAQs**: stored in `faqs.json` as a list of
  `{question, answer, keywords}` objects — easy to edit or extend.
- **Preprocess the text**: `preprocess()` in `app.py` uses **NLTK** to
  lowercase the text, tokenize it, strip punctuation/stopwords, and
  lemmatize each word (e.g. "running" → "run").
- **Match user questions**: all FAQ questions (plus a few extra keyword
  synonyms per FAQ) are vectorized with **TF-IDF** (`scikit-learn`), and
  the user's message is compared against every FAQ using **cosine
  similarity**. The FAQ with the highest similarity score is picked, as
  long as it clears a minimum confidence threshold — otherwise the bot
  says it doesn't know and suggests contacting support.
- **Display the best matching answer**: returned as the chatbot's reply.
- **Optional feature included**: a simple chat-style UI
  (`templates/index.html`) with user/bot message bubbles, so it feels
  like chatting rather than filling out a form.

## How to run it
1. Make sure you have Python 3.9+ installed.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the app (the first run will download a few small NLTK data
   packages automatically):
   ```bash
   python app.py
   ```
4. Open your browser at **http://127.0.0.1:5000**

Try asking things like "How do I get my password reset?", "when will my
package arrive", or "can I get my money back" — the bot will match these
to the closest FAQ even though they're phrased differently.

## Project structure
```
faq_chatbot/
├── app.py                 # Flask backend: preprocessing + TF-IDF matching
├── faqs.json               # Editable FAQ dataset
├── requirements.txt
├── templates/
│   └── index.html          # Chat UI
└── README.md
```

## Customizing it for your own product
1. Edit `faqs.json` — replace the sample questions/answers with your own.
   The optional `"keywords"` field lets you add a few extra words or
   phrases users might type, which improves matching without needing an
   exact wording match.
2. Adjust `CONFIDENCE_THRESHOLD` in `app.py` (default `0.25`) — raise it
   to make the bot more cautious about weak matches, lower it to make it
   answer more often.
3. If you'd like true "intent matching" instead of keyword similarity
   (e.g. to handle much more varied phrasing), you could later swap the
   TF-IDF step for sentence embeddings (like `sentence-transformers`) and
   compare embeddings with cosine similarity the same way.
