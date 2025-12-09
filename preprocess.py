import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# download necessary nltk data
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    words = text.split()

    cleaned = []
    for w in words:
        if w not in stop_words:
            cleaned.append(lemmatizer.lemmatize(w))

    return " ".join(cleaned)
