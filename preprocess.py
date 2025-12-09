import re
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    nltk_available = True
except Exception:
    nltk_available = False

if nltk_available:
    try:
        stopwords.words("english")
    except Exception:
        nltk.download("punkt", quiet=True)
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        nltk.download("stopwords", quiet=True)

def simple_clean(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    # remove html tags
    text = re.sub(r"<[^>]+>", " ", text)
    # remove urls
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # replace separators we used with space
    text = text.replace("|||", " ")
    # remove punctuation and numbers except keep basic letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_text(text: str) -> str:
    txt = simple_clean(text)
    if nltk_available:
        stops = set(stopwords.words("english"))
        tokens = txt.split()
        tokens = [t for t in tokens if t not in stops and len(t) > 1]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        return " ".join(tokens)
    else:
        small_stops = {"the","and","is","in","to","for","of","a","an","on","with","we","you","this"}
        tokens = [t for t in txt.split() if t not in small_stops and len(t) > 1]
        return " ".join(tokens)
