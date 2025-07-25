import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(subject: str, description: str) -> str:
    def clean(text):
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = nltk.word_tokenize(text)
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalpha()]
        return ' '.join(tokens)

    clean_subject = clean(subject)
    clean_description = clean(description)
    return clean_subject + ' ' + clean_description

