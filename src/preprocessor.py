import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """
    Clean the input text by removing punctuation, stop words, and lemmatizing.
    """
    if pd.isna(text):
        return ''
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+","", text)  # Remove URLs
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = nltk.word_tokenize(text)  # Tokenize the text
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalpha()]  # Lemmatize tokens
    return ' '.join(tokens)

# Only run below when executing this file directly
if __name__ == "__main__":
    print("Preprocessing Text...")

    df = pd.read_csv('D:/Users/balans/Desktop/PBI/Customer Support Ticket Prioritization/data/merged_tickets.csv')
    df['clean_subject'] = df['subject'].apply(clean_text)
    df['clean_description'] = df['description'].apply(clean_text)
    df['text'] = df['clean_subject'] + ' ' + df['clean_description']

    df[['ticket_id','text', 'priority_label']].to_csv('D:/Users/balans/Desktop/PBI/Customer Support Ticket Prioritization/data/cleaned_data.csv', index=False)
    print("Text preprocessing complete. File saved as 'cleaned_data.csv'.")
