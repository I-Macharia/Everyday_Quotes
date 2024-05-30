import numpy as np
import os
import random
import pandas as pd
import datetime
import pickle
import spacy
from nltk.corpus import stopwords
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to load data
def load_data():
    with open('data/quotes_vectorized.pickle', 'rb') as f:
        quotes_vectorized = pickle.load(f)
    with open('data/tweets_vectorized.pickle', 'rb') as q:
        tweets_vectorized = pickle.load(q)
    with open('data/quotes_2_pickle', 'rb') as d:
        quotes_2 = pickle.load(d)
    df = pd.read_csv('data/quotes_2.csv')
    with open('data/combined_text', 'rb') as ct:
        combined_text = pickle.load(ct)
    return quotes_vectorized, tweets_vectorized, df, quotes_2, combined_text

quotes_vectorized, tweets_vectorized, df, quotes_2, combined_text = load_data()

class DataHandler:
    def __init__(self, quotes_vectorized, tweets_vectorized, df):
        self.quotes_vectorized = quotes_vectorized
        self.tweets_vectorized = tweets_vectorized
        self.tweets_vectorized_downsampled = self.tweets_vectorized[np.random.choice(tweets_vectorized.shape[0], 3000, replace=False)]
        self.vectorizer = TfidfVectorizer(stop_words=stopwords.words('english')).fit(combined_text)
        self.df = df
    
    def find_quote_for_tweet(self, topic):
        cleaned_topic = self.clean_text(topic)
        vectorized_topic = self.vectorizer.transform([cleaned_topic])
        model = KNeighborsRegressor(n_neighbors=1)
        model.fit(self.quotes_vectorized, self.tweets_vectorized_downsampled)
        distance, index = model.kneighbors(vectorized_topic)
        quote = self.df.iloc[index[0][0]]['quote']
        author = self.df.iloc[index[0][0]]['author']
        return quote, author
    
    def clean_text(self, text):
        return text.lower().strip()

class ChatBot:
    def __init__(self, name, data_handler):
        self.name = name
        self.data_handler = data_handler
        self.conversation_history = []
        self.responses = {
            "hello": self.get_hello_response,
            "thanks": self.get_thanks_response,
            "yes": self.get_yes_response,
            "no": self.get_no_response,
            "time": self.get_time_response,
            "date": self.get_date_response,
            "dev": self.get_dev_response,
            "hello dev": self.get_hello_dev_response,
            "what can you do": self.get_capabilities_response,
            "capabilities": self.get_capabilities_response,
            "quote about time": self.get_quote_about_time_response,
        }

    @staticmethod
    def wake_up(text):
        wake_words = ["hey", "hi", "hello", "dev", "assistant", "time", "date", "quote", "thanks", "yes", "goodbye", "what can you do", "capabilities"]
        synonyms = ["howdy", "greetings", "salutations", "morning", "afternoon", "evening"]
        all_wake_words = wake_words + synonyms
        return any(word in text.lower() for word in all_wake_words)

    @staticmethod
    def action_time():
        return datetime.datetime.now().time().strftime('%H:%M')

    @staticmethod
    def action_date():
        return datetime.datetime.now().date().strftime('%B %d, %Y')

    def get_hello_response(self):
        return f"Hello, my name is {self.name}!"

    def get_thanks_response(self):
        return "You're welcome! Is there anything else you would like me to help you with?"

    def get_yes_response(self):
        return "Great! How can I assist you today?"

    def get_no_response(self):
        return "Bye! Have a great day!"

    def get_time_response(self):
        return f"The time is {self.action_time()}!"

    def get_date_response(self):
        return f"Today's date is {self.action_date()}!"

    def get_dev_response(self):
        return "I'm happy to help you with whatever you need."

    def get_hello_dev_response(self):
        return "I'm doing well, how about you?"

    def get_capabilities_response(self):
        return (
            "I can assist you with various tasks such as telling the current time, providing today's date, "
            "offering quotes on specific topics, and more. Just let me know how I can help!"
        )

    def get_quote_about_time_response(self):
        return self.data_handler.find_quote_for_tweet("time")

    def generate_response(self, text):
        if not self.wake_up(text):
            return "Sorry, I didn't understand that. Please try asking in a different way."

        doc = nlp(text)
        if "time" in [token.text for token in doc]:
            return self.get_time_response()

        elif "quote" in [token.text for token in doc] and "about" in [token.text for token in doc]:
            words = text.lower().split()
            topic_index = words.index("about") + 1
            if topic_index >= len(words):
                return "Please specify a topic for the quote."
            topic = " ".join(words[topic_index:])
            return self.data_handler.find_quote_for_tweet(topic)

        elif any(word.lower() in self.responses for word in text.split()):
            for word in text.split():
                if word.lower() in self.responses:
                    return self.responses[word.lower()]()

        return self.get_capabilities_response()
    
    def get_conversation_history(self):
        return self.conversation_history

if __name__ == "__main__":
    data_handler = DataHandler(quotes_vectorized, tweets_vectorized, df)
    ai = ChatBot(name="Dev", data_handler=data_handler)
    while True:
        user_input = input("You --> ")
        response = ai.generate_response(user_input)
        if isinstance(response, tuple):
            print(f"AI --> {response[0]}")
            print(f"AI --> {response[1]}")
        else:
            print(f"AI --> {response}")
