import datetime
import numpy as np
import speech_recognition as sr
from gtts import gTTS
import os
import transformers
import random
import pandas as pd

df = pd.read_csv(r'E:\Documents\data_science\post_capstone\Everyday_Quotes\Everyday_Quotes\data\quotes_3.csv')

def text_to_speech(text):
    print("AI --> ", text)
    speaker = gTTS(text=text, lang="en", slow=False)
    # speaker.save("res.mp3")
    # os.system("start res.mp3")  #if you have a macbook->afplay or for windows use->start
    # os.remove("res.mp3")


class ChatBot:
    def __init__(self, name, df):
        self.name = name
        self.df = df

    def wake_up(self, text):
        wake_words = ["dev", "bot", "assistant", "time", "hello", "quote", "thanks", "no", "yes"]
        return any(word in text.lower() for word in wake_words)

    @staticmethod
    def text_to_speech(text):
        print("AI --> ", text)
        speaker = gTTS(text=text, lang="en", slow=False)
        # speaker.save("res.mp3")
        # os.system("start res.mp3")  #if you have a macbook->afplay or for windows use->start
        # os.remove("res.mp3")
    
    def speech_to_text(self):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            audio = r.listen(source)
        try:
            self.text = r.recognize_google(audio)
            print("You --> ", self.text)
        except Exception:
            print("Sorry, I did not get that")
            self.text = ""

    @staticmethod
    def action_time():
        return datetime.datetime.now().time().strftime('%H:%M')
    
    @staticmethod
    def get_random_quote(df, topic=None):
        if not topic:
            return df.sample().quote.iloc[0]
        filtered_quotes = df[df.category == topic]
        if filtered_quotes.empty:
            return f"Sorry, I couldn't find a quote on the topic '{topic}'."
        return filtered_quotes.sample().quote.iloc[0]

if __name__ == "__main__":
    df = pd.read_csv(r'E:\Documents\data_science\post_capstone\Everyday_Quotes\Everyday_Quotes\data\quotes_3.csv')

    ai = ChatBot(name="Dev", df=df)
    while True:
        user_input = input("You --> ")
        ai.text = user_input

        if ai.wake_up(ai.text):
            if "time" in ai.text:
                res = f"The time is {ai.action_time()}!"
            elif "quote" in ai.text:
                topic =""
                words = ai.text.split()
                for word in words:
                    if "about" in word.lower() and "a" in word.lower() and "topic" in word.lower():
                        index = words.index(word)
                        topic = " ".join(words[index+1:]).strip()
                res = ai.get_random_quote(ai.df, topic)
            elif "hello" in ai.text:
                res = f"Hello, my name is {ai.name}!"
            elif "thanks" in ai.text:
                res = "Welcome, is there anything else you would like me to help you with?"
            elif "yes" in ai.text:
                res =  "Great! How can I assist you today"
            elif "no" in ai.text.lower() or "im fine" in ai.text.lower():
                print("Bye! Have a great day!")
                break
            else:
                res = "I'm here to help you with time or quotes. What else can I do for you?"
        else:
            res = "Sorry, I didn't catch that."

        text_to_speech(res)