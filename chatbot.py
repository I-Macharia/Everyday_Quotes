import datetime
import numpy as np
import speech_recognition as sr
from gtts import gTTS
import os
import transformers
import random

class ChatBot:
    def __init__(self, name):
        self.name = name

    def wake_up(self, text):
        return self.name in text.lower()

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
    def text_to_speech(text):
        print("AI --> ", text)
        speaker = gTTS(text=text, lang="en", slow=False)
        speaker.save("res.mp3")
        os.system("start res.mp3")  #if you have a macbook->afplay or for windows use->start
        os.remove("res.mp3")
    
    @staticmethod
    def get_random_quote(df):
        return df.sample().quote.iloc[0]
        
# Execute the AI
if __name__ == "__main__":
    ai = ChatBot(name="Dev")
    while True:
        ai.speech_to_text()
        if ai.wake_up(ai.text):
            res = "Hello I am Dev the AI, what can I do for you?"
        elif "time" in ai.text:
            res = ai.action_time()
        # Generate a random quote
        elif "quote" in ai.text:
            res = ai.get_random_quote(df)
        ai.text_to_speech(res)
        
        