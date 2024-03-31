import datetime
import numpy as np
import speech_recognition as sr
from gtts import gTTS
import os
import transformers
import random
import pandas as pd
import taipy as tp
from taipy.gui.builder import text, input, button
from taipy.gui import Gui, State, notify

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

    def action_time(self):
        return datetime.datetime.now().time().strftime('%H:%M')

    def get_random_quote(self, df, topic=None):
        if not topic:
            return df.sample().quote.iloc[0]
        filtered_quotes = df[df.category == topic]
        if filtered_quotes.empty:
            return f"Sorry, I couldn't find a quote on the topic '{topic}'."
        return filtered_quotes.sample().quote.iloc[0]

def on_input_change(state):
    user_input = state.user_input
    ai = ChatBot(name="Dev", df=df)
    if ai.wake_up(user_input):
        if "time" in user_input:
            res = f"The time is {ai.action_time()}!"
        elif "quote" in user_input:
            topic =""
            words = user_input.split()
            for word in words:
                if "about" in word.lower() and "a" in word.lower() and "topic" in word.lower():
                    index = words.index(word)
                    topic = " ".join(words[index+1:]).strip()
            res = ai.get_random_quote(ai.df, topic)
        elif "hello" in user_input:
            res = f"Hello, my name is {ai.name}!"
        elif "thanks" in user_input:
            res = "Welcome, is there anything else you would like me to help you with?"
        elif "yes" in user_input.lower():
            res =  "Great! How can I assist you today"
        elif "no" in user_input.lower() or "im fine" in user_input.lower():
            res = "Bye! Have a great day!"
            state.keep_running = False
        else:
            res = "I'm here to help you with time or quotes. What else can I do for you?"
    else:
        res = "Sorry, I didn't catch that."
    state.response = res
    text_to_speech(res)

if __name__ == "__main__":
    df = pd.read_csv(r'E:\Documents\data_science\post_capstone\Everyday_Quotes\Everyday_Quotes\data\quotes_3.csv')

    state = {"user_input": "", "response": "", "keep_running": True}

    tp.Page(
        tp.Input(bind=state, name="user_input", on_change=on_input_change, placeholder="Enter your message..."),
        tp.Text(value=f"{state['response']}", mode="md")
    )

    tp.run(title="Taipy Chatbot", debug=True, use_reloader=True)
