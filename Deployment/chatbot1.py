import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
from nltk import FreqDist
import streamlit as st
import pickle
from chatbot import ChatBot, DataHandler  # Import DataHandler class from chatbot module

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

def main_page():
    st.title("Chatbot App")

    quotes_vectorized, tweets_vectorized, df, quotes_2 , combined_text = load_data()

    name = "Dev"

    # Create a DataHandler instance
    data_handler = DataHandler(quotes_vectorized, tweets_vectorized, df)

    # Pass the data_handler to the ChatBot constructor
    ai = ChatBot(name=name, data_handler=data_handler)

    if conversation_history := ai.get_conversation_history():
        st.subheader("Conversation History")
        for interaction in conversation_history:
            st.write(f"You --> {interaction['user']}")
            st.write(f"AI --> {interaction['bot']}")
            st.write("---")

    # User input
    user_input = st.text_input("You -->")

    if st.button("Send"):
        if user_input:
            reply = ai.generate_response(user_input)
            st.write(f"AI --> {reply}")
        else:
            st.warning("Please enter a message.")

def about_page():  
    st.title("Daily Motivation Quotes")

    st.write("""

In a world filled with daily challenges and responsibilities, staying motivated is essential for personal growth and well-being. This data science project is dedicated to curating and delivering a diverse collection of carefully selected quotes. These inspirational snippets, sourced from various outlets including historical figures, popular literature, and notable personalities, aim to provide a source of encouragement, reflection, and empowerment for individuals facing the hustle of everyday life.
#### Objectives:

The primary objectives of this project are as follows:
1.	Curate Inspirational Quotes:
Gather a diverse collection of quotes from the Good Reads website, which boasts an extensive compilation of quotes spanning various genres and themes.
2.	Daily Motivational Updates: Develop a system to provide users with daily updates featuring a thoughtfully chosen quote. These updates will cater to different areas of life, ensuring a comprehensive and relatable experience.
3.	Tag-based Grouping: Implement a categorization mechanism that tags each quote based on its thematic content. This grouping will enable users to easily identify quotes that resonate with their specific preferences or current situations.

## Data Understanding
•	Source quotes from the Good Reads website, exploring the wide array of authors and themes available.

•	Analyze the structure of the collected data, including metadata such as author names, publication dates, and associated tags.
""")
    # Load data for graphs
    df = pd.read_csv('data/quotes_2.csv')
    quotes_2 = load_data()
    
    # st.write(df.columns)

    # Function to generate word cloud

    def author_names_word_cloud(data):
        wordcloud = WordCloud(width=800, height=400, background_color='black').generate(' '.join(data['author']))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

    def quotes_word_cloud(data):
        wordcloud = WordCloud(width=800, height=400, background_color='black').generate(' '.join(data['quote_2']))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

    # Define the function for the bar chart
    def author_contribution_bar_chart(data):
        
        # Get the top 10 authors by their number of quotes
        author_counts = data['author_2'].value_counts().head(10)
        
        # Create the bar chart
        plt.figure(figsize=(12, 6))
        sns.barplot(x=author_counts.values, y=author_counts.index, hue=author_counts.index, palette='viridis', dodge=False)
        plt.title('Top 10 Authors Contribution')
        plt.xlabel('Number of Quotes')
        plt.ylabel('Author')
        plt.legend(title='Authors', loc='lower right', labels=author_counts.index)
        
        # Display the plot 
        st.pyplot(plt)
        

    # # Convert lists of tags to tuples
    # df['Tags'] = df['Tags'].apply(tuple)

    # Calculate tag distribution
    tag_counts = df.groupby(by='Tags').size().nlargest(10).reset_index(name='Count')

    # Create the pie chart
    fig = px.pie(tag_counts, names='Tags', values='Count', title='Tags Distribution', custom_data=['Tags'])

    # Update layout to display custom data on hover
    fig.update_traces(hovertemplate='Tag: %{customdata[0]}<br>Count: %{value}')

    st.title('Word Clouds for Quotes Data')
    st.header('Word Cloud of Author Names')
    author_names_word_cloud(df)
    
    st.header('Word Cloud of Words in Quotes')
    quotes_word_cloud(df)
    
    st.title('Author Contribution Analysis')
    author_contribution_bar_chart(df)
    
    # Display the pie chart in Streamlit
    st.title('Tag Distribution Pie Chart')
    st.plotly_chart(fig)
   

# Add a sidebar with a selector for the page
pages = {'About': about_page, 'Main': main_page}
page = st.sidebar.selectbox('Go to', list(pages.keys()))

# Run the selected page
pages[page]()
