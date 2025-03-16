import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Define the file path
file_path = r"C:\Users\hrush\Downloads\Implementation of Chatbot using NLP\intents.json"


# Check if file exists
if not os.path.exists(file_path):
    st.error(f"Error: File not found at {file_path}. Please check the path or upload the file.")
    st.stop()

# Load intents.json safely
with open(file_path, "r", encoding="utf-8") as file:
    intents = json.load(file)

vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
    return "I'm sorry, I don't understand."

counter = 0

def main():
    global counter
    st.title("Intents of Chatbot using NLP")

    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")
        
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])
        
        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")
        
        if user_input:
            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")
            
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])
            
            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    elif choice == "Conversation History":
        st.header("Conversation History")
        
        if os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip the header row
                for row in csv_reader:
                    st.text(f"User: {row[0]}")
                    st.text(f"Chatbot: {row[1]}")
                    st.text(f"Timestamp: {row[2]}")
                    st.markdown("---")
        else:
            st.write("No conversation history available.")

    elif choice == "About":
        st.write("The goal of this project is to create a chatbot that can understand and respond to user input based on intents. The chatbot is built using Natural Language Processing (NLP) and Logistic Regression.")

        st.subheader("Project Overview:")
        st.write("""
        1. NLP techniques and Logistic Regression are used to train the chatbot on labeled intents and entities.
        2. Streamlit is used to build the chatbot interface, allowing users to input text and receive responses.
        """)

        st.subheader("Dataset:")
        st.write("""
        - Intents: Categories of user input (e.g., "greeting", "budget", "about").
        - Entities: Extracted details from user input.
        - Text: The user's input text.
        """)

        st.subheader("Streamlit Chatbot Interface:")
        st.write("The chatbot interface is built using Streamlit, with a text input box for users to interact and receive responses.")

        st.subheader("Conclusion:")
        st.write("""
        A chatbot was built using NLP and Logistic Regression. This project can be extended with additional data and more sophisticated NLP techniques such as deep learning.
        """)

if __name__ == '__main__':
    main()
