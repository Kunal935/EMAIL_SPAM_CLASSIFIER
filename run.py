import streamlit as st
import pickle
import re
import email
import string
import tldextract
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Load saved models and vectorizers
nltk.download('punkt')
nltk.download('stopwords')
nltk.download()
nltk.download('wordnet')
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Function to preprocess and clean the email text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [i for i in text if i.isalnum()]
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(i, pos='v') for i in text]
    return " ".join(text)

# Metadata extraction function
def extract_metadata(email_content):
    try:
        msg = email.message_from_string(email_content)
        sender = msg["From"] or ""
        domain = sender.split("@")[-1] if "@" in sender else "unknown"
        reply_to = msg["Reply-To"] or ""
        subject = msg["Subject"] or ""

        features = {
            "Sender_Domain": domain if domain != "unknown" else "unknown_domain",
            "Reply_To_Mismatch": int(reply_to != "" and reply_to != sender),
            "Subject_Length": len(subject),
            "Is_HTML": int(msg.get_content_type() == "text/html"),
            "Num_Links": len(re.findall(r"https?://[^\s]+", msg.get_payload(decode=True).decode(errors="ignore") if msg.get_payload() else "")),
            "Num_Attachments": sum(1 for part in msg.walk() if part.get_content_disposition() == "attachment"),
            "Is_Shortened_URL": int(any(tldextract.extract(url).domain in {"bit.ly", "tinyurl.com"} for url in re.findall(r"https?://[^\s]+", msg.get_payload(decode=True).decode(errors="ignore") if msg.get_payload() else "")))
        }
    except:
        features = {
            "Sender_Domain": "unknown_domain",
            "Reply_To_Mismatch": 0,
            "Subject_Length": 0,
            "Is_HTML": 0,
            "Num_Links": 0,
            "Num_Attachments": 0,
            "Is_Shortened_URL": 0
        }
    return features

# Streamlit user interface
st.markdown("""
    <style>
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-20px); }
    }
    .header {
        font-size: 3rem;
        color: #FF6347;
        animation: bounce 1s infinite;
    }
    .result {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
    }
    .result .spam {
        color: red;
        background-color: #FF6347;
        animation: alert 1s ease-in-out infinite;
        padding: 10px;
        border-radius: 5px;
    }
    .result .not-spam {
        color: green;
        animation: none;
    }
    @keyframes alert {
        0% { background-color: red; color: white; }
        50% { background-color: yellow; color: black; }
        100% { background-color: red; color: white; }
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="header">📧 Email Spam Classification 🚨</div>', unsafe_allow_html=True)

# Input email text from user
email_text = st.text_area("Enter Email Text", height=200)

# Create a button to trigger classification
if st.button("Classify Email"):
    if email_text:
        with st.spinner('Processing the email... Please wait.'):

            # Extract metadata and preprocess text
            metadata = extract_metadata(email_text)
            transformed_text = transform_text(email_text)
            metadata["Processed_Text"] = transformed_text

            # Display the metadata features in an interactive table
            st.subheader("Metadata Features Extracted")
            metadata_df = pd.DataFrame([metadata])
            st.dataframe(metadata_df, use_container_width=True)

            # Feature extraction for prediction
            X_text = vectorizer.transform([transformed_text]).toarray()

            # Encode the 'Sender_Domain' column using LabelEncoder
            domain_encoder = LabelEncoder()
            metadata['Sender_Domain'] = domain_encoder.fit_transform([metadata['Sender_Domain']])[0]

            # Prepare the metadata features as a numpy array
            metadata_columns = ['Sender_Domain', 'Reply_To_Mismatch', 'Subject_Length', 'Is_HTML', 'Num_Links', 'Num_Attachments', 'Is_Shortened_URL']
            X_metadata = np.array([[metadata[col] for col in metadata_columns]])

            # Combine text features and metadata features
            X = np.hstack([X_text, X_metadata])

            # Make prediction using the model
            prediction = model.predict(X)
            result = "Spam" if prediction == 1 else "Not Spam"
            
            # Display result with engaging feedback and animations
            if result == "Spam":
                st.markdown('<div class="result spam">🚨 Spam Email! 🚨</div>', unsafe_allow_html=True)
                st.write("This email contains suspicious elements, such as unknown domains, mismatched reply-to, and suspicious links.")
                st.image("gif/goku.gif", use_column_width=True)

            else:
                st.markdown('<div class="result not-spam">✅ Not Spam Email</div>', unsafe_allow_html=True)
                st.image("gif/tom.gif", use_column_width=True)

    else:
        st.error("Please enter some email text for classification.")
