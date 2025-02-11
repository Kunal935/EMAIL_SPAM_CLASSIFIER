# Email Spam Classification

This project is an email spam classification system that uses machine learning to classify emails as spam or not spam. The system 
processes raw email text, extracts metadata, and uses machine learning models to predict whether the email is spam or not.

## Features

- **Email Text Preprocessing**: Cleans and preprocesses raw email text (removes stop words, punctuation, and performs lemmatization).
- **Metadata Extraction**: Extracts metadata features like sender domain, reply-to mismatch, subject length, number of links, attachments, etc.
- **Spam Classification**: Utilizes machine learning models such as Multinomial Naive Bayes (MNB) and XGBoost to classify emails.
- **Streamlit UI**: A simple web interface using Streamlit to allow users to input email text and get predictions.

## Installation

To run this project, you need Python 3.x and the necessary dependencies.

### Steps to set up the environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/email-spam-classification.git
   cd email-spam-classification
Install the required libraries: Make sure you have pip installed, then run the following command in your terminal:

bash
Copy
Edit
pip install -r requirements.txt
Run the Streamlit app: After installing the dependencies, you can start the Streamlit app:

bash
Copy
Edit
streamlit run app.py
Usage
Open the web interface.
Paste the email text into the text box.
Click on Classify Email to get the spam prediction.
The result will show whether the email is Spam or Not Spam, along with the extracted metadata features.
Models
This project uses the following machine learning models:

Multinomial Naive Bayes (MNB)
XGBoost Classifier (XGB)
Files Included
app.py: Streamlit application that provides the UI for the email spam classification.
spam.csv: Dataset used for training the model (emails labeled as spam or not spam).
vectorizer.pkl: Saved TF-IDF vectorizer for text feature extraction.
model.pkl: Trained spam classification model.
Requirements
Python 3.x
Streamlit
NLTK
Scikit-learn
XGBoost
Pandas
NumPy
PrettyTable
TLDExtract
