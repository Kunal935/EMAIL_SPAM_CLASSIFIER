# Email Spam Classification

This project uses machine learning to classify emails as either spam or not spam based on their content and metadata. It leverages several features from the email message, such as the subject, sender's domain, and presence of suspicious links or attachments. The model is built using a combination of Natural Language Processing (NLP) techniques and machine learning algorithms.

## Overview

This repository contains two main parts:

1. **Data Preprocessing and Model Training:** We prepare the dataset, extract relevant metadata, clean and preprocess the text, and train a machine learning model to classify emails.
2. **Streamlit Web Application:** A simple user interface that allows you to input email text and get a classification of whether the email is spam or not.

## Features

- **Text Preprocessing:** Emails are processed by tokenizing, removing stopwords, and lemmatizing words.
- **Metadata Extraction:** Key email metadata like sender domain, subject length, number of links, and attachments are extracted for classification.
- **Model Evaluation:** We train and evaluate different machine learning models (Multinomial Naive Bayes and XGBoost) on the dataset to determine the best performer.
- **Interactive Web Interface:** Using Streamlit, the app allows users to input email content and instantly classify whether the email is spam or not.

 👇👇👇👇👇👇👇👇👇👇👇👇👇👇
👉👉 DIRECT LINK OF THIS GAME 👈👈 
 :- https://emailspamclassifier-tttmxmvf4p46na8dhbffii.streamlit.app/

👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆


                  OR  
--you can install locally in your computer--

## Installation

To run the project, follow these steps:

### Clone the repository:
```bash
git clone https://github.com/your_username/email_spam_classification.git
cd email_spam_classification
```

### Install the necessary dependencies:
```bash
pip install -r requirements.txt
```

The `requirements.txt` includes the following dependencies:

```
streamlit
nltk
scikit-learn
xgboost
pandas
numpy
prettytable
tldextract
pickle
```

Download or place the `spam.csv` dataset (used for training the model) in the project directory.

### Run the Streamlit application:
```bash
streamlit run app.py
```

This will open the app in your browser where you can start classifying emails.

## How It Works

### Data Preparation
The data consists of labeled emails, with each email marked as either spam or not spam. The text of each email is preprocessed by tokenizing it, converting it to lowercase, removing stopwords, and lemmatizing the words.

### Metadata Extraction
Additional features are extracted from the email metadata, such as the sender’s domain, whether the email is HTML formatted, the presence of links and attachments, and other factors that might indicate spam.

### Model Training
The processed email text and extracted metadata are used to train a machine learning model. We use Multinomial Naive Bayes and XGBoost classifiers, which are evaluated based on accuracy, precision, and confusion matrix.

### Streamlit Web Interface
The app allows you to input the email text, preprocesses the text and metadata, and classifies the email as spam or not spam using the trained model. It then displays the classification result along with the extracted metadata.

## Example Usage

To classify an email, simply copy and paste the email content into the text box on the web interface and press "Classify Email". The model will then display whether the email is spam or not.

### Example Output:
- **Spam Email:** "🚨 Spam Email! 🚨"
  - The email might contain suspicious elements such as unknown domains, mismatched reply-to addresses, or suspicious links.
- **Not Spam Email:** "✅ Not Spam Email"
  - The email does not show typical spam characteristics.

## Model Performance

We evaluate the model's performance using accuracy, precision, and confusion matrix. The models are trained on a dataset of labeled emails, and their effectiveness is tested on unseen data. The results are shown in a table format for easy comparison.



