import pickle
import re
import nltk
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load classifier and TF-IDF vectorizer
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# Function to clean resume text
def cleanResume(txt):
    # Remove URLs
    cleanTxt = re.sub(r'http\S+|www\S+', '', txt)
    # Replace RT and CC
    cleanTxt = re.sub(r'\bRT\b|\bCC\b', ' ', cleanTxt)
    # Remove mentions (e.g. @username)
    cleanTxt = re.sub(r'@\S+', ' ', cleanTxt)
    # Remove hashtags (e.g. #hashtag)
    cleanTxt = re.sub(r'#\S+', ' ', cleanTxt)
    # Remove special characters, keeping spaces
    cleanTxt = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~]', ' ', cleanTxt)
    # Remove non-ASCII characters
    cleanTxt = re.sub(r'[^\x00-\x7f]', ' ', cleanTxt)
    # Replace multiple spaces with a single space
    cleanTxt = re.sub(r'\s+', ' ', cleanTxt).strip()
    return cleanTxt

# Function to generate word cloud
def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# Web app function
def main():
    st.title("Resume Ranker!")

    # Upload file option
    upload_file = st.file_uploader('Upload Resume ', type=['txt', 'pdf', 'docx', 'rtf'])

    if upload_file is not None:
        try:
            # Read uploaded file
            resume_bytes = upload_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')

        # Clean resume text
        cleaned_resume = cleanResume(resume_text)
        cleaned_resume_vectorized = tfidf.transform([cleaned_resume])

        # Predict category
        prediction_id = clf.predict(cleaned_resume_vectorized)[0]

        # Define category mappings
        category_mapping = {
            6: 'Data Science',
            12: 'HR',
            0: 'Advocate',
            1: 'Arts',
            24: 'Web Designing',
            16: 'Mechanical Engineer',
            22: 'Sales',
            14: 'Health and fitness',
            5: 'Civil Engineer',
            15: 'Java Developer',
            4: 'Business Analyst',
            21: 'SAP Developer',
            2: 'Automation Testing',
            11: 'Electrical Engineering',
            18: 'Operations Manager',
            20: 'Python Developer',
            8: 'DevOps Engineer',
            17: 'Network Security Engineer',
            19: 'PMO',
            7: 'Database',
            13: 'Hadoop',
            10: 'ETL Developer',
            9: 'DotNet Developer',
            3: 'Blockchain',
            23: 'Testing'
        }

        # Get predicted category name
        category_name = category_mapping.get(prediction_id, "Unknown")

        # Display predicted category
        st.subheader(f"Predicted Category: {category_name}")

        # Generate and display word cloud
        st.subheader("This Resume")
        generate_word_cloud(cleaned_resume)

        # Feedback section
        st.subheader("Feedback")
        feedback = st.text_area("Please provide your feedback on the prediction:")
        if st.button("Submit Feedback"):
            st.success("Thank you for your feedback!")

        # Rating system
        st.subheader("Rate the Accuracy of the Prediction")
        rating = st.slider("Rate from 1 (Poor) to 5 (Excellent)", 1, 5, 3)
        if st.button("Submit Rating"):
            st.success(f"Thank you for rating the prediction: {rating} out of 5!")

        st.subheader("For more info, contact us!")

# Run the app
if __name__ == '__main__':
    main()