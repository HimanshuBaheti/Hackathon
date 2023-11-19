# Import necessary libraries
from flask import Flask, render_template, request
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.high_level import extract_text

nltk.download('stopwords')
nltk.download('punkt')

# Load your dataset
df = pd.read_csv('jobdata.csv')

# Function to extract text from a PDF file using pdfminer
def extract_text_from_pdf(pdf_path):
    text = extract_text(pdf_path)
    return text

# Function to extract keywords from a resume using NLTK
def extract_keywords_nltk(resume_text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = word_tokenize(resume_text)
    keywords = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]
    return " ".join(keywords)

# Function to get top matching job titles based on keywords
# Function to get top matching jobs based on keywords
def get_matching_jobs(user_keywords):
    vectorizer = CountVectorizer().fit(df['Key Skills'].fillna('').astype(str))
    user_vector = vectorizer.transform([user_keywords])

    # Convert sparse matrix to dense array
    user_vector = user_vector.toarray()

    cosine_similarities = cosine_similarity(user_vector, vectorizer.transform(df['Key Skills'].fillna('').astype(str)).toarray())
    top_indices = cosine_similarities.argsort()[0][-10:][::-1]
    top_jobs = df.loc[top_indices]

    return top_jobs


# Flask App
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        resume_file = request.files['resume_file']

        # Save the uploaded PDF file
        resume_path = 'uploads/user_resume.pdf'
        resume_file.save(resume_path)

        # Extract text from the PDF file
        resume_text = extract_text_from_pdf(resume_path)

        # Extract keywords from the user's resume using NLTK
        user_keywords = extract_keywords_nltk(resume_text)

        # Get top matching job titles
        top_jobs = get_matching_jobs(user_keywords)

        return render_template('result2.html', user_keywords=user_keywords, top_jobs=top_jobs)

if __name__ == '__main__':
    app.run(debug=True)
