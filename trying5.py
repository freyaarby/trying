import streamlit as st
import pandas as pd
df = pd.read_csv("arxiv_ml.csv")
print(df.head())

#Data Processing
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

#Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

#Melakukan Preprocessing
def preprocess_text(text):

    #Penghilangan Lowercasing
    text = text.lower()

    #Penghilangan Karakter Spesial
    text = re.sub(r'\W', ' ', text)

    #Tokenisasi
    tokens = word_tokenize (text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

#Menerapkan preprocessing ke dataset
df["processed_abstract"] = df["abstract"].apply(preprocess_text)

import nltk
nltk.download("punkt")  
from nltk.tokenize import sent_tokenize

def chunk_abstract(abstract, chunk_size=3):
    sentences = sent_tokenize(abstract)  # Pisahkan menjadi kalimat
    chunks = [" ".join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]
    return chunks

#Terapkan ke semua abstrak
df["chunks"] = df["abstract"].apply(chunk_abstract)

#Melihat hasilnya
print(df[["abstract", "chunks"]].head())

# Import TfidfVectorizer and cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Create a TfidfVectorizer object
vectorizer = TfidfVectorizer()

#Fit and transform the preprocessed abstracts to create the TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(df["processed_abstract"])

def retrieve_top_k(query, tfidf_matrix, vectorizer, df, k=5):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = scores.argsort()[-k:][::-1] 
    return df.iloc[top_indices] 

#Contoh penggunaan
query = "What is machine learning?"
retrieved_docs = retrieve_top_k(query, tfidf_matrix, vectorizer, df, k=5)

#Menyimpan hanya 5 dokumen teratas
df = retrieved_docs.reset_index(drop=True) 
print(df)


df = df.iloc[:5] 
print(df)

!pip install transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load tokenizer dan model T5
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

def format_input(question, context):
    return f"question: {question} context: {context}"

def generate_answer(question, df):
    combined_context = " ".join(df["chunks"].sum())  
    input_text = format_input(question, combined_context)

    # Tokenisasi input
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)

    # Generate jawaban
    output_ids = model.generate(input_ids, max_length=100, num_beams=5, early_stopping=True)
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return answer