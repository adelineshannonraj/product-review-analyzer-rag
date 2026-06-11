# Install packages
!pip install boltiotai
!pip install faiss-cpu
!pip install sentence-transformers
!pip install pandas numpy chardet openpyxl transformers torch

import pandas as pd
import re
import os
import numpy as np
import faiss
import chardet

from boltiotai import openai as bolt_openai
from sentence_transformers import SentenceTransformer
from google.colab import files
from transformers import pipeline

# API KEY
api_key = os.getenv("BOLTIOT_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key missing. Set BOLTIOT_API_KEY environment variable.")

# MEMORY OPTIMIZATION

def optimize_memory(df):
    for col in df.select_dtypes(include=['int64','float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer' if df[col].dtype=='int64' else 'float')
    return df


# TEXT CLEANING

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9 ]','',text)
    return text


# SENTIMENT MODEL

sentiment_pipeline = pipeline("sentiment-analysis")

# FILE ENCODING DETECTION

def detect_encoding(file_path):
    with open(file_path,'rb') as f:
        result = chardet.detect(f.read(100000))
    return result['encoding']


# DATASET LOAD

def load_dataset(file_path=None, subset_size=10000):

    if file_path is None:
        uploaded = files.upload()
        filename = list(uploaded.keys())[0]
        file_path = filename

    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path, engine='openpyxl')
    else:
        encoding = detect_encoding(file_path)

        try:
            df = pd.read_csv(file_path,encoding=encoding,on_bad_lines='skip',dtype=str)
        except:
            df = pd.read_csv(file_path,encoding='ISO-8859-1',on_bad_lines='skip',dtype=str)

    df = optimize_memory(df)

    review_column = df.columns[0]

    if len(df) > subset_size:
        df = df.sample(n=subset_size,random_state=42)

    df['original_text'] = df[review_column].fillna("")
    df['cleaned_text'] = df['original_text'].apply(clean_text)

    # Batch sentiment analysis
    sentiments = sentiment_pipeline(df['cleaned_text'].tolist(),batch_size=32)
    df['sentiment'] = [s['label'].lower() for s in sentiments]

    sentiment_counts = df['sentiment'].value_counts().to_dict()

    print("Sentiment Summary")
    print("Positive:",sentiment_counts.get('positive',0))
    print("Negative:",sentiment_counts.get('negative',0))
    print("Neutral:",sentiment_counts.get('neutral',0))

    return df

# BUILD FAISS INDEX

def build_faiss_index(df):

    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = model.encode(
        df['cleaned_text'].tolist(),
        convert_to_numpy=True
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    print("FAISS index built with", index.ntotal, "vectors")

    return index, model


# RAG QUERY FUNCTION

def analyze_reviews_with_gpt(query, df, index, model, k=5):

    query_embedding = model.encode([query],convert_to_numpy=True)

    if index.ntotal == 0:
        return "FAISS index is empty."

    distances,indices = index.search(query_embedding,k)

    retrieved_reviews = df.iloc[indices[0]]

    retrieved_text = "\n\n".join([
        f"Review: {review}\nSentiment: {sentiment}"
        for review,sentiment in zip(
            retrieved_reviews['original_text'],
            retrieved_reviews['sentiment']
        )
    ])

    prompt = f"""
Summarize the following product reviews.
Identify major themes, complaints and overall sentiment.

{retrieved_text}

User Question: {query}
"""

    try:

        response = bolt_openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role":"system","content":"You analyze customer reviews."},
                {"role":"user","content":prompt}
            ]
        )

        return response.choices[0].message.content

    except Exception as e:

        return f"API error: {str(e)}"

# QUERY PROCESSOR

def process_query(query, df, index, model):

    try:
        result = analyze_reviews_with_gpt(query, df, index, model)

        return {
            "query":query,
            "analysis":result
        }

    except Exception as e:

        return {
            "query":query,
            "error":str(e)
        }


# MAIN PIPELINE

df = load_dataset()

index, model = build_faiss_index(df)

print(df.head())


# INTERACTIVE LOOP


while True:

    user_query = input("Enter query (or 'quit'): ")

    if user_query.lower() == 'quit':
        break

    result = process_query(user_query, df, index, model)

    print(result)
