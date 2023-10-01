from transformers import TFAutoModel
from transformers import AutoTokenizer
import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image
from preprocess import text_preprocess  # Chắc chắn rằng bạn có preprocess.py ở cùng thư mục

st.set_page_config(
    page_title='Vietnamese ABSA',
    page_icon='https://cdn-icons-png.flaticon.com/512/8090/8090669.png',
    layout="centered"
)
st.image(image=Image.open('assets/homepage.png'), caption='Aspect-based Sentiment Analysis')

# Pre-trained model and tokenizer
PRETRAINED_MODEL = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

# Load your ASPECTS and REPLACEMENTS here
ASPECTS = [...]  # Fill in your ASPECTS
REPLACEMENTS = {...}  # Fill in your REPLACEMENTS dictionary

def create_model(optimizer):
    # Your model creation code here
    pass

# Your other functions (tokenize_function, preprocess_tokenized_dataset, predict) go here

# Driver
sentence = st.text_input(label='')

# Upload file
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if st.button('Analyze'):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        tokenized_data = tokenize_function(data)

        test_tf_dataset = preprocess_tokenized_dataset(tokenized_data, tokenizer)
        y_pred = predict(model, test_tf_dataset, 10, verbose=1)
        print(y_pred)

        aspect_polarity_dict = {aspect: [] for aspect in ASPECTS}
        print(aspect_polarity_dict)
        for sample_id in range(len(y_pred)):
            for aspect_id, aspect in enumerate(ASPECTS):
                if y_pred[sample_id][aspect_id] != 0:
                    aspect_polarity_dict[aspect].append(REPLACEMENTS[y_pred[sample_id][aspect_id]])
        print(aspect_polarity_dict)

        aspect_counts = {aspect: len(aspect_polarity_dict[aspect]) for aspect in aspect_polarity_dict}
        print(aspect_counts)
        plt.figure(figsize=(6, 6))
        plt.pie(aspect_counts.values(), labels=aspect_counts.keys(), autopct='%1.1f%%')
        plt.title('Aspect Distribution')
        plt.axis('equal')
        aspect_chart = plt.gcf()

        pol_pie_charts = {}
        for aspect in ASPECTS:
            plt.figure(figsize=(6, 6))
            counter = Counter(aspect_polarity_dict[aspect])
            plt.pie(counter.values(), labels=counter.keys(), autopct='%1.1f%%')
            plt.title(f'Polarity Distribution Of {aspect}')
            plt.axis('equal')
            pol_pie_charts[aspect] = plt.gcf()

        st.write('*Aspect Distribution:*')
        st.pyplot(aspect_chart)

        for aspect in ASPECTS:
            st.write(f'Polarity Distribution of {aspect}')
            st.pyplot(pol_pie_charts[aspect])
    else:
        clean_sentence = text_preprocess(sentence)
        tokenized_input = tokenizer(clean_sentence, padding='max_length', truncation=True)
        features = {x: [[tokenized_input[x]]] for x in tokenizer.model_input_names}
        pred = predict(model, Dataset.from_tensor_slices(features))

        sentiments = map(lambda x: REPLACEMENTS[x], pred[0])
        d = []
        for aspect, sentiment in zip(ASPECTS, sentiments):
            if sentiment:
                d.append(f'{aspect}#{sentiment}')
        st.markdown('*')
        st.write('*Sentence:*', sentence)
        st.write(', '.join(d))
