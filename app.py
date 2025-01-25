import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string
import re
import pickle
import html
import unicodedata
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

model = load_model('model.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

stop_words = set(stopwords.words('english'))

def remove_special_chars(text):
    re1 = re.compile(r'  +')
    x1 = text.lower().replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x1))

def remove_non_ascii(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

def to_lowercase(text):
    return text.lower()

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def replace_numbers(text):
    return re.sub(r'\d+', '', text)

def remove_stopwords(words, stop_words):
    return [word for word in words if word not in stop_words]

def lemmatize_words(words):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]

def lemmatize_verbs(words):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word, pos='v') for word in words])

def text2words(text):
    return word_tokenize(text)

def normalize_text(text):
    text = remove_special_chars(text)
    text = remove_non_ascii(text)
    text = remove_punctuation(text)
    text = to_lowercase(text)
    text = replace_numbers(text)
    words = text2words(text)
    words = remove_stopwords(words, stop_words)
    words = lemmatize_words(words)
    words = lemmatize_verbs(words)
    return ' '.join(words)

def preprocess_text(text, tokenizer, max_length=100):
    text = normalize_text(text)
    encoded_text = tokenizer.texts_to_sequences([text])
    padded_text = pad_sequences(encoded_text, maxlen=max_length, padding='post')
    return padded_text

st.title('Model Prediction')

user_input = st.text_area('Input your text:')

if st.button('Predict'):
    if user_input:
        processed_input = preprocess_text(user_input, tokenizer)
        prediction = model.predict(processed_input)
        
        if prediction >= 0.5:
            st.write('Positive review')
        else:
            st.write('Negative review')
st.write(f"Prediction score: {prediction}")
