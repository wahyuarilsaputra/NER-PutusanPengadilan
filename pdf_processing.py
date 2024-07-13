# pdf_processing.py
import PyPDF2
import re
import numpy as np
import pandas as pd
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import load_model
from gensim.models import Word2Vec
from pypdf import PdfReader

def pdf_to_text(file):
    text = ""
    reader = PdfReader(file)
    for page in reader.pages:
        text += page.extract_text()
    return text

def preprocess_text(text):
    text = text.replace("Mahkamah Agung Republik Indonesia\nMahkamah Agung Republik Indonesia\nMahkamah Agung Republik Indonesia\nMahkamah Agung Republik Indonesia\nMahkamah Agung Republik Indonesia\nDirektori Putusan Mahkamah Agung Republik Indonesia\nputusan.mahkamahagung.go.id\n", "")
    text = text.replace("\nDisclaimer\nKepaniteraan Mahkamah Agung Republik Indonesia berusaha untuk selalu mencantumkan informasi paling kini dan akurat sebagai bentuk komitmen Mahkamah Agung untuk pelayanan publik, transparansi dan akuntabilitas\npelaksanaan fungsi peradilan. Namun dalam hal-hal tertentu masih dimungkinkan terjadi permasalahan teknis terkait dengan akurasi dan keterkinian informasi yang kami sajikan, hal mana akan terus kami perbaiki dari waktu kewaktu.\nDalam hal Anda menemukan inakurasi informasi yang termuat pada situs ini atau informasi yang seharusnya ada, namun belum tersedia, maka harap segera hubungi Kepaniteraan Mahkamah Agung RI melalui :\nEmail : kepaniteraan@mahkamahagung.go.id", "")
    text = text.replace('P U T U S A N', 'PUTUSAN').replace('T erdakwa', 'Terdakwa').replace('T empat', 'Tempat').replace('T ahun', 'Tahun')
    text = text.replace('P  E  N  E  T  A  P  A  N', 'PENETAPAN').replace('J u m l a h', 'Jumlah').replace('M E N G A D I L I', 'MENGADILI')
    text = re.sub(r'(Hal\.\s*\S+(?:\s*\S+)?\.?\s*)|(Halaman \d+(?:\.\d+)?\s*)|Putusan Nomor \S+\s*', '', text)
    text = re.sub(r'\b0+(\d+)', r'\1', text)
    text = text.replace('\uf0d8', '').replace('\uf0b7', '').replace('\n', ' ')
    text = re.sub(r'([“”"])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'halaman\s*\d+\s*dari\s*\d+\s*', '', text)
    text = re.sub(r'^\s*dari\s+\d+\s+bkl\s+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*dari\s+\d+\s+smp\s+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+/PN', '/PN', text)
    text = re.sub(r'(\d+)\s*/pid', r'\1/pid', text)
    text = re.sub(r'(?i)(nomor)(\d+)', r'\1 \2', text)
    text = re.sub(r'(\d+)/\s*(pid\.\w+)/\s*(\d{4})/\s*(pn)', r'\1/\2/\3/\4', text, flags=re.IGNORECASE)
    return text.lower().strip()

def multiple_replace(text, replacements):
    regex = re.compile('|'.join(re.escape(key) for key in replacements.keys()))

    def replace(match):
        return replacements[match.group(0)]

    return regex.sub(replace, text)

def tokenize_text(text):
    tokens = re.findall(r'\w+|[^\w\s]', text)
    return tokens

def get_word_embedding(word, cbow_model):
    if word in cbow_model.wv:
        return cbow_model.wv[word]
    else:
        return np.zeros(cbow_model.vector_size)

def predict_labels(model, text, target_tags, cbow_model, idx2tag):
    # Dummy implementation, replace with actual prediction logic
    MAX_LEN = 200
    sentences = text.split(';')
    combined_results = {key: "" for key in target_tags}
    found_terdakwa = []
    tags_from_start = ["nomor putusan","nama terdakwa",  "melanggar UU", "putusan hukuman", "tuntutan hukuman"]
    tags_from_end = ["jenis dakwaan","tanggal putusan", "hakim ketua", "hakim anggota", "penuntut umum", "panitera"]

    for sent in sentences:
        words = sent.strip().split()
        words = [re.split(r'(\W)', word) for word in words]
        words = [item for sublist in words for item in sublist if item.strip()]
        embeddings = [get_word_embedding(word, cbow_model) for word in words]
        padded_sequence = pad_sequences([embeddings], maxlen=MAX_LEN, padding="post", dtype='float32', value=np.zeros(cbow_model.vector_size))
        predictions = model.predict(padded_sequence)
        predicted_labels = np.argmax(predictions, axis=-1)[0]
        predicted_tags = [idx2tag[idx] for idx in predicted_labels]
        temp_results = {key: [] for key in target_tags}

        for word, tag in zip(words, predicted_tags):
            for key, tags in target_tags.items():
                if tag in tags:
                    temp_results[key].append(word)
                    
        for key in temp_results:
            if key in tags_from_start and not combined_results[key] and temp_results[key]:
                combined_results[key] = ' '.join(sorted(set(temp_results[key]), key=temp_results[key].index))

    for sent in reversed(sentences):
        if all(combined_results[key] for key in tags_from_end):
            break
        words = sent.strip().split()
        words = [re.split(r'(\W)', word) for word in words]
        words = [item for sublist in words for item in sublist if item.strip()]
        embeddings = [get_word_embedding(word, cbow_model) for word in words]
        padded_sequence = pad_sequences([embeddings], maxlen=MAX_LEN, padding="post", dtype='float32', value=np.zeros(cbow_model.vector_size))
        predictions = model.predict(padded_sequence)
        predicted_labels = np.argmax(predictions, axis=-1)[0]
        predicted_tags = [idx2tag[idx] for idx in predicted_labels]
        temp_results = {key: [] for key in target_tags}

        for word, tag in zip(words, predicted_tags):
            for key, tags in target_tags.items():
                if tag in tags:
                    temp_results[key].append(word)

        for key in temp_results:
            if key in tags_from_end and not combined_results[key] and temp_results[key]:
                combined_results[key] = ' '.join(sorted(set(temp_results[key]), key=temp_results[key].index))

    if combined_results['hakim anggota']:
        combined_results['hakim anggota'] = process_hakim_anggota(combined_results['hakim anggota'])
                
    return combined_results

def process_hakim_anggota(text):
    matches = re.findall(r'(\w+\s\w+)', text)
    if matches:
        return ' dan '.join(matches)
    return text
