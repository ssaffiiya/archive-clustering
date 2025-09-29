import torch
import re

import numpy as np
import string
import stop_words

import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import pymorphy2

import unicodedata


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained('cointegrated/rubert-tiny2')
model = AutoModel.from_pretrained('cointegrated/rubert-tiny2')


russian_stopwords = stop_words.get_stop_words('ru')
russian_stopwords.extend(['...', '«', '»', 'здравствуйте','здравствуй','до свидания', 'добрый день', 'добрый вечер', 'доброе утро'])
english_stopwords = stop_words.get_stop_words('en')
all_stopwords = set(russian_stopwords + english_stopwords)


def normalize_unicode(text, form='NFC'):
    """Приводит текст к указанной Unicode-форме."""
    return unicodedata.normalize(form, text)


def clean_special_characters(text):
    """
    Удаляет специальные символы
    """
    
    text = re.sub(r"№", " ", text)
    text = re.sub(r"[●.․‥…]+", " ", text) 
    text = re.sub(r"–+|—+|-+|_+", " ", text)  

    text = re.sub(r"\s+", " ", text).strip()

    return text

def remove_stopwords(text, stopwords):
    """Удаляет стоп-слова из текста."""
    
    return ' '.join([word for word in text.split() if word not in stopwords])

def remove_punctuation(text):
    """Удаляет пунктуацию из текста."""
    return ''.join([ch if ch not in string.punctuation else ' ' for ch in text])

def remove_numbers(text):
    """Удаляет числа из текста."""
    return ''.join([i if not i.isdigit() else ' ' for i in text])

def remove_multiple_spaces(text):
    """Удаляет лишние пробелы."""
    text = re.sub(r'[\u200B\u00A0\u200C\u200D\s]+', ' ', text)

    return re.sub(r'\s+', ' ', text).strip()

def remove_quotes_and_brackets(text):
    """Удаляет все виды кавычек и скобок из текста."""
    return re.sub(r'[\"\“”"\'‘’ «»<>{}\[\]()\(\)]', ' ', text)
           

def remove_links(text):
    """Удаляет ссылки из текста."""
    text = re.sub(r'http[s]?://\S+', ' ', text)
    text = re.sub(r'www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    return text
def clean_text(text):

    header_pattern = r"^правительство российской федерации.*?образовательная программа"
    text = re.sub(header_pattern, "", text, flags=re.DOTALL | re.IGNORECASE)


    return text.strip()
def remove_bibliography(text):

    """Удаляет библиографию."""

    keywords = ['литература', 'список литературы', 'источники', 'библиография','references', 'bibliography', 'works cited']
    pattern = r'(' + '|'.join(keywords) + r')\b'

    match = list(re.finditer(pattern, text, flags=re.IGNORECASE))
    if match:
        last_match = match[-1]
        return text[:last_match.start()].strip()

    return text.strip()

def preprocess_text(text):

    """Полный процесс предобработки текста."""

    text = text.lower()
    
    text = remove_links(text)
    text=clean_special_characters(text)
    text = normalize_unicode(text)
    text = remove_numbers(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text, all_stopwords)
    text= remove_quotes_and_brackets(text)
    text = remove_multiple_spaces(text)
    text = clean_text(text)
    text = remove_bibliography(text)

    return text


def lemmatize_text(text):
    """Лемматизирует текст."""
    morph = pymorphy2.MorphAnalyzer()
    text_lem = [morph.parse(word)[0].normal_form for word in text.split(' ')]
    return ' '.join(text_lem)



def split_text(text, max_len=512, overlap=0):
    tokens = tokenizer.encode(text, truncation=False, add_special_tokens=True)
    segments = []
    for i in range(0, len(tokens), max_len - overlap):
        segment = tokens[i:i + max_len]
        segments.append(segment)
    return segments


def pad_embedding(embedding, target_len=2000):
    """

    """
    current_len = embedding.shape[0]

    if current_len > target_len:
        
        trimmed_embedding = embedding[:target_len - 1]  
        tail_info = embedding[target_len - 1:].mean(dim=0)  
        final_embedding = torch.cat([trimmed_embedding, tail_info.unsqueeze(0)], dim=0)

    elif current_len < target_len:
        
        padding = torch.zeros(target_len - current_len, device=embedding.device)
        final_embedding = torch.cat([embedding, padding], dim=0)

    else:
        
        final_embedding = embedding

    return final_embedding


def get_embedding_for_text(text, max_len=512, overlap=0, target_len=2000, title=False):
    """
    Получение эмбеддинга для текста с приведением к длине target_len.
    """
    segments = split_text(text, max_len, overlap)
    embeddings = []
    for segment in segments:
        input_ids = torch.tensor([segment]).to(device)
        attention_mask = torch.ones(input_ids.shape, device=device)
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            embedding = output.last_hidden_state.mean(dim=1).squeeze(0)
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
            embeddings.append(embedding)


    final_embedding = torch.cat(embeddings, dim=0)


    if not title:
          final_embedding = pad_embedding(final_embedding, target_len=target_len)


    return final_embedding
def process_multiple_texts(texts, max_len=512, overlap=0, title=False):
    """
    Функция предобработки и создания эбеддингов для текстов
    """
    all_embeddings = []
    for text in tqdm(texts, desc="Обработка текстов"):
        cleaned_text = preprocess_text(text)
        lemmatized_text = lemmatize_text(cleaned_text)
        embedding = get_embedding_for_text(lemmatized_text, max_len=max_len, overlap=overlap, title=title)
        all_embeddings.append(embedding.detach())



    return torch.stack(all_embeddings).cpu().numpy()



def get_embedd(input_data):

    ids = [item.get('id') for item in input_data]
    texts = [item.get('text') for item in input_data]
    annotations = [item.get('annotation') for item in input_data]
    titles = [item.get('title') for item in input_data]

 
    embeddings = process_multiple_texts(texts,max_len=2048, overlap=0) # эмбеддинги и обработка текстов
   
    embeddings_ann = process_multiple_texts(annotations,max_len=2048, overlap=0, title=True)
  

    embeddings_title = process_multiple_texts(titles,max_len=2048, overlap=0)
  


    result = []
    for idx, embedding in enumerate(embeddings):
        result.append({
            "id": ids[idx],
            "title": embeddings_title[idx].tolist(),
            "annotation": embeddings_ann[idx].tolist(),
            "text": embedding.tolist()
        })

    return result
    


if __name__ == "__main__":
    import sys 
    import codecs
    sys.stdin = codecs.getwriter('utf-8')(sys.stdin.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)
    request = input()
    input_texts = json.loads(request)
    embeddings =get_embedd(input_texts) 

print(json.dumps(embeddings))