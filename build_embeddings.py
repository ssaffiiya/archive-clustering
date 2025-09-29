import torch
import re
import pandas as pd
import numpy as np
import string
import stop_words
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import json
import umap
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import pymorphy2
import plotly.express as px


russian_stopwords = stop_words.get_stop_words('ru')
russian_stopwords.extend(['...', '«', '»', 'здравствуйте','здравствуй','до свидания', 'добрый день', 'добрый вечер', 'доброе утро'])
english_stopwords = stop_words.get_stop_words('en')
all_stopwords = set(russian_stopwords + english_stopwords)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModel.from_pretrained("xlm-roberta-base").to(device)


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
    return re.sub(r'\s+', ' ', text).strip()

def remove_links(text):
    """Удаляет ссылки из текста."""
    text = re.sub(r'http[s]?://\S+', ' ', text)
    text = re.sub(r'www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    return text
def clean_text(text):

    header_pattern = r"^правительство российской федерации.*?образовательная программа"
    text = re.sub(header_pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

    section_pattern = r"выпускная квалификационная работа.*?москва"
    text = re.sub(section_pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

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
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_stopwords(text, all_stopwords)
    text = remove_multiple_spaces(text)
    text = clean_text(text)
    text = remove_bibliography(text)

    return text

# Функция для лемматизации
def lemmatize_text(text):
    """Лемматизирует текст."""
    morph = pymorphy2.MorphAnalyzer()
    text_lem = [morph.parse(word)[0].normal_form for word in text.split(' ')]
    return ' '.join(text_lem)


def split_text(text, max_len=512, overlap=0):

    """
    Разделение текстов на сегменты 
    """
    tokens = tokenizer.encode(text, truncation=False, add_special_tokens=True)
    segments = []
    for i in range(0, len(tokens), max_len - overlap):
        segment = tokens[i:i + max_len]
        segments.append(segment)
    return segments


def get_embedding_for_text(text, max_len=512, overlap=0):

    """
    Полдучение эмбеддинга для текста
    """
    segments = split_text(text, max_len, overlap)
    embeddings = []
    for segment in segments:
        input_ids = torch.tensor([segment]).to(device)
        attention_mask = torch.ones(input_ids.shape, device=device)
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            embedding = output.last_hidden_state.mean(dim=1).squeeze(0)
            embeddings.append(embedding)
    final_embedding = torch.cat(embeddings, dim=0)
    return final_embedding

def pad_embeddings(embeddings, max_length=None):
    """
    Дополняет эмбеддинги до одинаковой длины.
    """
    if max_length is None:
        max_length = max(embedding.size(0) for embedding in embeddings)

    padded_embeddings = []
    for embedding in embeddings:
        padding_length = max_length - embedding.size(0)
        if padding_length > 0:
            padded_embedding = torch.cat([embedding, torch.zeros(padding_length, device=embedding.device)])
        else:
            padded_embedding = embedding[:max_length]
        padded_embeddings.append(padded_embedding)

    return padded_embeddings


# Функция для обработки нескольких текстов
def process_multiple_texts(texts, max_len=512, overlap=0):
    """
    Функция предобработки и создания эбеддингов для текстов
    """
    all_embeddings = []
    for text in tqdm(texts, desc="Обработка текстов"):
        cleaned_text = preprocess_text(text)
        lemmatized_text = lemmatize_text(cleaned_text)
        embedding = get_embedding_for_text(lemmatized_text, max_len, overlap)
        all_embeddings.append(embedding.detach())
    padded_embeddings = pad_embeddings(all_embeddings)
    return torch.stack(padded_embeddings).cpu().numpy()


def reduce_dimensions_umap(data, n_components=100, n_neighbors=30, min_dist=0.7, metric='cosine'):
    """
    Функция для сокращения размерности данных с использованием UMAP.
    """
  
    reducer = umap.UMAP(n_components=n_components, 
                        n_neighbors=n_neighbors, 
                        min_dist=min_dist, 
                        metric=metric, random_state=42)
   
    reduced_data = reducer.fit_transform(data)
    
    return reduced_data

def cluster_texts(embeddings, n_clusters=6):

    """
    Функция кластеризации.
    """

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    silhouette_avg = silhouette_score(embeddings, labels)
    db_index = davies_bouldin_score(embeddings, labels)
    calinski = calinski_harabasz_score(embeddings, labels)

    #print(f"Индекс силуэта: {silhouette_avg:.2f} (чем ближе к 1, тем лучше)")
    #print(f"Индекс Калински-Харабаза: {calinski:.2f} (чем больше, тем лучше)")
    #print(f"Индекс Дэвиса-Болдина: {db_index:.2f} (чем ближе к 0, тем лучше)")

    return labels

def visualize_clusters(embeddings, labels):

    """
    Визуализация 2d
    """

    reducer = umap.UMAP(n_components=2,n_neighbors=30, min_dist=0.8, init='spectral',  metric='cosine',random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)
    df = pd.DataFrame(embeddings_2d, columns=["x1", "x2"])
    df["label"] = labels

    fig_bert = px.scatter(
        df, x="x1", y="x2", color="label",
        title="Кластеризация текстов с использованием UMAP",
        labels={"x1": "UMAP 1", "x2": "UMAP 2"},
        width=800, height=600,
        color_continuous_scale="Viridis"
    )
    fig_bert.show()

    return fig_bert.to_json()



def visualize_clusters_3d(embeddings, labels):
    
    reducer = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.8, init='spectral', metric='cosine', random_state=42)
    embeddings_3d = reducer.fit_transform(embeddings)
    
    
    df = pd.DataFrame(embeddings_3d, columns=["x1", "x2", "x3"])
    df["label"] = labels

    # Строим 3D график
    fig_bert3d = px.scatter_3d(
        df, x="x1", y="x2", z="x3", color="label",
        title="Кластеризация текстов с использованием UMAP",
        labels={"x1": "UMAP 1", "x2": "UMAP 2", "x3": "UMAP 3"},
        width=800, height=600,
        color_continuous_scale="Viridis"
    )
    #fig_bert3d.show()
    return fig_bert3d.to_json()




def cluster_text_data(texts, n_clusters=6):
    """
    
    """
    embeddings = process_multiple_texts(texts,max_len=512, overlap=0)
    
    reduced_embeddings=reduce_dimensions_umap(embeddings)

    labels = cluster_texts(reduced_embeddings, n_clusters)
    
    two_d_graph_json = visualize_clusters(embeddings,labels )
    three_d_graph_json = visualize_clusters_3d(embeddings,labels )


    return {
        "2d_scatter":two_d_graph_json  ,
        "3d_scatter": three_d_graph_json,
      
    }


if __name__ == "__main__":
    
    texts=['d','d','f']  

    
    result = cluster_text_data(texts)

    # Сохраняем результат в JSON-файл
    with open("cluster_results.json", "w") as f:
        json.dump(result, f, indent=4)

    print("Результаты сохранены в 'cluster_results.json'")
    


