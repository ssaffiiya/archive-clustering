import io
import sys
import json
import numpy as np
from sklearn.cluster import KMeans
import umap
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation

# Уменьшение размерности с помощью UMAP
def cluster_texts(embeddings, n_clusters=6):
    """
    Функция кластеризации.
    """
    ap = AffinityPropagation(damping=0.6, preference=None, random_state=42)
    labels = ap.fit_predict(embeddings)
    return labels

def visualize_clusters(embeddings, labels):
    """
    Для 2D визуализации
    """
    reducer = umap.UMAP(n_components=2,n_neighbors=200,min_dist=0.8, metric='cosine',init='pca',random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)
    return embeddings_2d

def get_cluster_data(input_data, n_clusters=6):

    embeddings = np.array([item['textEmbedding'] for item in input_data])

    labels = cluster_texts(embeddings, n_clusters)
    embeddings_2d = visualize_clusters(embeddings, labels)
   
    scatter_data = [
    {
        "x": emb[0].tolist() if hasattr(emb[0], 'tolist') else emb[0],
        "y": emb[1].tolist() if hasattr(emb[1], 'tolist') else emb[1],
        "label": lbl.tolist() if hasattr(lbl, 'tolist') else lbl,
        "title": input_data[i]["title"],  
         "author": input_data[i]["author"]["fullname"],
         "advisor": input_data[i]["advisor"]["fullname"]
    }
    for i, (emb, lbl) in enumerate(zip(embeddings_2d, labels))
]
    

    return scatter_data





if name == "main": 
    import sys 
    import codecs
    sys.stdin = codecs.getwriter('utf-8')(sys.stdin.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
    request = input()
    input_texts = json.loads(request)
    result = get_cluster_data(input_texts, n_clusters=6)

    print(json.dumps(result))