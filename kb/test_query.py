import json
import requests
from sentence_transformers import SentenceTransformer
import time

ES = "http://localhost:9200"
INDEX_NAME = "frwiki_v1"
HEADERS = {"content-type": "application/json;charset=UTF-8"}
LAN = "fr"

url = f"{ES}/{INDEX_NAME}/_knn_search"
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
text = ["La Chambre aborde la discussion générale le projet d'impôt sur le revenu.",
        "La Chambre aborde la discussion générale le projet d'impôt sur le revenu.",
        "La Chambre aborde la discussion générale le projet d'impôt sur le revenu.",
        "La Chambre aborde la discussion générale le projet d'impôt sur le revenu.",
        "La Chambre aborde la discussion générale le projet d'impôt sur le revenu.",
        "La Chambre aborde la discussion générale le projet d'impôt sur le revenu.",
        "La Chambre aborde la discussion générale le projet d'impôt sur le revenu.",
        "La Chambre aborde la discussion générale le projet d'impôt sur le revenu.",
        "La Chambre aborde la discussion générale le projet d'impôt sur le revenu.",
        "La Chambre aborde la discussion générale le projet d'impôt sur le revenu.",
        "La Chambre aborde la discussion générale le projet d'impôt sur le revenu.",
        "La Chambre aborde la discussion générale le projet d'impôt sur le revenu.",
        "La Chambre aborde la discussion générale le projet d'impôt sur le revenu."]

s_t = time.time()
vector = embedder.encode(text)
e_t = time.time()
print(f"**TIME emb: {e_t-s_t} [s]")

exit()
print("ok")
query = {
    "knn": {
        "field": "text_vector",
        "query_vector": vector[0].tolist(),
        "k": 5,
        "num_candidates": 100
    },
    "fields": [
        "text",
        "paragraph",
        "title",
        "url"
    ]
}

row = json.dumps(query, ensure_ascii=False) + "\n"
s_t = time.time()
response = requests.get(url, data=row.encode("utf-8"), headers=HEADERS)
e_t = time.time()
print(f"**TIME: {e_t-s_t} [s]")
print(f"Query: {text}")
a = response.json()
for hit in response.json()['hits']['hits']:
    print(f"Score: {hit['_score']}")
    fields = hit['fields']
    print(f"Title: {fields['title']}")
    print(f"Text: {fields['text']}")
    print(f"Paragraph: {fields['paragraph']}")
    print("------")

"""
query = {
    "size": 5,
    "query": {
        "script_score": {
            "query": {
                "match_all": {}
            },
            "script": {
                "source": "cosineSimilarity(params.query_vector, doc['text_vector']) + 1.0",
                "params": {"query_vector": vector[0].tolist()}
            }
        }
    },
    "fields": [
        "text",
        "paragraph",
        "title",
        "url"
    ],
    "_source": False
}
"""
"""
query = {
    "size": 5,
    "query": {
        "bool": {}
    },
    "fields": [
        "text",
        "paragraph",
        "title",
        "url"
    ],
    "_source": False
}
query["query"]["bool"]["should"] = [{ "match": { "text":  text[0] }}]
"""
"""
query = {
    "size": 2,
    "query": {
        "bool": {}
    },
    "highlight": {
        "pre_tags": ["<hit>"],
        "post_tags": ["</hit>"],
        "fields": {
            "text": {},
            "title": {}
        }
    }
}
query["query"]["bool"]["should"] = [{ "match": { "text":  text[0] }}]
"""

