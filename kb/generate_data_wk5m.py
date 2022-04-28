import csv
import os
import re
from tqdm import tqdm
import json
import requests
from sentence_transformers import SentenceTransformer
import time
import argparse
import utils
# model is in ["minilm", "mpnet"]
modele = "minilm"
MODEL = {
    "minilm": {
        "name": "paraphrase-multilingual-MiniLM-L12-v2",
        "dim": 384
    },
    "mpnet": {
        "name": "paraphrase-multilingual-mpnet-base-v2",
        "dim": 768
    }
}

ES = "http://localhost:9200"
HEADERS = {"content-type": "application/json;charset=UTF-8"}
LAN = "en"
INDEX_NAME_CORPUS = "wk5m_corpus"
INDEX_NAME_GRAPH = "wk5m_kgemb"

print(f"Loading Sentence transformer {MODEL[modele]}...")
embedder = SentenceTransformer(MODEL[modele]["name"])
print("Sentence transformer loaded !")

entity_dict = dict()
with open("graphvite_data/wikidata5m_text.txt", "r") as file:
    for i, line in enumerate(file):
        e_id = line.split("\t")[0]
        e_context = line.split("\t")[-1]
        entity_dict[e_id] = e_context

def parse_arguments():
    """Returns a command line parser

    Returns
    ----------
    argparse.Namespace

    """

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--in_file",
                        dest="in_file",
                        help="""Path to tsv file.""",
                        type=str,
                        default="/home/cgonzale/clef-hipe-2022-l3i/KB-NER/kb/datasets/hipe2020/en/HIPE-2022-v2.1-hipe2020-dev-en.tsv")
    parser.add_argument("-l", "--lan",
                        default=LAN,
                        type=str)
    parser.add_argument("-m", "--modele",
                        default="minilm",
                        type=str)

    return parser.parse_args()

def search_in_graph(sentence, entity_ids, lan, k=10):

    founded_entity = False
    index = 0
    while not founded_entity and index < len(entity_ids):
        url = f"{ES}/{lan + INDEX_NAME_GRAPH}/_search"

        query = {
            "size": 1,
            "query": {
                "bool": {
                    "should": [{"match": {"entity_id": entity_ids[index]}}]
                }
            }
        }
        query_txt = json.dumps(query, ensure_ascii=False) + "\n"
        response = requests.get(url, data=query_txt.encode("utf-8"), headers=HEADERS).json()
        if len(response["hits"]["hits"]) > 0:
            founded_entity = True
        else:
            index += 1

    if len(response["hits"]["hits"]) == 0:
        print(sentence)
        print(entity_ids)
        exit()
    entity_text_vector = response["hits"]["hits"][0]["_source"]["entity_vector"]

    url = f"{ES}/{lan + INDEX_NAME_GRAPH}/_knn_search"
    query = {
        "knn": {
            "field": "entity_vector",
            "query_vector": entity_text_vector,
            "k": k,
            "num_candidates": 100
        },
        "fields": [
            "entity_id"
        ]
    }
    query_txt = json.dumps(query, ensure_ascii=False) + "\n"
    response = requests.get(url, data=query_txt.encode("utf-8"), headers=HEADERS)

    return response.json()


def vectorize_batch(sentences):
    s_t = time.time()
    vectors = embedder.encode(sentences)
    e_t = time.time()
    print(f"**TIME vectorize batch: {e_t - s_t} [s]")

    return vectors


def search_sentence(vector, lan, model, k=10):

    url = f"{ES}/{lan + INDEX_NAME_CORPUS + model}/_knn_search"

    query = {
        "knn": {
            "field": "entity_text_vector",
            "query_vector": vector,
            "k": k,
            "num_candidates": 100
        },
        "fields": [
            "entity_id",
            "entity_text",
            "entity_text_vector"
        ]
    }
    query_txt = json.dumps(query, ensure_ascii=False) + "\n"

    response = requests.get(url, data=query_txt.encode("utf-8"), headers=HEADERS)

    return response.json()


def retrieve_entity_texts(response):

    tmp_contexts = []

    for hit in response['hits']['hits']:
        score = hit['_score']
        fields = hit['fields']
        entity_id = fields['entity_id'][0]
        entity_text = entity_dict.get(entity_id)
        if entity_text is None:
            continue
        tmp_contexts.append({
            'entity_id': entity_id,
            'score': score,
            'entity_text': entity_text
        })

    return tmp_contexts


def search_batch(queries, lan, model, k):
    vectors = vectorize_batch(queries).tolist()
    contexts = []
    print("Searching batch")
    s_t = time.time()
    for i, sentence in tqdm(enumerate(queries)):
        response = search_sentence(vectors[i], lan, model, 25)
        #score = response['hits']['hits'][0]['_score']
        #entity_id = response['hits']['hits'][0]['fields']['entity_id'][0]
        entity_ids = []
        for hit in response['hits']['hits']:
            entity_ids.append(hit["fields"]["entity_id"][0])

        response = search_in_graph(sentence, entity_ids, lan, 25)
        tmp_contexts = retrieve_entity_texts(response)[:k]
        #tmp_contexts[0]["score"] = score
        contexts.append(tmp_contexts)
    e_t = time.time()
    print(f"**TIME search batch: {e_t - s_t} [s]")

    return queries, contexts


def batch_iter(sentences, size=50):
    batch = list()
    for i, sentence in enumerate(sentences):
        batch.append(sentence)
        if len(batch) >= size:
            yield batch
            batch.clear()
    if len(batch) > 0:
        yield batch


def parse_paragraph(paragraph):

    return re.sub(r"<e:([\w'’\-.:|() ^>]+)>([\w’\-.:|()' ]+)</e>", "<e> \g<1> </e>", paragraph)


def write_kb(queries, contexts, out_file):

    with open(out_file, "a") as f:
        for i, query in enumerate(queries):
            f.write(f"Q:\t{query}")
            for context in contexts[i]:
                score = context["score"]
                entity_text = context["entity_text"].replace("\n","")
                entity_id = context["entity_id"]
                sentence = entity_text.split(".")[0]
                #register = f"{sentence}\t{score}\t{entity_id}\t{entity_text}\n"
                #simple context just first sentence
                register = f"{sentence}\t{score}\t{entity_id}\t{sentence}\n"
                f.write(register)


def main():
    args = parse_arguments()
    lan = args.lan
    model = args.modele
    with open(args.in_file, 'r') as f:
        lines = f.readlines()

    out_file = args.in_file.replace(".tsv", ".kbg")
    sentences = utils.process_sentences(lines)
    batch_size = 25
    k = 10
    for i, queries in tqdm(enumerate(batch_iter(sentences, batch_size))):
        print(f"{i*batch_size}/{len(sentences)}")
        queries, contexts = search_batch(queries, lan, model, k)
        write_kb(queries, contexts, out_file)


if __name__ == '__main__':
    """
    Starts the whole app from the command line
    """

    main()



