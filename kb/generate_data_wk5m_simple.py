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

ENTITY_TIME_FILE = "/home/cgonzale/clef-hipe-2022-l3i/kb/graphvite_data/wikidata5m_temp_existing.txt"
INTERVAL = 10
entity_dict = dict()
with open(ENTITY_TIME_FILE, "r") as ap:
    entity_times = ap.read().split("\n")
for entity_time in entity_times[:-1]:
    e_id, e_time = entity_time.split("\t")
    entity_dict[e_id] = int(e_time)


print(f"Loading Sentence transformer {MODEL[modele]}...")
embedder = SentenceTransformer(MODEL[modele]["name"])
print("Sentence transformer loaded !")


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
                        type=str)
    parser.add_argument("-l", "--lan",
                        default=LAN,
                        type=str)

    return parser.parse_args()


def search_batch(queries, lan, k):
    model = modele
    queries, years = list(zip(*queries))
    vectors = vectorize_batch(queries).tolist()
    responses = []
    print("Searching batch")
    s_t = time.time()
    for i, sentence in tqdm(enumerate(queries)):
        responses.append(search_sentence(vectors[i], lan, model, k))
    e_t = time.time()
    print(f"**TIME search batch: {e_t - s_t} [s]")

    return queries, years, responses


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


def write_kb(queries, years, responses, out_file):

    with open(out_file, "a") as f:
        for i, query in enumerate(queries):
            f.write(f"Q:\t{query}")
            count = 0
            for hit in responses[i]['hits']['hits']:
                score = hit['_score']
                fields = hit['fields']
                entity_id = fields['entity_id'][0]
                entity_text = fields['entity_text'][0].replace("\n","")
                sentence = entity_text.split(".")[0]
                #register = f"{sentence}\t{score}\t{entity_id}\t{entity_text}\n"
                #simple context just first sentence
                register = f"{sentence}\t{score}\t{entity_id}\t{sentence}\n"
                if entity_id in entity_dict:
                    if entity_dict[entity_id] >= years[i] + INTERVAL or entity_dict[entity_id] <= years[i] - INTERVAL:
                        print(register)
                        continue
                f.write(register)
                count += 1
                if count == 10:
                    break


def main():
    args = parse_arguments()
    lan = args.lan

    with open(args.in_file, 'r') as f:
        lines = f.readlines()

    out_file = args.in_file.replace(".tsv", ".kb")
    sentences = utils.process_sentences_time(lines)
    batch_size = 10
    k = 25
    for i, queries in tqdm(enumerate(batch_iter(sentences, batch_size))):
        print(f"{i*batch_size}/{len(sentences)}")
        queries, years, responses = search_batch(queries, lan, k)
        write_kb(queries, years, responses, out_file)


if __name__ == '__main__':
    """
    Starts the whole app from the command line
    """

    main()



