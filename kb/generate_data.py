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
ST_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'
ES = "http://localhost:9200"
HEADERS = {"content-type": "application/json;charset=UTF-8"}
LAN = "fr"
INDEX_NAME = "wiki_v1"

print(f"Loading Sentence transformer {ST_MODEL}...")
embedder = SentenceTransformer(ST_MODEL)
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
                        type=str,
                        default="/home/cgonzale/clef-hipe-2022-l3i/KB-NER/kb/datasets/hipe2020/fr/HIPE-2022-v2.1-hipe2020-dev-fr.tsv")
    parser.add_argument("-l", "--lan",
                        default=LAN,
                        type=str)

    return parser.parse_args()


def search_batch(queries, lan, k):
    vectors = vectorize_batch(queries).tolist()
    responses = []
    print("Searching batch")
    s_t = time.time()
    for i, sentence in tqdm(enumerate(queries)):
        responses.append(search_sentence(vectors[i], lan, k))
    e_t = time.time()
    print(f"**TIME search batch: {e_t - s_t} [s]")

    return queries, responses


def vectorize_batch(sentences):
    s_t = time.time()
    vectors = embedder.encode(sentences)
    e_t = time.time()
    print(f"**TIME vectorize batch: {e_t - s_t} [s]")

    return vectors


def search_sentence(vector, lan, k=10):

    url = f"{ES}/{lan+INDEX_NAME}/_knn_search"

    query = {
        "knn": {
            "field": "text_vector",
            "query_vector": vector,
            "k": k,
            "num_candidates": 100
        },
        "fields": [
            "text",
            "paragraph",
            "title",
            "url"
        ]
    }
    query_txt = json.dumps(query, ensure_ascii=False) + "\n"
    s_t = time.time()
    response = requests.get(url, data=query_txt.encode("utf-8"), headers=HEADERS)
    e_t = time.time()
    print(f"**TIME: {e_t-s_t} [s]")

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


def write_kb(queries, responses, out_file):

    with open(out_file, "a") as f:
        for i, query in enumerate(queries):
            f.write(f"Q:\t{query}")
            for hit in responses[i]['hits']['hits']:
                score = hit['_score']
                fields = hit['fields']
                title = fields['title'][0]
                text = fields['text'][0]
                url = fields['url'][0]
                paragraph = fields['paragraph'][0]
                paragraph = parse_paragraph(paragraph)
                register = f"{text}\t{score}\t{title}\t{paragraph}\n"
                #register = f"{text}\t{paragraph}\t{title}\t{score}\t{url}\n"
                f.write(register)


def main():
    args = parse_arguments()
    lan = args.lan
    with open(args.in_file, 'r') as f:
        lines = f.readlines()

    out_file = args.in_file.replace(".tsv", ".kb")
    sentences = utils.process_sentences(lines)
    batch_size = 10
    k = 10
    for i, queries in tqdm(enumerate(batch_iter(sentences, batch_size))):
        print(f"{i*batch_size}/{len(sentences)}")
        queries, responses = search_batch(queries, lan, k)
        write_kb(queries, responses, out_file)


if __name__ == '__main__':
    """
    Starts the whole app from the command line
    """

    main()



