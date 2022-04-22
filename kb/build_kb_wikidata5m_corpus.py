from typing import List, Iterable, Dict
import json
import time
from tqdm import tqdm
import requests
import argparse
from sentence_transformers import SentenceTransformer


parser = argparse.ArgumentParser()
parser.add_argument("--lan", default="en", type=str)
parser.add_argument("--model", default="minilm", type=str)
args = parser.parse_args()

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

LAN = args.lan
ES = "http://localhost:9200"
INDEX_NAME = f"{LAN}wk5m_corpus{args.model}"
HEADERS = {'Accept': 'application/json', 'Content-type': 'application/json'}
WIKIDATA5M_CORPUS = "./graphvite_data/wikidata5m_text.txt"


analyzer = 'standard'
search_analyzer = 'standard'

CONFIG = {
    "settings": {
        "number_of_shards": 1
    },
    "mappings": {
        "properties": {
            "entity_id": {"type": "text", "index": True},
            "entity_text": {"type": "text", "index": False},
            "entity_text_vector": {"type": "dense_vector",
                                   "dims": MODEL[args.model]["dim"],
                                   "index": True,
                                   "similarity": "cosine"
                                   }
        }
    }
}

embedder = SentenceTransformer(MODEL[args.model]["name"])


def batch_iter(size=10000) -> Iterable[List[Dict]]:
    batch = list()
    with open(WIKIDATA5M_CORPUS, mode="r") as file:
        for i, line in enumerate(file):
            entity_id = line.split("\t")[0]
            entity_text = "".join(line.split("\t")[1:])
            entity_vector = embedder.encode([entity_text])
            batch.append('{"index":{}}')
            batch.append(json.dumps(
                {'entity_id': entity_id,
                 "entity_text": entity_text,
                 'entity_text_vector': entity_vector[0].tolist()
                 },
                ensure_ascii=False
            ))
            if len(batch) >= size:
                yield batch
                batch.clear()
        if len(batch) > 0:
            yield batch


def add_copus():

    """
    res = requests.put(f"{ES}/{INDEX_NAME}", json=CONFIG, headers=HEADERS)
    if res.status_code != 200:
        print(json.dumps(res.json(), indent=2))
        raise RuntimeError("failed to create index mapping!")
    """
    def add_batch(batch: List):
        content = "\n".join(batch) + "\n"
        res = requests.post(url, data=content.encode("utf-8"), headers=HEADERS)
        if res.status_code != 200:
            failures.append(i)

    url = f"{ES}/{INDEX_NAME}/_bulk"
    batch_size = 10000
    failures = list()
    timer = time.time()
    for i, batch in tqdm(enumerate(batch_iter(batch_size))):
        add_batch(batch)
        if i % 100 == 0:
            timer = time.time() - timer
            print(i, ", time", timer)
            timer = time.time()

    if len(failures) > 0:
        with open("fail", mode='w') as file:
            file.write('\n'.join(failures) + '\n')
            print('write', file.name)
    else:
        print("success")

def main():
    r = requests.get(ES, headers=HEADERS)
    assert r.status_code == 200
    add_copus()
    return


if __name__ == "__main__":
    main()

