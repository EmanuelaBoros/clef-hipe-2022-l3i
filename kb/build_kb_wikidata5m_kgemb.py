from typing import List, Iterable, Dict
import json
import time
from tqdm import tqdm
import requests
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--lan", default="en", type=str)
args = parser.parse_args()

LAN = args.lan
ES = "http://localhost:9200"
INDEX_NAME = f"{LAN}wk5m_kgemb"
HEADERS = {'Accept': 'application/json', 'Content-type': 'application/json'}
WIKIDATA5M_MODEL = "./graphvite_data/rotate_wikidata5m.pkl"


analyzer = 'standard'
search_analyzer = 'standard'

CONFIG = {
    "settings": {
        "number_of_shards": 1
    },
    "mappings": {
        "properties": {
            "entity_id": {"type": "text", "analyzer": analyzer, "search_analyzer": search_analyzer},
            "entity_vector": {"type": "dense_vector",
                                   "dims": 512,
                                   "index": True,
                                   "similarity": "cosine"
                              }
        }
    }
}


def batch_iter(size=10000) -> Iterable[List[Dict]]:
    batch = list()
    with open(WIKIDATA5M_MODEL, mode="rb") as file:
        model = pickle.load(file)
        id2entity = model.graph.id2entity
        for i, entity_embeddings in enumerate(model.solver.entity_embeddings):
            entity_id = id2entity[i]
            entity_vector = entity_embeddings
            batch.append('{"index":{}}')
            batch.append(json.dumps(
                {'entity_id': entity_id,
                 'entity_vector': entity_vector.tolist()
                 },
                ensure_ascii=False
            ))
            if len(batch) >= size:
                yield batch
                batch.clear()
        if len(batch) > 0:
            yield batch


def add_copus():


    res = requests.put(f"{ES}/{INDEX_NAME}", json=CONFIG, headers=HEADERS)
    if res.status_code != 200:
        print(json.dumps(res.json(), indent=2))
        raise RuntimeError("failed to create index mapping!")


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

