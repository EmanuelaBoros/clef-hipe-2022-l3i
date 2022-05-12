import argparse

ENTITY_TIME_FILE = "/home/cgonzale/clef-hipe-2022-l3i/kb/graphvite_data/wikidata5m_temp_existing.txt"
CONTEXT_FILE = "/home/cgonzale/clef-hipe-2022-l3i/KB-NER/kb/datasets/ajmc/en/kb/en_wk5m_simple/HIPE-2022-v2.1-ajmc-train-en-all.kb"
TARGET_DATE = 1881
INTERVAL = 10
entity_dict = dict()

parser = argparse.ArgumentParser()
parser.add_argument("--cfile", default=CONTEXT_FILE, type=str)
args = parser.parse_args()


with open(ENTITY_TIME_FILE, "r") as ap:
    entity_times = ap.read().split("\n")
for entity_time in entity_times[:-1]:
    e_id, e_time = entity_time.split("\t")
    entity_dict[e_id] = int(e_time)

with open(args.cfile, "r") as ap:
    context_content = ap.read().split("\n")

query_id = 0

"""
context_size = 11
for i in range(len(context_content)):
    a = context_content[i*context_size:i*context_size+context_size]
    if len(a) > 0:
        query = a[0]
        contexts = a[1:]
        for context in contexts:
            e_text = context.split("\t")[0]
            e_id = context.split("\t")[2]
            if e_id in entity_dict:
                print(query)
                print(f"\t{e_id}\t{entity_dict[e_id]}\t{e_text}")
"""

out_text = ""
for line in context_content:
    if len(line.split("\t")) > 2:
        e_id = line.split("\t")[2]
        if e_id in entity_dict:
            if entity_dict[e_id] >= TARGET_DATE+INTERVAL or entity_dict[e_id] <= TARGET_DATE-INTERVAL:
                continue
        out_text += line

CONTEXT_FILE_OUT = args.cfile.replace(".kb", "_time.kb")
with open(CONTEXT_FILE_OUT, "w") as ap:
    ap.write(out_text)

