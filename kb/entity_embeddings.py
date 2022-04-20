import pickle


def load_alias(alias_file):
    alias2object = {}
    ambiguous = set()
    with open(alias_file, "r") as fin:
        for line in fin:
            tokens = line.strip().split("\t")
            object = tokens[0]
            for alias in tokens[1:]:
                if alias in alias2object and alias2object[alias] != object:
                    ambiguous.add(alias)
                alias2object[alias] = object
        for alias in ambiguous:
            alias2object.pop(alias)
    return alias2object


with open("./graphvite_data/rotate_wikidata5m.pkl", "rb") as fin:
    model = pickle.load(fin)

entity2id = model.graph.entity2id
relation2id = model.graph.relation2id
entity_embeddings = model.solver.entity_embeddings
relation_embeddings = model.solver.relation_embeddings

alias2entity = load_alias("./graphvite_data/wikidata5m_entity.txt")
alias2relation = load_alias("./graphvite_data/wikidata5m_relation.txt")

print(entity_embeddings[entity2id[alias2entity["machine learning"]]])
print(relation_embeddings[relation2id[alias2relation["field of work"]]])