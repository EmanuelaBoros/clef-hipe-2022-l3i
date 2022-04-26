# -*- coding: utf-8 -*-

with open('predictions_dev.tsv', 'r') as f:
    lines = f.readlines()
    
with open('predictions_dev_post.tsv', 'w') as g:
    for line in lines:
        tokens = line.split('\t')
        if len(tokens) > 4:
            entity_types = tokens[1:6]
            if entity_types[0] == 'O':
                entity_type = [entity_type for entity_type in entity_types[1:] if entity_type not in ['O']]
                if len(entity_type) > 0:
                    print(entity_type)
                    entity_type = entity_type[0].split('.')[0]
                    print(entity_type)
#                    tokens[1] = entity_type
                    tokens[1:6] = ['O'] * len(tokens[1:6])
        
        g.write('\t'.join(tokens))
        
        