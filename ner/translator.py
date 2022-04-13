# -*- coding: utf-8 -*-

from pygoogletranslation import Translator
import numpy as np

def back_translate(sequence, PROB = 1):
    languages = ['en', 'de']#, 'th', 'tr', 'ur', 'ru', 'bg', 'de', 'ar', 'zh-cn', 'hi',
                 #'sw', 'vi', 'es', 'el']
    
    #instantiate translator
    translator = Translator()#service_urls=['translate.googleapis.com'])
    
    #store original language so we can convert back
    #import pdb;pdb.set_trace()
    org_lang = 'en'#translator.detect(sequence).lang
    
    #randomly choose language to translate sequence to  
    random_lang = 'de'#np.random.choice([lang for lang in languages if lang is not org_lang])
    
#    if org_lang in languages:
    #translate to new language and back to original
    translated = translator.translate(sequence, dest=random_lang).text
    #translate back to original language
    translated_back = translator.translate(translated, dest = org_lang).text

    #apply with certain probability
    if np.random.uniform(0, 1) <= PROB:
        output_sequence = translated_back
    else:
        output_sequence = sequence
            
    #if detected language not in our list of languages, do nothing
#    else:
#        output_sequence = sequence
    
    return output_sequence

#for i in range(5):
#    output = back_translate('I genuinely have no idea what the output of this sequence of words will be')
#    print(output)
    
DIR = '/home/eboros/projects/DATA/conll_2003/'
import spacy, os

#nlp = spacy.load("en_core_web_sm")
#nlp.add_pipe(nlp.create_pipe('sentencizer'))


true_tags = []
true_words = []
tags = []
idx = 0
docs = []
with open(os.path.join(DIR, 'original/eng.train'), 'r') as f:
    
    docs.append(idx)
    lines = f.readlines()
    
    all_sentences = []
    sentence = []
    tags = []
    nn = []
    for line in lines:
        if len(line.strip()) > 1:
#            import pdb;pdb.set_trace()
            if 'DOCSTART' not in line:
                pred = line.split('\t')[3]
                
                sentence.append(line.split('\t')[0])
                tags.append(line.split('\t')[-1].strip())
                nn.append(line.split('\t')[1])
#                if '.' != line.split('\t')[0]:
#                    true_tags.append(pred.strip())
#                    true_words.append(line.split('\t')[0])
#                    tags.append(line.strip())
#                    idx += 1
            else:
                sentence.append(line)
                tags.append(None)
                nn.append(None)
        else:
            all_sentences.append((sentence, tags, nn))
            all_sentences.append(('', None, None))
            sentence = []
            tags = []
            nn = []

from nltk.tokenize import word_tokenize  
#import pdb;pdb.set_trace()

print(all_sentences[0])
with open(os.path.join(DIR, 'original/eng.train.translated_german'), 'a') as f:
    for sentence in all_sentences:
        if len(sentence[0]) > 0:
            
#            import pdb;pdb.set_trace()
            
            if 'DOCSTART' in sentence[0][0]:
                f.write(sentence[0][0] + '\n')
            
            else:
                print(sentence[0])
                
                try:
                    output = word_tokenize(back_translate(' '.join(sentence[0])))
                    tags = sentence[1] + ['O'] * len(output)
                    nn = sentence[2] + ['O'] * len(output)
                    
                    print(output)
                    print(len(output), len(tags))
                    for word, tag, n in zip(output, tags, nn):
        #            import pdb;pdb.set_trace()
                        f.write(word + '\t' + n + '\t' + tag + '\n')
                except:
                    continue
        else:
            f.write('\n')
            f.flush()
    
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                