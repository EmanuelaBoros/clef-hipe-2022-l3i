# -*- coding: utf-8 -*-

import argparse
import os
from multiprocessing import Pool
import spacy
import re
from spacy_langdetect import LanguageDetector
from nltk.tokenize import sent_tokenize
nlp = spacy.load('xx_ent_wiki_sm')
nlp.add_pipe(nlp.create_pipe('sentencizer'))
nlp.add_pipe(LanguageDetector(), name="language_detector", last=True)

nlp_process = spacy.load('xx_ent_wiki_sm')
nlp_process.add_pipe(nlp_process.create_pipe('sentencizer'))

files = ['illustrierte_kronen_zeitung_krz19210923.txt']

def tokenize(file_path):

    with open(file_path, 'r') as f:
        text = f.read().strip()

    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    text = re.sub(' +', ' ', text)
    texts = sent_tokenize(text)
    doc = nlp(text[:10000])
    language = doc._.language['language']
#    if not os.path.exists(os.path.join(output_dir, language, file_path.split('/')[-1])):
#    print(file_path)
#    print(language, len(texts))
#    print(file_path)
#    print(os.path.join(output_dir, language, file_path.split('/')[-1]))

    if not os.path.exists(os.path.join(output_dir, language)):
        os.mkdir(os.path.join(output_dir, language))
    
    del doc
    
    with open(os.path.join(output_dir, language, file_path.split('/')[-1]), 'w') as f:
        f.write('TOKEN	NE-COARSE-LIT	NE-COARSE-METO	NE-FINE-LIT	NE-FINE-METO	NE-FINE-COMP	NE-NESTED	NEL-LIT	NEL-METO	MISC\n')

        for sentence in texts:
            sentence = sentence.strip()
            sentence = sentence.replace("'", " ' ")
            sentence = sentence.replace('"', ' " ')
            sentence = sentence.replace('-', ' - ')
            sentence = sentence.replace('.', ' . ')
            if len(sentence) > 0:
                sentence = sentence.replace('\n', ' ')
                sentence = sentence.replace('\t', ' ')
                sentence = re.sub(' +', ' ', sentence)
                sentence = re.sub(' +', ' ', sentence)
                doc_process = nlp_process(sentence)
                for token in doc_process:
                    if len(token.text.strip()) > 0:
                        #                    print(token.text + '\t' + 'O	O	O	O	O	_	_	_	_')
                        token = token.text
                        f.write(token + '\t' + 'O	O	O	O	O	_	_	_	_\n')
                f.write('\n')
    del texts

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--output_dir', type=str,
                        default='../newseye_texts_10_07_2020_processed/')
    parser.add_argument('--processes', type=int, default=40)
    args = parser.parse_args()

    dataset = args.dataset
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

#    files = [os.path.join(dataset, f)
#             for f in os.listdir(dataset) if f.endswith(".txt")]
    files = [os.path.join(dataset, f) for f in files]
##    files = ['../' + str(f) for f in lines]
#    pool = Pool(args.processes)
    from tqdm import tqdm
##    results = pool.map(tokenize, files)
#    tqdm.tqdm(pool.imap(tokenize, files), total=len(files))
#    pool.close()
    import time
    def _foo(my_number):
       square = my_number * my_number
       time.sleep(1)
       return square 

    with Pool(processes=args.processes) as p:
#        max_ = 30
        with tqdm(total=len(files)) as pbar:
            for i, _ in enumerate(p.imap_unordered(tokenize, files)):
                pbar.update()




















